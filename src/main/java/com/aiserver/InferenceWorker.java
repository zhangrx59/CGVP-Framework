package com.aiserver;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.connection.stream.*;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;

@Component
public class InferenceWorker {

    private final RedisTemplate<String, Object> redisTemplate;
    private final InferenceJobRepo jobRepo;
    private final InferenceResultRepo resultRepo;
    private final CaseRepo caseRepo;
    private final CaseImageRepo imageRepo;
    private final AiClient aiClient;

    private final String streamKey;
    private final String group;
    private final String consumer;
    private final long pollMs;

    public InferenceWorker(
            RedisTemplate<String, Object> redisTemplate,
            InferenceJobRepo jobRepo,
            InferenceResultRepo resultRepo,
            CaseRepo caseRepo,
            CaseImageRepo imageRepo,
            AiClient aiClient,
            @Value("${app.queue.stream}") String streamKey,
            @Value("${app.queue.group}") String group,
            @Value("${app.queue.consumer}") String consumer,
            @Value("${app.queue.poll-ms}") long pollMs
    ) {
        this.redisTemplate = redisTemplate;
        this.jobRepo = jobRepo;
        this.resultRepo = resultRepo;
        this.caseRepo = caseRepo;
        this.imageRepo = imageRepo;
        this.aiClient = aiClient;
        this.streamKey = streamKey;
        this.group = group;
        this.consumer = consumer;
        this.pollMs = pollMs;

        ensureGroup();
    }

    /** 每秒拉一次队列（后面你要更“生产”，我们会改成独立 worker 服务） */
    @Scheduled(fixedDelayString = "${app.queue.poll-ms}")
    public void poll() {
        try {
            List<MapRecord<String, Object, Object>> records = redisTemplate.opsForStream().read(
                    Consumer.from(group, consumer),
                    StreamReadOptions.empty().count(10).block(Duration.ofMillis(200)),
                    StreamOffset.create(streamKey, ReadOffset.lastConsumed())
            );

            if (records == null || records.isEmpty()) return;

            for (MapRecord<String, Object, Object> r : records) {
                try {
                    String jobIdStr = String.valueOf(r.getValue().get("jobId"));
                    Long jobId = Long.valueOf(jobIdStr);

                    processJob(jobId);

                    // 成功才 ACK
                    redisTemplate.opsForStream().acknowledge(streamKey, group, r.getId());
                } catch (Exception e) {
                    // 不 ACK，让它留在 pending（后续可以做自动 reclaim/重试）
                    System.err.println("[WORKER] fail recordId=" + r.getId() + " err=" + e.getMessage());
                }
            }
        } catch (Exception e) {
            System.err.println("[WORKER] poll error: " + e.getMessage());
        }
    }

    private void processJob(Long jobId) {
        InferenceJob job = jobRepo.findById(jobId)
                .orElseThrow(() -> new IllegalArgumentException("job not found"));

        // 只处理 QUEUED（避免重复）
        if (!"QUEUED".equals(job.getStatus())) {
            return;
        }

        // 抢占：把 QUEUED -> RUNNING（最简版：直接改；后面我们可以做更严格的 CAS 更新）
        job.setStatus("RUNNING");
        job.setAttemptCount(job.getAttemptCount() + 1);
        job.setStartedAt(Instant.now());
        job.setLastError(null);
        jobRepo.save(job);

        try {
            Long caseId = job.getCaseId();
            Case c = caseRepo.findById(caseId)
                    .orElseThrow(() -> new IllegalArgumentException("case not found"));

            List<CaseImage> images = imageRepo.findByCaseId(caseId);
            if (images.isEmpty()) throw new IllegalArgumentException("no image uploaded");

            var imagePaths = images.stream().map(CaseImage::getFilePath).toList();

            var caseData = new java.util.HashMap<String, Object>();
            caseData.put("patientName", c.getPatientName());
            caseData.put("patientSex", c.getPatientSex());
            caseData.put("patientAge", c.getPatientAge());
            caseData.put("chiefComplaint", c.getChiefComplaint());
            caseData.put("history", c.getHistory());

            AiDtos.InferReq req = new AiDtos.InferReq(job.getId(), caseId, imagePaths, caseData);

            AiDtos.InferResp resp = aiClient.infer(req);

            InferenceResult r = new InferenceResult();
            r.setJobId(job.getId());
            r.setCaseId(caseId);
            r.setResultJson(JsonUtil.toJson(resp));
            resultRepo.save(r);

            job.setStatus("SUCCEEDED");
            job.setFinishedAt(Instant.now());
            jobRepo.save(job);

            System.out.println("[WORKER] job " + jobId + " SUCCEEDED");
        } catch (Exception e) {
            job.setStatus("FAILED");
            job.setFinishedAt(Instant.now());
            job.setLastError(e.getClass().getSimpleName() + ": " + e.getMessage());
            jobRepo.save(job);

            System.err.println("[WORKER] job " + jobId + " FAILED: " + job.getLastError());

            // 失败不抛出，让 record 留在 pending（先这样，下一步我们做重试/转死信）
            throw e;
        }
    }

    private void ensureGroup() {
        try {
            // 如果 stream 不存在，先创建一个空 stream（XADD）
            redisTemplate.opsForStream().add(
                    MapRecord.create(streamKey, Map.of("init", "1"))
            );

            // 创建 group（若已存在会抛异常，忽略即可）
            redisTemplate.opsForStream().createGroup(streamKey, ReadOffset.latest(), group);
            System.out.println("[WORKER] created consumer group: " + group);
        } catch (Exception ignore) {
            // group 已存在 or 其他重复创建情况
        }
    }
}
