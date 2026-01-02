package com.aiserver;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.connection.stream.*;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.time.Instant;
import java.util.*;

@Component
public class InferenceWorker {

    private final RedisTemplate<String, Object> redisTemplate;
    private final InferenceJobRepo jobRepo;
    private final InferenceResultRepo resultRepo;
    private final CaseRepo caseRepo;
    private final CaseImageRepo imageRepo;
    private final InferApiClient inferApiClient;

    private final String streamKey;
    private final String group;
    private final String consumer;
    private final long pollMs;

    // 你要求严格字段（中文 key）必须齐全
    private static final List<String> REQUIRED_FIELDS = List.of(
            "年龄","性别","父籍贯","母籍贯","是否吸烟","是否饮酒","农药","皮肤癌病史","癌症病史",
            "生活环境是否有自来水","生活环境是否有下水道","皮肤光型","区域",
            "直径1","直径2","瘙痒","是否长大","疼痛","形态变化","出血","是否隆起"
    );

    public InferenceWorker(
            RedisTemplate<String, Object> redisTemplate,
            InferenceJobRepo jobRepo,
            InferenceResultRepo resultRepo,
            CaseRepo caseRepo,
            CaseImageRepo imageRepo,
            InferApiClient inferApiClient,
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
        this.inferApiClient = inferApiClient;
        this.streamKey = streamKey;
        this.group = group;
        this.consumer = consumer;
        this.pollMs = pollMs;

        ensureGroup();
    }

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
                    Object jobIdObj = r.getValue().get("jobId");
                    if (jobIdObj == null) {
                        // 例如 init=1 这种消息，直接 ACK
                        redisTemplate.opsForStream().acknowledge(streamKey, group, r.getId());
                        continue;
                    }

                    Long jobId = Long.valueOf(String.valueOf(jobIdObj));
                    processJob(jobId);

                    // 成功才 ACK
                    redisTemplate.opsForStream().acknowledge(streamKey, group, r.getId());
                } catch (Exception e) {
                    // 不 ACK，让它留在 pending（后续可以做 reclaim/重试）
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

        // 抢占：QUEUED -> RUNNING
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

            // 取最新图片（按 id 最大）
            CaseImage latest = images.stream()
                    .max(Comparator.comparing(CaseImage::getId))
                    .orElseThrow();

            String imagePath = latest.getFilePath();

            // 严格 meta.json：key 必须齐全，缺的填空
            String metaJsonString = buildStrictMetaJson(c);

            // 调 FastAPI（multipart -> txt）
                        String txt = inferApiClient.inferReportMultipartTxt(imagePath, metaJsonString);

            // 保存结果
                        InferenceResult r = new InferenceResult();
                        r.setJobId(job.getId());
                        r.setCaseId(caseId);

            // 兜底：resultJson 字段也存原文（虽然名字叫 json，但我们现在存 txt）
                        r.setResultJson(txt);

            // 从第一行解析 probs，并推断 predLabel（保持你原本字段可用）
                        ParsedTxtReport parsed = ParsedTxtReport.parse(txt);
                        if (parsed != null) {
                            if (parsed.predLabel != null) r.setPredLabel(parsed.predLabel);
                            if (parsed.probsJson != null) r.setProbsJson(parsed.probsJson);
                        }

            // reportJson 字段现在改作“报告纯文本”保存（避免 DTO 解析 JSON 失败）
                        r.setReportJson(txt);

            // modelVersion（可选）你可以在 FastAPI 文本里加版本行再解析，这里先不填
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

            throw e;
        }
    }

    /**
     * 你要求 meta.json 严格包含 21 个字段。
     * 当前 Case 实体只有 patientAge/patientSex 等少量信息，所以这里做“最小可跑通”映射：
     *  - 年龄 <- patientAge
     *  - 性别 <- patientSex
     *  - 其他字段暂填空字符串（但 key 必须存在）
     *
     * 后续你把病例表扩展字段后，只需要在这里把值补上即可。
     */
    private String buildStrictMetaJson(Case c) {
        Map<String, Object> m = new LinkedHashMap<>();

        // 先全部填空，保证 key 齐全
        for (String k : REQUIRED_FIELDS) {
            m.put(k, "");
        }

        // 再覆盖你现在有的数据
        if (c.getPatientAge() != null) m.put("年龄", c.getPatientAge());
        if (c.getPatientSex() != null) m.put("性别", c.getPatientSex());

        // （可选）你现有 Case 里有 chiefComplaint/history，可先拼到“形态变化/区域”等字段里做联调
        // 这里我不乱塞，保持严格字段+不擅自杜撰含义

        return JsonUtil.toJson(m);
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
        }
    }

    private static class ParsedTxtReport {
        String predLabel;   // 例如 mel
        String probsJson;   // {"akiec":0.000003,"bcc":0.0,...}

        static ParsedTxtReport parse(String txt) {
            if (txt == null || txt.isBlank()) return null;

            // 找到第一行：预测标签：{akiec:0.000003 bcc:...}
            String[] lines = txt.split("\\r?\\n");
            String first = null;
            for (String ln : lines) {
                if (ln != null && ln.trim().startsWith("预测标签")) { first = ln.trim(); break; }
            }
            if (first == null) return null;

            int l = first.indexOf('{');
            int r = first.lastIndexOf('}');
            if (l < 0 || r <= l) return null;

            String inside = first.substring(l + 1, r).trim(); // akiec:0.000003 bcc:0.000001 ...
            if (inside.isBlank()) return null;

            java.util.Map<String, Double> probs = new java.util.LinkedHashMap<>();
            String[] parts = inside.split("\\s+");
            for (String p : parts) {
                if (p.isBlank()) continue;
                int k = p.indexOf(':');
                if (k <= 0) continue;
                String key = p.substring(0, k).trim();
                String val = p.substring(k + 1).trim();
                try {
                    double dv = Double.parseDouble(val);
                    probs.put(key, dv);
                } catch (Exception ignore) {}
            }
            if (probs.isEmpty()) return null;

            // 取最大概率作为 predLabel
            String best = null;
            double bestV = -1.0;
            for (var e : probs.entrySet()) {
                if (e.getValue() != null && e.getValue() > bestV) {
                    bestV = e.getValue();
                    best = e.getKey();
                }
            }

            ParsedTxtReport out = new ParsedTxtReport();
            out.predLabel = best;
            out.probsJson = JsonUtil.toJson(probs);
            return out;
        }
    }

}
