package com.aiserver;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.connection.stream.MapRecord;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.List;
import java.util.Map;

@Service
public class InferService {

    private final CaseRepo caseRepo;
    private final CaseImageRepo imageRepo;
    private final InferenceJobRepo jobRepo;
    private final RedisTemplate<String, Object> redisTemplate;

    private final String streamKey;

    public InferService(
            CaseRepo caseRepo,
            CaseImageRepo imageRepo,
            InferenceJobRepo jobRepo,
            RedisTemplate<String, Object> redisTemplate,
            @Value("${app.queue.stream}") String streamKey
    ) {
        this.caseRepo = caseRepo;
        this.imageRepo = imageRepo;
        this.jobRepo = jobRepo;
        this.redisTemplate = redisTemplate;
        this.streamKey = streamKey;
    }

    /**
     * 现在 runInfer 只负责：
     * - 权限校验
     * - 创建 job（QUEUED）
     * - XADD 入队（jobId）
     * 返回 job，供前端轮询 /jobs/{jobId}
     */
    public InferenceJob runInfer(Long caseId) {
        JwtService.JwtUser me = currentUser();

        Case c = caseRepo.findById(caseId)
                .orElseThrow(() -> new IllegalArgumentException("case not found"));

        // 仍然保持最严格：只能创建者推理（后续可扩展同科室/管理员）
        if (!c.getCreatedBy().equals(me.userId())) {
            throw new SecurityException("no permission");
        }

        List<CaseImage> images = imageRepo.findByCaseId(caseId);
        if (images.isEmpty()) {
            throw new IllegalArgumentException("no image uploaded");
        }

        // 1) 创建 job
        InferenceJob job = new InferenceJob();
        job.setCaseId(caseId);
        job.setCreatedBy(me.userId());
        job.setStatus("QUEUED");
        job.setAttemptCount(0);
        job.setLastError(null);
        job.setStartedAt(null);
        job.setFinishedAt(null);
        jobRepo.save(job);

        // 2) 入队：写入 jobId
        redisTemplate.opsForStream().add(
                MapRecord.create(streamKey, Map.of("jobId", String.valueOf(job.getId())))
        );

        return job;
    }

    private JwtService.JwtUser currentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        return (JwtService.JwtUser) auth.getPrincipal();
    }
}
