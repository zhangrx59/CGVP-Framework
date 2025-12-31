package com.aiserver;

import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.HashMap;
import java.util.List;

@Service
public class InferService {

    private final CaseRepo caseRepo;
    private final CaseImageRepo imageRepo;
    private final InferenceJobRepo jobRepo;
    private final InferenceResultRepo resultRepo;
    private final AiClient aiClient;

    public InferService(CaseRepo caseRepo,
                        CaseImageRepo imageRepo,
                        InferenceJobRepo jobRepo,
                        InferenceResultRepo resultRepo,
                        AiClient aiClient) {
        this.caseRepo = caseRepo;
        this.imageRepo = imageRepo;
        this.jobRepo = jobRepo;
        this.resultRepo = resultRepo;
        this.aiClient = aiClient;
    }

    public InferenceJob runInfer(Long caseId) {
        JwtService.JwtUser me = currentUser();

        Case c = caseRepo.findById(caseId)
                .orElseThrow(() -> new IllegalArgumentException("case not found"));

        // ✅ 先保持最严格：只能创建者推理（后面再放宽为同科室/管理员）
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
        job.setStatus("RUNNING");
        job.setAttemptCount(1);
        job.setStartedAt(Instant.now());
        jobRepo.save(job);

        try {
            // 2) 组装 FastAPI 请求
            var imagePaths = images.stream().map(CaseImage::getFilePath).toList();

            var caseData = new HashMap<String, Object>();
            caseData.put("patientName", c.getPatientName());
            caseData.put("patientSex", c.getPatientSex());
            caseData.put("patientAge", c.getPatientAge());
            caseData.put("chiefComplaint", c.getChiefComplaint());
            caseData.put("history", c.getHistory());

            AiDtos.InferReq req = new AiDtos.InferReq(job.getId(), caseId, imagePaths, caseData);

            // 3) 调 FastAPI
            AiDtos.InferResp resp = aiClient.infer(req);

            // 4) 落结果
            InferenceResult r = new InferenceResult();
            r.setJobId(job.getId());
            r.setCaseId(caseId);
            r.setResultJson(JsonUtil.toJson(resp)); // 用我们下面的 JsonUtil
            resultRepo.save(r);

            // 5) 更新 job
            job.setStatus("SUCCEEDED");
            job.setFinishedAt(Instant.now());
            job.setLastError(null);
            jobRepo.save(job);

            return job;

        } catch (Exception e) {
            job.setStatus("FAILED");
            job.setFinishedAt(Instant.now());
            job.setLastError(e.getClass().getSimpleName() + ": " + e.getMessage());
            jobRepo.save(job);
            throw e;
        }
    }

    private JwtService.JwtUser currentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        return (JwtService.JwtUser) auth.getPrincipal();
    }
}
