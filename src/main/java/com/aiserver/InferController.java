package com.aiserver;

import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

// ⭐ NEW：你现在开始用 @PreAuthorize 的话，需要这个 import
import org.springframework.security.access.prepost.PreAuthorize;

@RestController
public class InferController {

    private final InferService inferService;
    private final InferenceJobRepo jobRepo;
    private final InferenceResultRepo resultRepo;

    public InferController(InferService inferService,
                           InferenceJobRepo jobRepo,
                           InferenceResultRepo resultRepo) {
        this.inferService = inferService;
        this.jobRepo = jobRepo;
        this.resultRepo = resultRepo;
    }

    // ⭐ MODIFIED：护士不能推理；医生/管理员可以
    // ✅ 路径保持不变：POST /cases/{caseId}/infer :contentReference[oaicite:2]{index=2}
    @PreAuthorize("hasAnyRole('DOCTOR','ADMIN')")
    @PostMapping("/cases/{caseId}/infer")
    public Map<String, Object> infer(@PathVariable Long caseId) {
        InferenceJob job = inferService.runInfer(caseId);
        return Map.of("jobId", job.getId(), "status", job.getStatus());
    }

    // ⭐ MODIFIED：护士不能查看 job（因为能间接获取推理状态/结果）
    // ✅ 路径保持不变：GET /jobs/{jobId} :contentReference[oaicite:3]{index=3}
    @PreAuthorize("hasAnyRole('DOCTOR','ADMIN')")
    @GetMapping("/jobs/{jobId}")
    public InferenceJob getJob(@PathVariable Long jobId) {
        return jobRepo.findById(jobId).orElseThrow(() -> new IllegalArgumentException("job not found"));
    }

    // ⭐ MODIFIED：护士不能查看病例最新结果
    // ✅ 路径保持不变：GET /cases/{caseId}/result :contentReference[oaicite:4]{index=4}
    @PreAuthorize("hasAnyRole('DOCTOR','ADMIN')")
    @GetMapping("/cases/{caseId}/result")
    public InferenceResult getLatestResult(@PathVariable Long caseId) {
        return resultRepo.findTopByCaseIdOrderByIdDesc(caseId)
                .orElseThrow(() -> new IllegalArgumentException("result not found"));
    }

    /**
     * 获取某个病例的所有推理记录（列表）
     */
    // ⭐ MODIFIED：护士不能查看病例推理记录列表
    // ✅ 路径保持不变：GET /cases/{caseId}/inferences :contentReference[oaicite:5]{index=5}
    @PreAuthorize("hasAnyRole('DOCTOR','ADMIN')")
    @GetMapping("/cases/{caseId}/inferences")
    public List<InferenceDtos.InferenceResultView> getCaseInferences(@PathVariable Long caseId) {
        List<InferenceResult> results = resultRepo.findByCaseIdOrderByIdDesc(caseId);
        return results.stream()
                .map(InferenceDtos.InferenceResultView::from)
                .collect(Collectors.toList());
    }

    /**
     * 获取某次推理的完整结果（含 report）
     */
    // ⭐ MODIFIED：护士不能查看某次推理结果
    // ✅ 路径保持不变：GET /inferences/{jobId} :contentReference[oaicite:6]{index=6}
    @PreAuthorize("hasAnyRole('DOCTOR','ADMIN')")
    @GetMapping("/inferences/{jobId}")
    public InferenceDtos.InferenceResultView getInferenceByJobId(@PathVariable Long jobId) {
        InferenceResult result = resultRepo.findByJobId(jobId)
                .orElseThrow(() -> new IllegalArgumentException("inference result not found for jobId: " + jobId));
        return InferenceDtos.InferenceResultView.from(result);
    }

    // ⭐ MODIFIED：护士不能查看某次推理结果
    // ✅ 路径保持不变：GET /inferences/{jobId} :contentReference[oaicite:6]{index=6}
    @PreAuthorize("hasAnyRole('DOCTOR','ADMIN')")
    @GetMapping(value = "/inferences/{jobId}/report.txt", produces = "text/plain; charset=UTF-8")
    public String getInferenceTxt(@PathVariable Long jobId) {
        InferenceResult result = resultRepo.findByJobId(jobId)
                .orElseThrow(() -> new IllegalArgumentException("inference result not found for jobId: " + jobId));
        String txt = result.getReportJson();
        if (txt == null || txt.isBlank()) {
            // 兜底：resultJson 里也存了原文
            txt = result.getResultJson();
        }
        return (txt == null) ? "" : txt;
    }

}
