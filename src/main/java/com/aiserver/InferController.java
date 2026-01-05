package com.aiserver;

import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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

    // ⭐ MODIFIED：仅医生可以推理（DOCTOR）
    @PreAuthorize("hasRole('DOCTOR')") // ⭐ MODIFIED
    @PostMapping("/cases/{caseId}/infer")
    public Map<String, Object> infer(@PathVariable Long caseId) {
        InferenceJob job = inferService.runInfer(caseId);
        return Map.of("jobId", job.getId(), "status", job.getStatus());
    }

    // ⭐ MODIFIED：仅医生可以查看 job
    @PreAuthorize("hasRole('DOCTOR')") // ⭐ MODIFIED
    @GetMapping("/jobs/{jobId}")
    public InferenceJob getJob(@PathVariable Long jobId) {
        return jobRepo.findById(jobId).orElseThrow(() -> new IllegalArgumentException("job not found"));
    }

    // ⭐ MODIFIED：仅医生可以查看病例最新推理结果
    @PreAuthorize("hasRole('DOCTOR')") // ⭐ MODIFIED
    @GetMapping("/cases/{caseId}/result")
    public InferenceResult getLatestResult(@PathVariable Long caseId) {
        return resultRepo.findTopByCaseIdOrderByIdDesc(caseId)
                .orElseThrow(() -> new IllegalArgumentException("result not found"));
    }

    /**
     * 获取某个病例的所有推理记录（列表）
     */
    @PreAuthorize("hasRole('DOCTOR')") // ⭐ MODIFIED
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
    @PreAuthorize("hasRole('DOCTOR')") // ⭐ MODIFIED
    @GetMapping("/inferences/{jobId}")
    public InferenceDtos.InferenceResultView getInferenceByJobId(@PathVariable Long jobId) {
        InferenceResult result = resultRepo.findByJobId(jobId)
                .orElseThrow(() -> new IllegalArgumentException("inference result not found for jobId: " + jobId));
        return InferenceDtos.InferenceResultView.from(result);
    }

    // txt 报告也仅医生可看
    @PreAuthorize("hasRole('DOCTOR')") // ⭐ MODIFIED
    @GetMapping(value = "/inferences/{jobId}/report.txt", produces = "text/plain; charset=UTF-8")
    public String getInferenceTxt(@PathVariable Long jobId) {
        InferenceResult result = resultRepo.findByJobId(jobId)
                .orElseThrow(() -> new IllegalArgumentException("inference result not found for jobId: " + jobId));
        String txt = result.getReportJson();
        if (txt == null || txt.isBlank()) {
            txt = result.getResultJson();
        }
        return (txt == null) ? "" : txt;
    }
}
