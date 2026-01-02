package com.aiserver;

import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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

    @PostMapping("/cases/{caseId}/infer")
    public Map<String, Object> infer(@PathVariable Long caseId) {
        InferenceJob job = inferService.runInfer(caseId);
        return Map.of("jobId", job.getId(), "status", job.getStatus());
    }

    @GetMapping("/jobs/{jobId}")
    public InferenceJob getJob(@PathVariable Long jobId) {
        return jobRepo.findById(jobId).orElseThrow(() -> new IllegalArgumentException("job not found"));
    }

    @GetMapping("/cases/{caseId}/result")
    public InferenceResult getLatestResult(@PathVariable Long caseId) {
        return resultRepo.findTopByCaseIdOrderByIdDesc(caseId)
                .orElseThrow(() -> new IllegalArgumentException("result not found"));
    }

    /**
     * 获取某个病例的所有推理记录（列表）
     */
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
    @GetMapping("/inferences/{jobId}")
    public InferenceDtos.InferenceResultView getInferenceByJobId(@PathVariable Long jobId) {
        InferenceResult result = resultRepo.findByJobId(jobId)
                .orElseThrow(() -> new IllegalArgumentException("inference result not found for jobId: " + jobId));
        return InferenceDtos.InferenceResultView.from(result);
    }

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
