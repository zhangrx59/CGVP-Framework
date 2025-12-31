package com.aiserver;

import org.springframework.web.bind.annotation.*;

import java.util.Map;

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
}
