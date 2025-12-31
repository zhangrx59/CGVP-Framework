package com.aiserver;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface InferenceResultRepo extends JpaRepository<InferenceResult, Long> {
    Optional<InferenceResult> findTopByCaseIdOrderByIdDesc(Long caseId);
    Optional<InferenceResult> findByJobId(Long jobId);
}
