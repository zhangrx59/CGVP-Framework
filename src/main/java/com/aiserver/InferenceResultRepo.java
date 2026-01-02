package com.aiserver;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface InferenceResultRepo extends JpaRepository<InferenceResult, Long> {

    Optional<InferenceResult> findByJobId(Long jobId);

    Optional<InferenceResult> findTopByCaseIdOrderByIdDesc(Long caseId);

    List<InferenceResult> findByCaseIdOrderByIdDesc(Long caseId);
}
