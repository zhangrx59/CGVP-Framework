package com.aiserver;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface InferenceResultRepo extends JpaRepository<InferenceResult, Long> {

    Optional<InferenceResult> findByJobId(Long jobId);

    Optional<InferenceResult> findTopByCaseIdOrderByIdDesc(Long caseId);

    List<InferenceResult> findByCaseIdOrderByIdDesc(Long caseId);

    // ✅ NEW：覆盖保存用（同一病例只保留最新一条）
    void deleteByCaseId(Long caseId); // ✅ NEW
}
