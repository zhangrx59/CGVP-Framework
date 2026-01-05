package com.aiserver;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface CaseImageRepo extends JpaRepository<CaseImage, Long> {
    List<CaseImage> findByCaseId(Long caseId);
}

