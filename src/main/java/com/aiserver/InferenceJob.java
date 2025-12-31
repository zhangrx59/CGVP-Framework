package com.aiserver;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "inference_jobs", indexes = {
        @Index(name = "idx_jobs_case_id", columnList = "caseId")
})
public class InferenceJob {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private Long caseId;

    @Column(nullable = false)
    private Long createdBy;

    @Column(nullable = false, length = 16)
    private String status; // QUEUED/RUNNING/SUCCEEDED/FAILED

    @Column(nullable = false)
    private Integer attemptCount = 0;

    @Column(length = 2000)
    private String lastError;

    private Instant createdAt = Instant.now();
    private Instant startedAt;
    private Instant finishedAt;

    // getters/setters
    public Long getId() { return id; }

    public Long getCaseId() { return caseId; }
    public void setCaseId(Long caseId) { this.caseId = caseId; }

    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public Integer getAttemptCount() { return attemptCount; }
    public void setAttemptCount(Integer attemptCount) { this.attemptCount = attemptCount; }

    public String getLastError() { return lastError; }
    public void setLastError(String lastError) { this.lastError = lastError; }

    public Instant getCreatedAt() { return createdAt; }

    public Instant getStartedAt() { return startedAt; }
    public void setStartedAt(Instant startedAt) { this.startedAt = startedAt; }

    public Instant getFinishedAt() { return finishedAt; }
    public void setFinishedAt(Instant finishedAt) { this.finishedAt = finishedAt; }
}
