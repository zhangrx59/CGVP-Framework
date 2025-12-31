package com.aiserver;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "inference_results",
        uniqueConstraints = @UniqueConstraint(name = "uk_results_job_id", columnNames = "jobId"),
        indexes = @Index(name = "idx_results_case_id", columnList = "caseId"))
public class InferenceResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private Long jobId;

    @Column(nullable = false)
    private Long caseId;

    @Lob
    @Column(nullable = false)
    private String resultJson;

    private Instant createdAt = Instant.now();

    public Long getId() { return id; }
    public Long getJobId() { return jobId; }
    public void setJobId(Long jobId) { this.jobId = jobId; }

    public Long getCaseId() { return caseId; }
    public void setCaseId(Long caseId) { this.caseId = caseId; }

    public String getResultJson() { return resultJson; }
    public void setResultJson(String resultJson) { this.resultJson = resultJson; }

    public Instant getCreatedAt() { return createdAt; }
}
