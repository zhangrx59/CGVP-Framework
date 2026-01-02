package com.aiserver;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(
        name = "inference_results",
        uniqueConstraints = @UniqueConstraint(name = "uk_results_job_id", columnNames = "job_id"),
        indexes = {
                @Index(name = "idx_results_case_id", columnList = "case_id")
        }
)
public class InferenceResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** 对应 inference_jobs.id */
    @Column(name = "job_id", nullable = false, updatable = false)
    private Long jobId;

    /** 对应 cases.id */
    @Column(name = "case_id", nullable = false, updatable = false)
    private Long caseId;

    /** FastAPI 原始返回（完整 JSON），用于兜底展示/追溯 */
    @Lob
    @Column(name = "result_json", columnDefinition = "LONGTEXT", nullable = false)
    private String resultJson;

    /** 便于检索/列表展示 */
    @Column(name = "pred_label")
    private String predLabel;

    /** probs 原样 JSON（便于前端画图） */
    @Lob
    @Column(name = "probs_json", columnDefinition = "LONGTEXT")
    private String probsJson;

    /** report 原样 JSON（中文结构化报告） */
    @Lob
    @Column(name = "report_json", columnDefinition = "LONGTEXT")
    private String reportJson;

    /** 例如: medgemma-4b-it+lora@202512 */
    @Column(name = "model_version")
    private String modelVersion;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt = Instant.now();

    // ===== getters / setters =====

    public Long getId() { return id; }

    public Long getJobId() { return jobId; }
    public void setJobId(Long jobId) { this.jobId = jobId; }

    public Long getCaseId() { return caseId; }
    public void setCaseId(Long caseId) { this.caseId = caseId; }

    public String getResultJson() { return resultJson; }
    public void setResultJson(String resultJson) { this.resultJson = resultJson; }

    public String getPredLabel() { return predLabel; }
    public void setPredLabel(String predLabel) { this.predLabel = predLabel; }

    public String getProbsJson() { return probsJson; }
    public void setProbsJson(String probsJson) { this.probsJson = probsJson; }

    public String getReportJson() { return reportJson; }
    public void setReportJson(String reportJson) { this.reportJson = reportJson; }

    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }

    public Instant getCreatedAt() { return createdAt; }
}
