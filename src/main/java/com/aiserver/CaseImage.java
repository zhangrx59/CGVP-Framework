package com.aiserver;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "case_images", indexes = {
        @Index(name = "idx_case_images_case_id", columnList = "caseId")
})
public class CaseImage {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private Long caseId;

    @Column(nullable = false, length = 255)
    private String fileName;     // 原始文件名

    @Column(nullable = false, length = 512)
    private String filePath;     // 本地路径

    @Column(nullable = false, length = 64)
    private String contentType;  // image/jpeg, image/png

    @Column(nullable = false)
    private Long fileSize;

    @Column(nullable = false)
    private Instant createdAt = Instant.now();

    /* ===== getter ===== */
    public Long getId() { return id; }
    public Long getCaseId() { return caseId; }
    public String getFileName() { return fileName; }
    public String getFilePath() { return filePath; }
    public String getContentType() { return contentType; }
    public Long getFileSize() { return fileSize; }
    public Instant getCreatedAt() { return createdAt; }

    /* ===== setter ===== */
    public void setCaseId(Long caseId) { this.caseId = caseId; }
    public void setFileName(String fileName) { this.fileName = fileName; }
    public void setFilePath(String filePath) { this.filePath = filePath; }
    public void setContentType(String contentType) { this.contentType = contentType; }
    public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
}
