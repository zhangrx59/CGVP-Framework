package com.aiserver;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "cases")
public class Case {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // 患者信息
    @Column(length = 64)
    private String patientName;

    @Column(length = 1)
    private String patientSex; // M / F / U

    private Integer patientAge;

    // 病例描述
    @Column(nullable = false, length = 1000)
    private String chiefComplaint;

    @Column(length = 2000)
    private String history;

    // 病例状态
    @Column(nullable = false, length = 16)
    private String status; // NEW / READY / DONE

    // 关联信息
    @Column(nullable = false)
    private Long createdBy;

    @Column(length = 64)
    private String dept;

    @Column(nullable = false)
    private Instant createdAt = Instant.now();

    /* ===== getter / setter ===== */

    public Long getId() { return id; }

    public String getPatientName() { return patientName; }
    public void setPatientName(String patientName) { this.patientName = patientName; }

    public String getPatientSex() { return patientSex; }
    public void setPatientSex(String patientSex) { this.patientSex = patientSex; }

    public Integer getPatientAge() { return patientAge; }
    public void setPatientAge(Integer patientAge) { this.patientAge = patientAge; }

    public String getChiefComplaint() { return chiefComplaint; }
    public void setChiefComplaint(String chiefComplaint) { this.chiefComplaint = chiefComplaint; }

    public String getHistory() { return history; }
    public void setHistory(String history) { this.history = history; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }

    public String getDept() { return dept; }
    public void setDept(String dept) { this.dept = dept; }

    public Instant getCreatedAt() { return createdAt; }
}
