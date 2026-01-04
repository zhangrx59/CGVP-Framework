package com.aiserver;

import jakarta.validation.constraints.NotBlank;

public class CaseDtos {

    public record CreateReq(
            String patientName,
            String patientSex,     // M/F/U
            Integer patientAge,
            @NotBlank String chiefComplaint,
            String history
    ) {}

    // ⭐ NEW：用于修改病例（字段保持一致，最小化）
    public record UpdateReq(
            String patientName,
            String patientSex,     // M/F/U
            Integer patientAge,
            @NotBlank String chiefComplaint,
            String history
    ) {}

    public record CaseView(
            Long id,
            String patientName,
            String patientSex,
            Integer patientAge,
            String chiefComplaint,
            String history,
            String status,
            Long createdBy,
            String dept
    ) {}
}
