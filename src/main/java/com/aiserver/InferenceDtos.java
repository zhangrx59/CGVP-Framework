package com.aiserver;

import java.time.Instant;

public class InferenceDtos {

    /**
     * 推理结果视图（用于返回给前端）
     * - probsJson: 字符串（JSON 字符串也可以，前端想解析再解析）
     * - reportText: 纯文本（我们现在将 reportJson 字段作为纯文本存储）
     */
    public record InferenceResultView(
            Long id,
            Long jobId,
            Long caseId,
            String predLabel,
            String probsJson,
            String reportText,
            String modelVersion,
            Instant createdAt,
            String rawResult
    ) {
        public static InferenceResultView from(InferenceResult r) {
            return new InferenceResultView(
                    r.getId(),
                    r.getJobId(),
                    r.getCaseId(),
                    r.getPredLabel(),
                    r.getProbsJson(),
                    r.getReportJson(),  // 现在这里是 txt
                    r.getModelVersion(),
                    r.getCreatedAt(),
                    r.getResultJson()   // 兜底原文
            );
        }
    }
}
