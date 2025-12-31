package com.aiserver;

import java.util.List;
import java.util.Map;

public class AiDtos {

    public record InferReq(
            Long jobId,
            Long caseId,
            List<String> imagePaths,
            Map<String, Object> caseData
    ) {}

    public record InferResp(
            Long jobId,
            String model,
            Map<String, Object> finalResult,
            List<Map<String, Object>> predictions,
            String note
    ) {}
}
