package com.aiserver;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.time.Duration;

@Component
public class AiClient {

    private final RestClient restClient;

    public AiClient(
            @Value("${app.ai.base-url}") String baseUrl,
            @Value("${app.ai.connect-timeout-ms}") int connectTimeoutMs,
            @Value("${app.ai.read-timeout-ms}") int readTimeoutMs
    ) {
        var rf = new SimpleClientHttpRequestFactory();
        rf.setConnectTimeout(connectTimeoutMs);
        rf.setReadTimeout(readTimeoutMs);

        this.restClient = RestClient.builder()
                .baseUrl(baseUrl)
                .requestFactory(rf)
                .build();
    }

    public AiDtos.InferResp infer(AiDtos.InferReq req) {
        return restClient.post()
                .uri("/infer")
                .body(req)
                .retrieve()
                .body(AiDtos.InferResp.class);
    }
}
