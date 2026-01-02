package com.aiserver;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;

@Component
public class InferApiClient {

    private final RestTemplate restTemplate;

    @Value("${app.fastapi.base-url:http://127.0.0.1:8000}")
    private String baseUrl;

    public InferApiClient() {
        // 最小生产化：超时必须设置
        SimpleClientHttpRequestFactory rf = new SimpleClientHttpRequestFactory();
        rf.setConnectTimeout((int) Duration.ofSeconds(3).toMillis());
        rf.setReadTimeout((int) Duration.ofSeconds(120).toMillis());
        this.restTemplate = new RestTemplate(rf);
    }

    /**
     * FastAPI 接口：POST /infer_multipart
     * multipart/form-data:
     *  - meta_json: 一个 json 文件
     *  - image: 一张图片文件
     *
     * 返回 Map（建议 FastAPI 返回：predLabel + probs + 其他说明字段）
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> inferMultipart(String imagePath, String metaJsonString) {
        if (imagePath == null || imagePath.isBlank()) {
            throw new IllegalArgumentException("imagePath is blank");
        }
        if (metaJsonString == null || metaJsonString.isBlank()) {
            throw new IllegalArgumentException("metaJsonString is blank");
        }

        FileSystemResource imageFile = new FileSystemResource(imagePath);
        if (!imageFile.exists()) {
            throw new IllegalArgumentException("image file not found: " + imagePath);
        }

        // meta_json 作为“文件”上传
        ByteArrayResource metaFile = new ByteArrayResource(metaJsonString.getBytes(StandardCharsets.UTF_8)) {
            @Override
            public String getFilename() {
                return "meta.json";
            }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

        // meta_json part：Content-Type = application/json
        HttpHeaders metaHeaders = new HttpHeaders();
        metaHeaders.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<ByteArrayResource> metaPart = new HttpEntity<>(metaFile, metaHeaders);
        body.add("meta_json", metaPart);

        // image part
        body.add("image", imageFile);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> req = new HttpEntity<>(body, headers);

        ResponseEntity<Map> resp = restTemplate.postForEntity(
                baseUrl + "/infer_report_multipart",
                req,
                Map.class
        );

        if (!resp.getStatusCode().is2xxSuccessful() || resp.getBody() == null) {
            throw new IllegalStateException("FastAPI infer failed, status=" + resp.getStatusCode());
        }

        return (Map<String, Object>) resp.getBody();
    }
}
