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

@Component
public class InferApiClient {

    private final RestTemplate restTemplate;

    @Value("${app.fastapi.base-url:http://127.0.0.1:8000}")
    private String baseUrl;

    public InferApiClient() {
        SimpleClientHttpRequestFactory rf = new SimpleClientHttpRequestFactory();
        rf.setConnectTimeout((int) Duration.ofSeconds(3).toMillis());
        rf.setReadTimeout((int) Duration.ofSeconds(180).toMillis()); // 推理可能较慢
        this.restTemplate = new RestTemplate(rf);
    }

    /**
     * 调 FastAPI：POST /infer_report_multipart_txt
     * multipart/form-data:
     *  - meta_json: json 文件
     *  - image: 图片文件
     *
     * 返回：text/plain (UTF-8) 的报告内容（多行）
     */
    public String inferReportMultipartTxt(String imagePath, String metaJsonString) {
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
            public String getFilename() { return "meta.json"; }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

        // meta_json part：Content-Type=application/json
        HttpHeaders metaHeaders = new HttpHeaders();
        metaHeaders.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<ByteArrayResource> metaPart = new HttpEntity<>(metaFile, metaHeaders);
        body.add("meta_json", metaPart);

        // image part
        body.add("image", imageFile);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        headers.setAccept(java.util.List.of(MediaType.TEXT_PLAIN, MediaType.ALL));

        HttpEntity<MultiValueMap<String, Object>> req = new HttpEntity<>(body, headers);

        ResponseEntity<String> resp = restTemplate.exchange(
                baseUrl + "/infer_report_multipart_txt",
                HttpMethod.POST,
                req,
                String.class
        );

        if (!resp.getStatusCode().is2xxSuccessful() || resp.getBody() == null) {
            throw new IllegalStateException("FastAPI txt infer failed, status=" + resp.getStatusCode());
        }

        return resp.getBody();
    }
}
