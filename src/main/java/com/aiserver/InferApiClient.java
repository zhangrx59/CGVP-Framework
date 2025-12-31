//package com.aiserver;
//import org.springframework.beans.factory.annotation.Value;
//import org.springframework.core.io.ByteArrayResource;
//import org.springframework.core.io.FileSystemResource;
//import org.springframework.http.HttpEntity;
//import org.springframework.http.HttpHeaders;
//import org.springframework.http.MediaType;
//import org.springframework.http.ResponseEntity;
//import org.springframework.http.client.SimpleClientHttpRequestFactory;
//import org.springframework.stereotype.Component;
//import org.springframework.stereotype.Service;
//import org.springframework.util.LinkedMultiValueMap;
//import org.springframework.util.MultiValueMap;
//import org.springframework.web.client.RestClient;
//import org.springframework.web.client.RestTemplate;
//
//import java.nio.charset.StandardCharsets;
//import java.util.Map;

@Service
public class InferApiClient {

    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ai.infer.baseUrl:http://127.0.0.1:8000}")
    private String baseUrl;

    // imagePath: 你本地图片绝对路径
    // metaJsonString: 由 Case 字段拼成 JSON 字符串（必须含固定字段）
    public Map<String, Object> inferMultipart(String imagePath, String metaJsonString) {

        // 1) 文件 part：图片
        FileSystemResource imageFile = new FileSystemResource(imagePath);

        // 2) 文件 part：meta_json（把字符串当“文件”发过去）
        ByteArrayResource metaFile = new ByteArrayResource(metaJsonString.getBytes(StandardCharsets.UTF_8)) {
            @Override
            public String getFilename() {
                return "meta.json";
            }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("meta_json", metaFile);
        body.add("image", imageFile);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> req = new HttpEntity<>(body, headers);

        ResponseEntity<Map> resp = restTemplate.postForEntity(
                baseUrl + "/infer_multipart",
                req,
                Map.class
        );
        return resp.getBody();
    }
}
