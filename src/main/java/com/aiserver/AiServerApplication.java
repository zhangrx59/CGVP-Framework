package com.aiserver;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties(JwtProps.class)
public class AiServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(AiServerApplication.class, args);
    }
}

