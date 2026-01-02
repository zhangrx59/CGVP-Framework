package com.aiserver;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
@EnableConfigurationProperties(JwtProps.class)
public class AiServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(AiServerApplication.class, args);
    }
}

