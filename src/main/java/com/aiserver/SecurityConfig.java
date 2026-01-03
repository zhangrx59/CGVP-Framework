package com.aiserver;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

// ⭐ NEW：你用了 @PreAuthorize，就必须启用方法级鉴权
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;

import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
@EnableMethodSecurity // ⭐ NEW
public class SecurityConfig {

    private final JwtService jwtService; // ⭐ MODIFIED：注入 JwtService（你原项目就是这样用的）

    public SecurityConfig(JwtService jwtService) { // ⭐ MODIFIED：不再注入 JwtAuthFilter
        this.jwtService = jwtService;
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {

        http
                .csrf(csrf -> csrf.disable())
                .authorizeHttpRequests(auth -> auth
                        // ⭐ MODIFIED：放行登录/注册/健康检查（按你现有路由）
                        .requestMatchers("/auth/**", "/health").permitAll()
                        .anyRequest().authenticated()
                )
                // ⭐ MODIFIED：按你原项目风格，直接 new Filter
                .addFilterBefore(
                        new JwtAuthFilter(jwtService),
                        org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter.class
                );

        return http.build();
    }
}
