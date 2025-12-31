package com.aiserver;

import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
public class MeController {

    @GetMapping("/me")
    public Map<String, Object> me(Authentication authentication) {
        var jwtUser = (JwtService.JwtUser) authentication.getPrincipal();
        return Map.of(
                "userId", jwtUser.userId(),
                "username", jwtUser.username(),
                "role", jwtUser.role()
        );
    }
}
