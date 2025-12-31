package com.aiserver;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

public class AuthDtos {

    public record RegisterReq(
            @NotBlank String username,
            @NotBlank String password,
            @NotNull String role,   // DOCTOR/NURSE/ADMIN
            String dept
    ) {}

    public record LoginReq(
            @NotBlank String username,
            @NotBlank String password
    ) {}

    public record UserView(
            Long id,
            String username,
            String role,
            String dept
    ) {}

    public record LoginResp(
            UserView user,
            String token // 下一步加 JWT，这里先返回占位 null
    ) {}
}
