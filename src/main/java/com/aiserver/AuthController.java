package com.aiserver;

import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/auth")
public class AuthController {

    private final AuthService authService;
    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/register")
    public AuthDtos.UserView register(@Valid @RequestBody AuthDtos.RegisterReq req) {
        return authService.register(req);
    }

    @PostMapping("/login")
    public AuthDtos.LoginResp login(@Valid @RequestBody AuthDtos.LoginReq req) {
        return authService.login(req);
    }
}
