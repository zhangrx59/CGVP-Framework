package com.aiserver;

import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class AuthService {

    private static final String adminKey = "123456"; // ⭐ NEW

    private final UserRepo userRepo;
    private final JwtService jwtService;
    private final PasswordEncoder encoder = new BCryptPasswordEncoder();

    public AuthService(UserRepo userRepo, JwtService jwtService) {
        this.userRepo = userRepo;
        this.jwtService = jwtService;
    }

    public AuthDtos.UserView register(AuthDtos.RegisterReq req) {

        if (req.username() == null || req.username().isBlank()) {
            throw new IllegalArgumentException("username required");
        }
        if (req.password() == null || req.password().isBlank()) {
            throw new IllegalArgumentException("password required");
        }
        if (req.role() == null || req.role().isBlank()) {
            throw new IllegalArgumentException("role required");
        }

        String role = req.role().trim().toUpperCase();

        // ⭐ NEW：只允许三种角色
        if (!role.equals("DOCTOR") && !role.equals("NURSE") && !role.equals("ADMIN")) {
            throw new IllegalArgumentException("invalid role");
        }

        // ⭐ NEW：管理员注册必须校验密钥
        if (role.equals("ADMIN")) {
            if (req.adminKey() == null || ! adminKey.equals(req.adminKey().trim())) {
                throw new IllegalArgumentException("invalid admin key");
            }
        }

        if (userRepo.existsByUsername(req.username())) {
            throw new IllegalArgumentException("username already exists");
        }

        User u = new User();
        u.setUsername(req.username());
        u.setPasswordHash(encoder.encode(req.password()));
        u.setRole(role);               // ⭐ MODIFIED：现在允许 DOCTOR/NURSE/ADMIN
        u.setDept(req.dept());

        u = userRepo.save(u);

        return new AuthDtos.UserView(u.getId(), u.getUsername(), u.getRole(), u.getDept());
    }

    public AuthDtos.LoginResp login(AuthDtos.LoginReq req) {
        User u = userRepo.findByUsername(req.username())
                .orElseThrow(() -> new IllegalArgumentException("invalid credentials"));

        if (!encoder.matches(req.password(), u.getPasswordHash())) {
            throw new IllegalArgumentException("invalid credentials");
        }

        String token = jwtService.issueToken(u);

        return new AuthDtos.LoginResp(
                new AuthDtos.UserView(u.getId(), u.getUsername(), u.getRole(), u.getDept()),
                token
        );
    }
}
