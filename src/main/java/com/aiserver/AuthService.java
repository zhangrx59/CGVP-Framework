package com.aiserver;

import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class AuthService {

    private final UserRepo userRepo;
    private final JwtService jwtService;
    private final PasswordEncoder encoder = new BCryptPasswordEncoder();

    // ✅ 注意：只有一个构造器，并且在类的大括号里面
    public AuthService(UserRepo userRepo, JwtService jwtService) {
        this.userRepo = userRepo;
        this.jwtService = jwtService;
    }

    public AuthDtos.UserView register(AuthDtos.RegisterReq req) {
        if (userRepo.existsByUsername(req.username())) {
            throw new IllegalArgumentException("username already exists");
        }
        User u = new User();
        u.setUsername(req.username());
        u.setPasswordHash(encoder.encode(req.password()));
        u.setRole(req.role());
        u.setDept(req.dept());
        userRepo.save(u);
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
