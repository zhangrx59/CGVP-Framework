package com.aiserver;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;
import org.springframework.stereotype.Service;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;

@Service
public class JwtService {

    // token 有效期（你原来就有）
    private static final long EXP_MS = 24 * 60 * 60 * 1000L;

    private final SecretKey key;

    public JwtService() {
        // 保持你原有的 key 构造方式
        this.key = Keys.hmacShaKeyFor(
                "very-secret-key-for-demo-please-change".getBytes(StandardCharsets.UTF_8)
        );
    }

    /**
     * 签发 JWT
     * ❗ 注意：这里【绝对不能】限制角色
     */
    public String issueToken(User u) {

        // ✅ 不要在这里判断 role
        // ADMIN / DOCTOR / NURSE 都应该能拿到 token

        return Jwts.builder()
                .setSubject(u.getUsername())
                .claim("uid", u.getId())
                .claim("role", u.getRole())
                .setIssuedAt(new Date())
                .setExpiration(new Date(System.currentTimeMillis() + EXP_MS))
                .signWith(key, SignatureAlgorithm.HS256)
                .compact();
    }

    /**
     * 解析 JWT
     */
    public JwtUser parse(String token) {
        Claims c = Jwts.parser()
                .setSigningKey(key)
                .build()
                .parseClaimsJws(token)
                .getBody();

        Long uid = ((Number) c.get("uid")).longValue();
        String role = (String) c.get("role");

        return new JwtUser(
                uid,
                c.getSubject(),
                role
        );
    }

    /**
     * JWT 中的用户信息
     */
    public record JwtUser(
            Long userId,
            String username,
            String role
    ) {}
}
