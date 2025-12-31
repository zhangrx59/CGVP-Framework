package com.aiserver;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.springframework.stereotype.Service;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Date;
import java.util.Map;

@Service
public class JwtService {

    private final JwtProps props;

    public JwtService(JwtProps props) {
        this.props = props;
    }

    public String issueToken(User u) {
        var key = Keys.hmacShaKeyFor(props.secret().getBytes(StandardCharsets.UTF_8));
        Instant now = Instant.now();
        Instant exp = now.plusSeconds(props.accessTokenMinutes() * 60L);

        return Jwts.builder()
                .issuer(props.issuer())
                .subject(String.valueOf(u.getId()))
                .issuedAt(Date.from(now))
                .expiration(Date.from(exp))
                .claims(Map.of(
                        "username", u.getUsername(),
                        "role", u.getRole(),
                        "dept", u.getDept()
                ))
                .signWith(key)
                .compact();
    }

    public JwtUser parse(String token) {
        var key = Keys.hmacShaKeyFor(props.secret().getBytes(StandardCharsets.UTF_8));

        var claims = Jwts.parser()
                .verifyWith((javax.crypto.SecretKey) key)
                .requireIssuer(props.issuer())
                .build()
                .parseSignedClaims(token)
                .getPayload();

        return new JwtUser(
                Long.valueOf(claims.getSubject()),
                claims.get("username", String.class),
                claims.get("role", String.class),
                claims.get("dept", String.class)
        );
    }
    public record JwtUser(Long userId, String username, String role, String dept) {}
}
