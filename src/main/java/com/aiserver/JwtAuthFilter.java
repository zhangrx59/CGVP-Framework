package com.aiserver;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.List;

public class JwtAuthFilter extends OncePerRequestFilter {

    private final JwtService jwtService;

    public JwtAuthFilter(JwtService jwtService) {
        this.jwtService = jwtService;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {

        String auth = request.getHeader("Authorization");
        if (auth == null || !auth.startsWith("Bearer ")) {
            chain.doFilter(request, response);
            return;
        }

        String token = auth.substring("Bearer ".length()).trim();
        try {
            var user = jwtService.parse(token);

            // ⭐ MODIFIED：统一规范化 role，兼容 "ROLE_ADMIN" / "admin" / " Admin "
            String role = user.role();
            if (role == null) role = "";
            role = role.trim().toUpperCase();

            // ⭐ NEW：如果 role 里自带 "ROLE_" 前缀，去掉它
            if (role.startsWith("ROLE_")) {
                role = role.substring("ROLE_".length());
            }

            // ⭐ MODIFIED：只拼一次 ROLE_
            var authorities = role.isEmpty()
                    ? List.<SimpleGrantedAuthority>of()
                    : List.of(new SimpleGrantedAuthority("ROLE_" + role));

            // ⭐ NEW：principal 也用“去前缀后的 role”，让 Service 层判断永远一致
            var principal = new JwtService.JwtUser(user.userId(), user.username(), role);

            var authentication = new UsernamePasswordAuthenticationToken(principal, null, authorities);
            SecurityContextHolder.getContext().setAuthentication(authentication);

        } catch (Exception ignored) {
            SecurityContextHolder.clearContext();
        }

        chain.doFilter(request, response);
    }
}
