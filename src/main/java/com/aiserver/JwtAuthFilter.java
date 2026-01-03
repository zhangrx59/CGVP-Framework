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
            // ⭐ MODIFIED：role 做一下规范化，避免出现 "doctor" / " Doctor " 导致权限失效
            String role = user.role();
            if (role == null) role = "";
            role = role.trim().toUpperCase();

            // ⭐ MODIFIED：role 为空时不给 authority（避免 ROLE_）
            var authorities = role.isEmpty()
                    ? List.<SimpleGrantedAuthority>of()
                    : List.of(new SimpleGrantedAuthority("ROLE_" + role));

            var principal = user; // 直接把 JwtUser 放进去
            var authentication = new UsernamePasswordAuthenticationToken(principal, null, authorities);
            SecurityContextHolder.getContext().setAuthentication(authentication);

        } catch (Exception ignored) {
            SecurityContextHolder.clearContext();
            // 不直接返回，让后面的鉴权规则决定（/me 会 401）
        }

        chain.doFilter(request, response);
    }
}
