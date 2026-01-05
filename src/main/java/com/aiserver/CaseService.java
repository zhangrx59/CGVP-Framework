package com.aiserver;

import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CaseService {

    private final CaseRepo caseRepo;
    private final UserRepo userRepo;

    public CaseService(CaseRepo caseRepo, UserRepo userRepo) {
        this.caseRepo = caseRepo;
        this.userRepo = userRepo;
    }

    public CaseDtos.CaseView create(CaseDtos.CreateReq req) {
        JwtService.JwtUser me = currentUser();

        // 🔑 从数据库查当前用户，拿 dept
        User user = userRepo.findById(me.userId())
                .orElseThrow(() -> new IllegalStateException("user not found"));

        Case c = new Case();
        c.setPatientName(req.patientName());
        c.setPatientSex(req.patientSex());
        c.setPatientAge(req.patientAge());
        c.setChiefComplaint(req.chiefComplaint());
        c.setHistory(req.history());

        c.setStatus("NEW");
        c.setCreatedBy(user.getId());
        c.setDept(user.getDept()); // ✅ 来自数据库

        caseRepo.save(c);

        return toView(c);
    }

    // ⭐ MODIFIED：只有 DOCTOR/ADMIN 才能查看病例详情
    public CaseDtos.CaseView getById(Long id) {
        ensureDoctorOrAdmin(); // ⭐ NEW：权限校验

        Case c = caseRepo.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("case not found"));
        return toView(c);
    }

    // ⭐ NEW：删除病例（仅 DOCTOR / ADMIN）
    public void deleteById(Long id) {
        // ⭐ NEW：权限校验
        ensureDoctorOrAdmin();

        // ⭐ NEW：存在性检查（沿用你现在 getById 的 “case not found” 风格）
        if (!caseRepo.existsById(id)) {
            throw new IllegalArgumentException("case not found");
        }

        caseRepo.deleteById(id);
    }

    // ⭐ NEW：修改病例（仅 DOCTOR / ADMIN）
    public CaseDtos.CaseView updateById(Long id, CaseDtos.UpdateReq req) {
        ensureDoctorOrAdmin(); // ⭐ NEW：复用权限校验（你删功能里已经有/或按下面补一个）

        Case c = caseRepo.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("case not found"));

        // ⭐ NEW：按请求覆盖（最小化：全量覆盖，不搞复杂 patch）
        c.setPatientName(req.patientName());
        c.setPatientSex(req.patientSex());
        c.setPatientAge(req.patientAge());
        c.setChiefComplaint(req.chiefComplaint());
        c.setHistory(req.history());

        caseRepo.save(c);
        return toView(c);
    }


    // ⭐ NEW：查看所有病例（不限部门），仅 DOCTOR/ADMIN
    public List<CaseDtos.CaseView> listAll() {
        ensureDoctorOrAdmin(); // ⭐ NEW：权限校验

        return caseRepo.findAll().stream()
                .map(this::toView)
                .toList();
    }

    // ⭐ NEW：统一校验权限，避免 controller 分散判断
    private void ensureDoctorOrAdmin() {
        JwtService.JwtUser me = currentUser();
        String role = me.role() == null ? "" : me.role().trim().toUpperCase();

        if (!role.equals("DOCTOR") && !role.equals("ADMIN")) {
            // 你的 ApiExceptionHandler 会把 SecurityException 映射为 403
            throw new SecurityException("no permission");
        }
    }

    private JwtService.JwtUser currentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || auth.getPrincipal() == null) {
            throw new IllegalStateException("not authenticated");
        }
        return (JwtService.JwtUser) auth.getPrincipal();
    }


    // ⭐ MODIFIED：兼容 ROLE_ 前缀
    private String normRole(String role) {
        if (role == null) return "";
        role = role.trim().toUpperCase();
        if (role.startsWith("ROLE_")) role = role.substring("ROLE_".length());
        return role;
    }

    private CaseDtos.CaseView toView(Case c) {
        return new CaseDtos.CaseView(
                c.getId(),
                c.getPatientName(),
                c.getPatientSex(),
                c.getPatientAge(),
                c.getChiefComplaint(),
                c.getHistory(),
                c.getStatus(),
                c.getCreatedBy(),
                c.getDept()
        );
    }
}
