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

    // ⭐ MODIFIED：查看病例详情允许 NURSE/DOCTOR/ADMIN
    public CaseDtos.CaseView getById(Long id) {
        ensureCanReadCase(); // ⭐ MODIFIED（原 ensureDoctorOrAdmin）

        Case c = caseRepo.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("case not found"));
        return toView(c);
    }

    // 删除病例（仅 DOCTOR / ADMIN）
    public void deleteById(Long id) {
        ensureDoctorOrAdmin(); // ✅ 保持：写权限

        if (!caseRepo.existsById(id)) {
            throw new IllegalArgumentException("case not found");
        }

        caseRepo.deleteById(id);
    }

    // 修改病例（仅 DOCTOR / ADMIN）
    public CaseDtos.CaseView updateById(Long id, CaseDtos.UpdateReq req) {
        ensureDoctorOrAdmin(); // ✅ 保持：写权限

        Case c = caseRepo.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("case not found"));

        c.setPatientName(req.patientName());
        c.setPatientSex(req.patientSex());
        c.setPatientAge(req.patientAge());
        c.setChiefComplaint(req.chiefComplaint());
        c.setHistory(req.history());

        caseRepo.save(c);
        return toView(c);
    }

    // ⭐ MODIFIED：查看所有病例允许 NURSE/DOCTOR/ADMIN
    public List<CaseDtos.CaseView> listAll() {
        ensureCanReadCase(); // ⭐ MODIFIED（原 ensureDoctorOrAdmin）

        return caseRepo.findAll().stream()
                .map(this::toView)
                .toList();
    }


    // ⭐ NEW：读权限（护士也可以）
    private void ensureCanReadCase() {
        JwtService.JwtUser me = currentUser();
        String role = me.role() == null ? "" : me.role().trim().toUpperCase();
        if (role.startsWith("ROLE_")) role = role.substring("ROLE_".length());

        if (!role.equals("NURSE") && !role.equals("DOCTOR") && !role.equals("ADMIN")) {
            throw new SecurityException("no permission");
        }
    }


    // ✅ CHANGED：写权限仍只允许 DOCTOR/ADMIN，同时用 normRole 兼容 ROLE_
    private void ensureDoctorOrAdmin() {
        JwtService.JwtUser me = currentUser();
        String role = normRole(me.role()); // ✅ CHANGED

        if (!role.equals("DOCTOR") && !role.equals("ADMIN")) {
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

    // 兼容 ROLE_ 前缀
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
