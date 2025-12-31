package com.aiserver;

import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

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

    public CaseDtos.CaseView getById(Long id) {
        Case c = caseRepo.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("case not found"));
        return toView(c);
    }

    private JwtService.JwtUser currentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || auth.getPrincipal() == null) {
            throw new IllegalStateException("not authenticated");
        }
        return (JwtService.JwtUser) auth.getPrincipal();
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
