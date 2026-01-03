package com.aiserver;

import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/cases")
public class CaseController {

    private final CaseService caseService;

    public CaseController(CaseService caseService) {
        this.caseService = caseService;
    }

    // ⭐ MODIFIED：护士可以创建病例（上传病例）
    @PreAuthorize("hasAnyRole('NURSE','DOCTOR','ADMIN')")
    @PostMapping
    public CaseDtos.CaseView create(@RequestBody CaseDtos.CreateReq req) {
        return caseService.create(req);
    }

    // ⭐ MODIFIED：护士不能查看病例
    @PreAuthorize("hasAnyRole('DOCTOR','ADMIN')")
    @GetMapping("/{id}")
    public CaseDtos.CaseView get(@PathVariable Long id) {
        return caseService.getById(id);
    }
}
