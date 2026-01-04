package com.aiserver;

import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import java.util.Map;

@RestController
@RequestMapping("/cases")
public class CaseController {

    private final CaseService caseService;

    public CaseController(CaseService caseService) {
        this.caseService = caseService;
    }

    // 创建病例（需要登录）
    @PostMapping
    public CaseDtos.CaseView create(@Valid @RequestBody CaseDtos.CreateReq req) {
        return caseService.create(req);
    }

    // 查看病例（需要登录）
    @GetMapping("/{id}")
    public CaseDtos.CaseView get(@PathVariable Long id) {
        return caseService.getById(id);
    }

    // ⭐ NEW：查看所有病例（DOCTOR/ADMIN）
    @GetMapping
    public List<CaseDtos.CaseView> listAll() {
        return caseService.listAll();
    }

    // ⭐ NEW：删除病例（DOCTOR / ADMIN）
    @DeleteMapping("/{id}")
    public Map<String, Object> delete(@PathVariable Long id) {
        caseService.deleteById(id);
        return Map.of("ok", true);
    }


}
