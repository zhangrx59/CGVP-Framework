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

    // 创建病例（需要登录：护士/医生/管理员）
    @PostMapping
    public CaseDtos.CaseView create(@Valid @RequestBody CaseDtos.CreateReq req) {
        return caseService.create(req);
    }

    // 查看病例（需要登录：护士/医生/管理员）
    @GetMapping("/{id}")
    public CaseDtos.CaseView get(@PathVariable Long id) {
        return caseService.getById(id);
    }

    // 查看所有病例（需要登录：护士/医生/管理员）
    @GetMapping
    public List<CaseDtos.CaseView> listAll() {
        return caseService.listAll();
    }

    // 删除病例（仅管理员 ADMIN）
    @DeleteMapping("/{id}")
    public Map<String, Object> delete(@PathVariable Long id) {
        caseService.deleteById(id);
        return Map.of("ok", true);
    }

    // 修改病例（医生 DOCTOR / 管理员 ADMIN）
    @PutMapping("/{id}")
    public CaseDtos.CaseView update(@PathVariable Long id, @Valid @RequestBody CaseDtos.UpdateReq req) {
        return caseService.updateById(id, req);
    }
}
