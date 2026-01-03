package com.aiserver;

import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/cases")
public class CaseImageController {

    private final CaseImageService imageService;

    public CaseImageController(CaseImageService imageService) {
        this.imageService = imageService;
    }

    // ⭐ MODIFIED：加角色限制（护士/医生/管理员都可以上传）
    @PreAuthorize("hasAnyRole('NURSE','DOCTOR','ADMIN')")
    @PostMapping("/{caseId}/images")
    public CaseImage upload(
            @PathVariable Long caseId,
            @RequestParam("file") MultipartFile file
    ) throws Exception {
        return imageService.upload(caseId, file);
    }
}

