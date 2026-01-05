package com.aiserver;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.core.io.Resource;

import java.util.List;

@RestController
@RequestMapping("/cases")
public class CaseImageController {

    private final CaseImageService imageService;

    public CaseImageController(CaseImageService imageService) {
        this.imageService = imageService;
    }

    @PreAuthorize("hasAnyRole('NURSE','DOCTOR','ADMIN')")
    @PostMapping("/{caseId}/images")
    public CaseImage upload(@PathVariable Long caseId, @RequestParam("file") MultipartFile file) throws Exception {
        return imageService.upload(caseId, file);
    }

    // ✅ NEW：图片列表（给前端展示用）
    @PreAuthorize("hasAnyRole('DOCTOR','ADMIN')")
    @GetMapping("/{caseId}/images")
    public List<CaseImage> list(@PathVariable Long caseId) {
        return imageService.list(caseId);
    }

    // ✅ NEW：图片二进制（<img src="...">）
    @PreAuthorize("hasAnyRole('DOCTOR','ADMIN')")
    @GetMapping(value = "/{caseId}/images/{imageId}/raw")
    public ResponseEntity<Resource> raw(@PathVariable Long caseId, @PathVariable Long imageId) throws Exception {
        return imageService.raw(caseId, imageId);
    }
}


