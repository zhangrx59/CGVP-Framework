package com.aiserver;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/cases")
public class CaseImageController {

    private final CaseImageService imageService;

    public CaseImageController(CaseImageService imageService) {
        this.imageService = imageService;
    }

    @PostMapping("/{caseId}/images")
    public CaseImage upload(
            @PathVariable Long caseId,
            @RequestParam("file") MultipartFile file
    ) throws Exception {
        return imageService.upload(caseId, file);
    }
}
