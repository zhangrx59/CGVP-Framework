package com.aiserver;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

@Service
public class CaseImageService {

    private final CaseRepo caseRepo;
    private final CaseImageRepo imageRepo;

    @Value("${app.upload.base-dir}")
    private String baseDir;

    public CaseImageService(CaseRepo caseRepo, CaseImageRepo imageRepo) {
        this.caseRepo = caseRepo;
        this.imageRepo = imageRepo;
    }

    public CaseImage upload(Long caseId, MultipartFile file) throws IOException {
        JwtService.JwtUser me = currentUser();

        // 1️⃣ 校验病例存在
        Case c = caseRepo.findById(caseId)
                .orElseThrow(() -> new IllegalArgumentException("case not found"));

        // 2️⃣ 校验权限（只能上传自己创建的病例）
        if (!c.getCreatedBy().equals(me.userId())) {
            throw new SecurityException("no permission");
        }

        // 3️⃣ 校验文件类型
        if (file.isEmpty()) {
            throw new IllegalArgumentException("empty file");
        }
        if (!file.getContentType().startsWith("image/")) {
            throw new IllegalArgumentException("only image allowed");
        }

        // 4️⃣ 生成目录与文件名
        File dir = new File(baseDir + "/case-" + caseId);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        String ext = file.getOriginalFilename().substring(
                file.getOriginalFilename().lastIndexOf(".")
        );
        String newName = UUID.randomUUID() + ext;

        File dest = new File(dir, newName);
        file.transferTo(dest);

        // 5️⃣ 保存数据库记录
        CaseImage img = new CaseImage();
        img.setCaseId(caseId);
        img.setFileName(file.getOriginalFilename());
        img.setFilePath(dest.getAbsolutePath());
        img.setContentType(file.getContentType());
        img.setFileSize(file.getSize());
        System.out.println("UPLOAD BASE DIR = " + new File(baseDir).getAbsolutePath());
        imageRepo.save(img);
        return img;

    }

    private JwtService.JwtUser currentUser() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        return (JwtService.JwtUser) auth.getPrincipal();
    }

}
