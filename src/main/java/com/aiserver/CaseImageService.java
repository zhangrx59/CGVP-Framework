package com.aiserver;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.core.io.FileSystemResource;
import java.io.File;
import java.io.IOException;
import java.util.UUID;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.http.MediaType;
import org.springframework.http.HttpHeaders;
import java.io.File;
import java.nio.file.Files;      // ⭐ NEW：覆盖上传时删除旧文件
import java.util.List;           // ⭐ NEW：覆盖上传时查询旧记录


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


        // ⭐ NEW：覆盖上传（同一病例只保留一张图）
        // 先删除该病例旧的图片记录 + 对应磁盘文件
        List<CaseImage> olds = imageRepo.findByCaseId(caseId);
        if (olds != null && !olds.isEmpty()) {
            for (CaseImage old : olds) {
                try {
                    if (old.getFilePath() != null && !old.getFilePath().isBlank()) {
                        Files.deleteIfExists(new File(old.getFilePath()).toPath());
                    }
                } catch (Exception ignore) {
                    // ⭐ NEW：旧文件删除失败不影响本次上传（避免历史脏数据导致无法上传）
                }
            }
            imageRepo.deleteAll(olds);
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


    public List<CaseImage> list(Long caseId) {
        // 你现有系统里，“查看病例”只给 DOCTOR/ADMIN，通常不需要 createdBy 限制；
        // 如果你希望仍然限制为创建者可见，就在这里复用 currentUser() + createdBy 校验。
        return imageRepo.findByCaseId(caseId);
    }

    public ResponseEntity<Resource> raw(Long caseId, Long imageId) {
        CaseImage img = imageRepo.findById(imageId)
                .orElseThrow(() -> new IllegalArgumentException("image not found"));

        if (!img.getCaseId().equals(caseId)) {
            throw new IllegalArgumentException("image not in this case");
        }

        File f = new File(img.getFilePath());
        if (!f.exists()) {
            throw new IllegalArgumentException("file missing on disk");
        }

        Resource res = new FileSystemResource(f);

        MediaType mt = MediaType.APPLICATION_OCTET_STREAM;
        try {
            if (img.getContentType() != null && img.getContentType().startsWith("image/")) {
                mt = MediaType.parseMediaType(img.getContentType());
            }
        } catch (Exception ignored) {}

        return ResponseEntity.ok()
                .contentType(mt)
                .header(HttpHeaders.CACHE_CONTROL, "no-cache")
                .body(res);
    }
}
