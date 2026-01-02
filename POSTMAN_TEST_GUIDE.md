# Postman 测试指南 - AI 皮肤病变诊断系统

## 📋 前置条件

1. **确保服务已启动**
   - Java 后端服务运行在：`http://localhost:8080`
   - Python AI 推理服务运行在：`http://127.0.0.1:8000`
   - MySQL 数据库已启动并连接
   - Redis 已启动

2. **Postman 设置**
   - 建议创建一个新的 Collection，命名为 "AI-Server-Tests"
   - 所有请求的基础 URL：`http://localhost:8080`

---

## 🧪 测试步骤（按顺序执行）

### 步骤 1：健康检查（可选）

**目的**：验证服务是否正常运行

**请求配置**：
- **Method**: `GET`
- **URL**: `http://localhost:8080/health`
- **Headers**: 无需设置
- **Body**: 无

**预期响应**：
```json
{
    "status": "ok"
}
```

**状态码**: `200 OK`

---

### 步骤 2：用户注册

**目的**：创建一个新用户账户

**请求配置**：
- **Method**: `POST`
- **URL**: `http://localhost:8080/auth/register`
- **Headers**: 
  ```
  Content-Type: application/json
  ```
- **Body** (选择 `raw` → `JSON`):
```json
{
    "username": "doctor001",
    "password": "password123",
    "role": "DOCTOR",
    "dept": "皮肤科"
}
```

**预期响应**：
```json
{
    "id": 1,
    "username": "doctor001",
    "role": "DOCTOR",
    "dept": "皮肤科"
}
```

**状态码**: `200 OK`

**注意事项**：
- 如果用户名已存在，会返回 `400 Bad Request`，错误信息：`"username already exists"`
- `role` 可选值：`DOCTOR`、`NURSE`、`ADMIN`
- `dept` 可以为空或任意字符串

---

### 步骤 3：用户登录

**目的**：获取 JWT Token（后续所有需要认证的接口都需要此 Token）

**请求配置**：
- **Method**: `POST`
- **URL**: `http://localhost:8080/auth/login`
- **Headers**: 
  ```
  Content-Type: application/json
  ```
- **Body** (选择 `raw` → `JSON`):
```json
{
    "username": "doctor001",
    "password": "password123"
}
```

**预期响应**：
```json
{
    "user": {
        "id": 1,
        "username": "doctor001",
        "role": "DOCTOR",
        "dept": "皮肤科"
    },
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhaS1zZXJ2ZXIiLCJzdWIiOiIxIiwiaWF0IjoxNzM1MjM0NTY3LCJleHAiOjE3MzUyNDE3NjcsInVzZXJuYW1lIjoiZG9jdG9yMDAxIiwicm9sZSI6IkRPQ1RPUiIsImRlcHQiOiLpkrHlp5PljLoifQ.xxxxx"
}
```

**状态码**: `200 OK`

**重要**：
- **复制 `token` 字段的值**，后续所有需要认证的请求都需要在 Header 中添加：
  ```
  Authorization: Bearer {你的token}
  ```
- Token 有效期为 120 分钟（2小时）

---

### 步骤 4：创建病例

**目的**：创建一个新的病例记录

**请求配置**：
- **Method**: `POST`
- **URL**: `http://localhost:8080/cases`
- **Headers**: 
  ```
  Content-Type: application/json
  Authorization: Bearer {步骤3获取的token}
  ```
- **Body** (选择 `raw` → `JSON`):
```json
{
    "patientName": "张三",
    "patientSex": "M",
    "patientAge": 45,
    "chiefComplaint": "背部发现皮肤病变，持续3个月",
    "history": "无特殊病史"
}
```

**预期响应**：
```json
{
    "id": 1,
    "patientName": "张三",
    "patientSex": "M",
    "patientAge": 45,
    "chiefComplaint": "背部发现皮肤病变，持续3个月",
    "history": "无特殊病史",
    "status": "NEW",
    "createdBy": 1,
    "dept": "皮肤科"
}
```

**状态码**: `200 OK`

**注意事项**：
- `patientSex` 可选值：`M`（男）、`F`（女）、`U`（未知）
- `chiefComplaint` 是必填字段（不能为空）
- 记录返回的 `id` 值（这里是 `1`），后续步骤需要使用

---

### 步骤 5：上传病例图片

**目的**：为病例上传皮肤病变图片

**请求配置**：
- **Method**: `POST`
- **URL**: `http://localhost:8080/cases/{caseId}/images`
  - 将 `{caseId}` 替换为步骤4返回的病例ID（例如：`1`）
  - 完整URL示例：`http://localhost:8080/cases/1/images`
- **Headers**: 
  ```
  Authorization: Bearer {步骤3获取的token}
  ```
  **注意**：不要设置 `Content-Type`，Postman 会自动设置为 `multipart/form-data`
- **Body** (选择 `form-data`):
  - Key: `file` (类型选择 `File`)
  - Value: 点击 `Select Files`，选择一张皮肤病变图片（支持 jpg、png 等格式）

**预期响应**：
```json
{
    "id": 1,
    "caseId": 1,
    "fileName": "lesion.jpg",
    "filePath": "C:\\Users\\zhangrx59\\IdeaProjects\\AI-server\\data\\uploads\\case-1\\xxxxx.jpg",
    "contentType": "image/jpeg",
    "fileSize": 245678,
    "createdAt": "2025-01-01T10:30:00Z"
}
```

**状态码**: `200 OK`

**注意事项**：
- 只能上传图片文件（`image/*`）
- 文件大小限制：最大 50MB
- 只能为**自己创建的病例**上传图片（权限控制）
- 一个病例可以上传多张图片，系统会使用最新的一张进行推理

---

### 步骤 6：触发 AI 推理

**目的**：提交推理任务，系统会异步处理

**请求配置**：
- **Method**: `POST`
- **URL**: `http://localhost:8080/cases/{caseId}/infer`
  - 将 `{caseId}` 替换为步骤4返回的病例ID（例如：`1`）
  - 完整URL示例：`http://localhost:8080/cases/1/infer`
- **Headers**: 
  ```
  Authorization: Bearer {步骤3获取的token}
  ```
- **Body**: 无

**预期响应**：
```json
{
    "jobId": 1,
    "status": "QUEUED"
}
```

**状态码**: `200 OK`

**重要**：
- **记录返回的 `jobId` 值**（这里是 `1`），后续查询需要使用
- 任务状态会经历：`QUEUED` → `RUNNING` → `SUCCEEDED` 或 `FAILED`
- 推理是异步进行的，需要等待几秒到几分钟（取决于 Python 服务处理速度）

---

### 步骤 7：查询任务状态（轮询）

**目的**：检查推理任务是否完成

**请求配置**：
- **Method**: `GET`
- **URL**: `http://localhost:8080/jobs/{jobId}`
  - 将 `{jobId}` 替换为步骤6返回的jobId（例如：`1`）
  - 完整URL示例：`http://localhost:8080/jobs/1`
- **Headers**: 
  ```
  Authorization: Bearer {步骤3获取的token}
  ```
- **Body**: 无

**可能的状态响应**：

**情况1：任务还在队列中**
```json
{
    "id": 1,
    "caseId": 1,
    "createdBy": 1,
    "status": "QUEUED",
    "attemptCount": 0,
    "lastError": null,
    "createdAt": "2025-01-01T10:35:00Z",
    "startedAt": null,
    "finishedAt": null
}
```

**情况2：任务正在处理中**
```json
{
    "id": 1,
    "caseId": 1,
    "createdBy": 1,
    "status": "RUNNING",
    "attemptCount": 1,
    "lastError": null,
    "createdAt": "2025-01-01T10:35:00Z",
    "startedAt": "2025-01-01T10:35:05Z",
    "finishedAt": null
}
```

**情况3：任务成功完成** ✅
```json
{
    "id": 1,
    "caseId": 1,
    "createdBy": 1,
    "status": "SUCCEEDED",
    "attemptCount": 1,
    "lastError": null,
    "createdAt": "2025-01-01T10:35:00Z",
    "startedAt": "2025-01-01T10:35:05Z",
    "finishedAt": "2025-01-01T10:36:30Z"
}
```

**情况4：任务失败**
```json
{
    "id": 1,
    "caseId": 1,
    "createdBy": 1,
    "status": "FAILED",
    "attemptCount": 1,
    "lastError": "IllegalStateException: FastAPI infer failed, status=500",
    "createdAt": "2025-01-01T10:35:00Z",
    "startedAt": "2025-01-01T10:35:05Z",
    "finishedAt": "2025-01-01T10:35:10Z"
}
```

**状态码**: `200 OK`

**建议**：
- 在 Postman 中可以设置自动重试或使用 Collection Runner
- 或者手动刷新几次，直到 `status` 变为 `SUCCEEDED` 或 `FAILED`
- 通常推理需要 30 秒到 2 分钟

---

### 步骤 8：查询病例的所有推理记录 ⭐（核心功能）

**目的**：查看某个病例的所有历史推理记录，包括最新的报告

**请求配置**：
- **Method**: `GET`
- **URL**: `http://localhost:8080/cases/{caseId}/inferences`
  - 将 `{caseId}` 替换为步骤4返回的病例ID（例如：`1`）
  - 完整URL示例：`http://localhost:8080/cases/1/inferences`
- **Headers**: 
  ```
  Authorization: Bearer {步骤3获取的token}
  ```
- **Body**: 无

**预期响应**（成功完成推理后）：
```json
[
    {
        "id": 1,
        "jobId": 1,
        "caseId": 1,
        "predLabel": "mel",
        "probsJson": {
            "akiec": 0.15,
            "bcc": 0.20,
            "nev": 0.10,
            "mel": 0.55
        },
        "reportJson": {
            "诊断结果": "黑色素瘤",
            "诊断类别": "mel",
            "各类别概率": {
                "光化性角化病": 15.0,
                "基底细胞癌": 20.0,
                "痣": 10.0,
                "黑色素瘤": 55.0
            },
            "患者信息": {
                "年龄": 45,
                "性别": "M",
                "区域": ""
            },
            "临床特征": {
                "直径": "",
                "瘙痒": "",
                "疼痛": "",
                "是否长大": "",
                "形态变化": "",
                "出血": "",
                "是否隆起": ""
            },
            "病史信息": {
                "皮肤癌病史": "",
                "癌症病史": "",
                "是否吸烟": "",
                "是否饮酒": ""
            },
            "临床记录": "45-year-old male with a skin lesion on the unknown region. Past history of skin cancer: no; other malignancies: no. Lifestyle: smoking no, alcohol no, pesticide exposure no. Living environment: tap water no, sewer system no. Current symptoms and signs: pruritus absent, growth absent, pain absent, morphologic change absent, bleeding absent, elevation absent."
        },
        "modelVersion": "medgemma-4b-it+lora@202512",
        "createdAt": "2025-01-01T10:36:30Z",
        "resultJson": "{\"predLabel\":\"mel\",\"probs\":{\"akiec\":0.15,\"bcc\":0.20,\"nev\":0.10,\"mel\":0.55},\"clinicalNote\":\"...\",\"reportJson\":{...},\"modelVersion\":\"medgemma-4b-it+lora@202512\"}"
    }
]
```

**状态码**: `200 OK`

**说明**：
- 返回的是一个数组，按时间倒序排列（最新的在前）
- 每个元素包含完整的推理结果，包括：
  - `predLabel`：预测的诊断类别（akiec/bcc/nev/mel）
  - `probsJson`：各类别的概率（0-1之间的小数）
  - `reportJson`：**中文结构化报告**（这是核心功能）
  - `modelVersion`：使用的模型版本
  - `resultJson`：原始完整返回（用于兜底）

**如果还没有推理记录**：
```json
[]
```
返回空数组

---

### 步骤 9：查询特定推理的完整结果 ⭐（核心功能）

**目的**：通过 jobId 查询某次推理的完整结果

**请求配置**：
- **Method**: `GET`
- **URL**: `http://localhost:8080/inferences/{jobId}`
  - 将 `{jobId}` 替换为步骤6返回的jobId（例如：`1`）
  - 完整URL示例：`http://localhost:8080/inferences/1`
- **Headers**: 
  ```
  Authorization: Bearer {步骤3获取的token}
  ```
- **Body**: 无

**预期响应**（成功完成推理后）：
```json
{
    "id": 1,
    "jobId": 1,
    "caseId": 1,
    "predLabel": "mel",
    "probsJson": {
        "akiec": 0.15,
        "bcc": 0.20,
        "nev": 0.10,
        "mel": 0.55
    },
    "reportJson": {
        "诊断结果": "黑色素瘤",
        "诊断类别": "mel",
        "各类别概率": {
            "光化性角化病": 15.0,
            "基底细胞癌": 20.0,
            "痣": 10.0,
            "黑色素瘤": 55.0
        },
        "患者信息": {
            "年龄": 45,
            "性别": "M",
            "区域": ""
        },
        "临床特征": {
            "直径": "",
            "瘙痒": "",
            "疼痛": "",
            "是否长大": "",
            "形态变化": "",
            "出血": "",
            "是否隆起": ""
        },
        "病史信息": {
            "皮肤癌病史": "",
            "癌症病史": "",
            "是否吸烟": "",
            "是否饮酒": ""
        },
        "临床记录": "45-year-old male with a skin lesion on the unknown region. Past history of skin cancer: no; other malignancies: no. Lifestyle: smoking no, alcohol no, pesticide exposure no. Living environment: tap water no, sewer system no. Current symptoms and signs: pruritus absent, growth absent, pain absent, morphologic change absent, bleeding absent, elevation absent."
    },
    "modelVersion": "medgemma-4b-it+lora@202512",
    "createdAt": "2025-01-01T10:36:30Z",
    "resultJson": "{\"predLabel\":\"mel\",\"probs\":{\"akiec\":0.15,\"bcc\":0.20,\"nev\":0.10,\"mel\":0.55},\"clinicalNote\":\"...\",\"reportJson\":{...},\"modelVersion\":\"medgemma-4b-it+lora@202512\"}"
}
```

**状态码**: `200 OK`

**如果 jobId 不存在**：
```json
{
    "error": "inference result not found for jobId: 999"
}
```

**状态码**: `500 Internal Server Error`（实际应该返回 404，但当前实现是 500）

---

## 🔍 完整测试流程总结

### 快速测试清单

1. ✅ **健康检查** → `GET /health`
2. ✅ **注册用户** → `POST /auth/register`
3. ✅ **登录获取Token** → `POST /auth/login` → **复制token**
4. ✅ **创建病例** → `POST /cases` → **记录caseId**
5. ✅ **上传图片** → `POST /cases/{caseId}/images`
6. ✅ **触发推理** → `POST /cases/{caseId}/infer` → **记录jobId**
7. ✅ **查询任务状态** → `GET /jobs/{jobId}` → **等待SUCCEEDED**
8. ✅ **查询病例所有推理** → `GET /cases/{caseId}/inferences` → **验证reportJson存在**
9. ✅ **查询特定推理结果** → `GET /inferences/{jobId}` → **验证完整报告**

---

## 🐛 常见问题排查

### 问题1：401 Unauthorized
**原因**：Token 过期或未设置
**解决**：重新登录获取新 Token，确保 Header 中有 `Authorization: Bearer {token}`

### 问题2：403 Forbidden
**原因**：权限不足（例如：尝试操作其他用户创建的病例）
**解决**：确保使用创建病例的用户登录

### 问题3：推理任务一直 QUEUED
**原因**：
- Python 服务未启动
- Redis 未启动
- Worker 未运行
**解决**：检查 Python 服务（`http://127.0.0.1:8000/health`）和 Redis 连接

### 问题4：推理任务 FAILED
**原因**：
- Python 服务返回错误
- 图片格式不支持
- 元数据字段缺失
**解决**：查看 `lastError` 字段，检查 Python 服务日志

### 问题5：查询推理结果返回空
**原因**：推理任务还未完成或失败
**解决**：先查询任务状态，确保状态为 `SUCCEEDED`

---

## 📝 Postman Collection 变量设置建议

为了更方便测试，可以在 Postman 中设置环境变量：

1. 创建环境（Environment）：
   - `baseUrl`: `http://localhost:8080`
   - `token`: （登录后自动设置）
   - `caseId`: （创建病例后自动设置）
   - `jobId`: （触发推理后自动设置）

2. 使用变量：
   - URL: `{{baseUrl}}/cases/{{caseId}}/infer`
   - Header: `Authorization: Bearer {{token}}`

3. 使用 Tests 脚本自动保存变量：
   ```javascript
   // 登录后保存token
   if (pm.response.code === 200) {
       var jsonData = pm.response.json();
       pm.environment.set("token", jsonData.token);
   }
   
   // 创建病例后保存caseId
   if (pm.response.code === 200) {
       var jsonData = pm.response.json();
       pm.environment.set("caseId", jsonData.id);
   }
   ```

---

## ✅ 验收标准

测试成功的标志：

1. ✅ 能够成功注册和登录
2. ✅ 能够创建病例并上传图片
3. ✅ 能够触发推理任务
4. ✅ 推理任务状态最终变为 `SUCCEEDED`
5. ✅ **`GET /cases/{caseId}/inferences` 返回的列表中，最新记录包含 `reportJson` 字段，且 `reportJson` 不为 null**
6. ✅ **`GET /inferences/{jobId}` 返回的结果中包含完整的 `reportJson`，且包含中文诊断结果**

**核心验收点**：步骤8和步骤9必须能看到 `reportJson` 字段，且内容为中文结构化报告。

