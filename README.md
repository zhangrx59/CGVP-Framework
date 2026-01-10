# 毕业设计启动指南

## 1. 后端启动
打开 IntelliJ IDEA  
运行 `AiServerApplication.java`

---

## 2. 前端启动
打开 PowerShell，输入：
```bash
cd frontend
npm run dev
```
启动前端服务

---

## 3. 推理服务启动
打开 PowerShell，输入：
```bash
cd Infer
conda activate Monadepth   # 激活 Python 环境
uvicorn main:app --host 127.0.0.1 --port 8000
```
开启推理服务

---

## 4. Cloudflare 公网访问
打开 `cmd`，输入：
```bash
cd C:\
cloudflared.exe tunnel --url http://localhost:5173
```
等待日志出现类似：
```
2026-01-10T03:58:09Z INF |  https://dramatic-alan-author-decided.trycloudflare.com
```
该链接即为公开访问地址（每次启动不同）
