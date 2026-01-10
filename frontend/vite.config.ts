// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // ✅ 让 Vite 对本机所有网卡开放（127.0.0.1 / 本机 IP 都能访问）
    host: true,          // 兼容 Cloudflare & 本地浏览器

    // ✅ 固定端口，防止占用时悄悄改成 5174 之类
    port: 5173,
    strictPort: true,

    /**
     * ✅ 重点：允许 Cloudflare Tunnel 过来的 Host
     *  - ".trycloudflare.com"：放行所有 *.trycloudflare.com
     *  - "localhost" / "127.0.0.1"：本地开发访问也不会被拦
     */
    allowedHosts: [".trycloudflare.com", "localhost", "127.0.0.1"],

    // ✅ 保留你现在这套 proxy + rewrite 的逻辑（这是你验证过“能通后端”的配置）
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
