console.log(">>> Vite config loaded, allowedHosts =", [".trycloudflare.com"]);
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // 让 Vite 监听 0.0.0.0，而不是只监听 localhost
    host: true,

    // 固定端口
    port: 5173,
    strictPort: true,              // 新增：端口被占用就直接报错，防止悄悄改成 5174 之类

    /**
     * 关键：只用「字符串数组」形式，别用 true
     *  - ".trycloudflare.com" 表示允许所有 *.trycloudflare.com
     *  - localhost / 127.0.0.1 会自动放行，不用写
     */
    allowedHosts: [".trycloudflare.com"],

    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
  },
});
