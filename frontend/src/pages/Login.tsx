import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../api/auth";
import { setAuthToken } from "../api/http"; // ⭐ NEW：登录后立刻把 token 写入 axios header

export default function Login() {
  const nav = useNavigate();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);

    try {
      const resp = await login({ username, password });
      localStorage.setItem("token", resp.token);
      localStorage.setItem("role", resp.user.role);
      localStorage.setItem("userId", String(resp.user.id));
      localStorage.setItem("username", resp.user.username);

      setAuthToken(resp.token); // ⭐ NEW：关键！不刷新页面也能立刻带上 Authorization


      nav("/cases");
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "登录失败");
    }
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100%",
        position: "relative",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        paddingTop: 40, // ⭐ MODIFIED：留出上方 Header 的空间
      }}
    >
      {/* ⭐ MODIFIED：删除“中间的大标题”，只保留 Header 上的标题 */}

      <div style={{ width: 420 }}>
        <h3 style={{ marginBottom: 20 }}>登录</h3>

        <form onSubmit={onSubmit} style={{ display: "grid", gap: 14 }}>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="用户名"
          />

          <input
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="密码"
            type="password"
          />

          <button type="submit">登录</button>
        </form>

      </div>
    </div>
  );
}
