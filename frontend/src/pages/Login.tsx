import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../api/auth";
import { setAuthToken } from "../api/http";

export default function Login() {
  const nav = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);

    try {
      const data = await login({ username, password });

      // ⭐ MODIFIED：保存 token + role（用于前端分流）
      localStorage.setItem("token", data.token);
      localStorage.setItem("role", data.user.role); // ⭐ NEW
      setAuthToken(data.token);

      nav("/cases");
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "登录失败");
    }
  }

  return (
    <div style={{ maxWidth: 420 }}>
      <h3>登录</h3>

      <form onSubmit={onSubmit} style={{ display: "grid", gap: 8 }}>
        <input
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="username"
        />
        <input
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="password"
          type="password"
        />
        <button type="submit">登录</button>
      </form>

      {err && <div style={{ color: "crimson", marginTop: 8 }}>{err}</div>}
    </div>
  );
}
