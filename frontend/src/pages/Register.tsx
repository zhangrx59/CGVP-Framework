import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { register } from "../api/auth";
import type { RegisterReq } from "../api/auth"; // ✅ 必须是 import type

export default function Register() {
  const nav = useNavigate();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [dept, setDept] = useState("");
  const [role, setRole] = useState<RegisterReq["role"]>("DOCTOR");

  const [adminKey, setAdminKey] = useState(""); // ⭐ NEW
  const [err, setErr] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);

    try {
      const req: RegisterReq = {
        username,
        password,
        role,
        dept: dept || undefined,
        adminKey: role === "ADMIN" ? adminKey : undefined, // ⭐ NEW
      };

      await register(req);

      alert("注册成功，请登录");
      nav("/login");
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "注册失败");
    }
  }

  return (
    <div style={{ maxWidth: 460 }}>
      <h3>注册</h3>

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

        <input
          value={dept}
          onChange={(e) => setDept(e.target.value)}
          placeholder="dept（可选）"
        />

        {/* ⭐ NEW：角色选择 */}
        <select value={role} onChange={(e) => setRole(e.target.value as any)}>
          <option value="DOCTOR">医生</option>
          <option value="NURSE">护士</option>
          <option value="ADMIN">管理员</option>
        </select>

        {/* ⭐ NEW：管理员密钥 */}
        {role === "ADMIN" && (
          <input
            value={adminKey}
            onChange={(e) => setAdminKey(e.target.value)}
            placeholder="管理员密钥（123456）"
            type="password"
          />
        )}

        <button type="submit">注册</button>
      </form>

      {err && <div style={{ color: "crimson", marginTop: 8 }}>{err}</div>}
    </div>
  );
}
