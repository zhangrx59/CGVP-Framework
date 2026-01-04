import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { register } from "../api/auth";
import type { RegisterReq } from "../api/auth"; // ✅ type-only

export default function Register() {
  const nav = useNavigate();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [dept, setDept] = useState("");

  // ⭐ MODIFIED：默认身份改为 NURSE

  const [role, setRole] = useState<RegisterReq["role"] | "">("");

  const [adminKey, setAdminKey] = useState("");
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
        adminKey: role === "ADMIN" ? adminKey : undefined,
      };

      await register(req);
      alert("注册成功，请登录");
      nav("/login");
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "注册失败");
    }
  }

  // ⭐ NEW：统一一行输入框的外观（跟你现有输入框更一致）
  const rowStyle: React.CSSProperties = {
    position: "relative",
    display: "flex",
    alignItems: "center",
  };

  const selectStyle: React.CSSProperties = {
    width: "100%",
    borderRadius: 999,
    padding: "14px 44px 14px 18px", // 右侧留出箭头空间
    fontSize: 16,
    outline: "none",
    border: "1px solid rgba(255,255,255,0.18)",
    background: "rgba(255,255,255,0.06)",
    color: "rgba(255,255,255,0.92)",
    // 关键：去掉原生样式，自己画箭头
    appearance: "none",
    WebkitAppearance: "none",
    MozAppearance: "none",
  };

  const hintStyle: React.CSSProperties = {
    position: "absolute",
    left: 18,
    pointerEvents: "none",
    color: "rgba(255,255,255,0.35)", // ⭐ “身份”两个字虚化
    fontSize: 16,
  };

  const arrowStyle: React.CSSProperties = {
    position: "absolute",
    right: 16,
    pointerEvents: "none",
    color: "rgba(255,255,255,0.55)",
    fontSize: 14,
  };

return (
  // ⭐ MODIFIED：只负责把中间块水平居中（不改任何表单内容/高度/样式）
  <div
    style={{
      width: "100%",
      display: "flex",
      justifyContent: "center", // 水平居中
      paddingTop: 0,            // 你如果想整体往下挪，可以改这里（但不建议改高度相关）
    }}
  >
    {/* ⭐ 原来的容器完全不动 */}
    <div style={{ maxWidth: 460, width: "100%" }}>
      <h3>注册</h3>

      <form onSubmit={onSubmit} style={{ display: "grid", gap: 12 }}>
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

        <input
          value={dept}
          onChange={(e) => setDept(e.target.value)}
          placeholder="所在部门（可选）"
        />

        {/* ⭐ MODIFIED：身份栏 —— 和其他输入框同风格 + 右侧箭头 */}
        <div style={rowStyle}>
          <select
            value={role}
            onChange={(e) => setRole(e.target.value as any)}
            style={selectStyle}
          >
            {/* ⭐ 占位项，不可选 */}
            <option value="" disabled hidden>
              身份（请选择）
            </option>

            <option value="NURSE" style={{ color: "#111", background: "#fff" }}>
              护士
            </option>
            <option value="DOCTOR" style={{ color: "#111", background: "#fff" }}>
              医生
            </option>
            <option value="ADMIN" style={{ color: "#111", background: "#fff" }}>
              管理员
            </option>
          </select>
        </div>

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

      {err && <div style={{ color: "crimson", marginTop: 10 }}>{err}</div>}
    </div>
  </div>
);

}
