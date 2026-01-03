import { useNavigate } from "react-router-dom";
import { setAuthToken } from "../api/http";

export default function Landing() {
  const nav = useNavigate();

  function onLogout() {
    const ok = window.confirm("确定退出登录？");
    if (!ok) return;

    localStorage.removeItem("token");
    localStorage.removeItem("role"); // ⭐ NEW：顺带清掉 role
    setAuthToken(null);
    nav("/");
  }

  return (
    <div style={{ display: "grid", placeItems: "center", minHeight: "70vh" }}>
      <div style={{ width: "min(720px, 92vw)" }}>
        <div className="card">
          <div style={{ fontSize: 20, fontWeight: 900 }}>
            大模型赋能的皮肤病诊断系统
          </div>

          <div style={{ height: 14 }} />

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button onClick={() => nav("/login")}>登录</button>
            <button onClick={() => nav("/register")}>注册</button>
            <button onClick={onLogout}>退出</button>
          </div>

          <div className="muted" style={{ marginTop: 18, fontWeight: 800, fontSize: 13 }}>
            免责声明：本系统输出仅用于临床决策支持参考，不能替代医生诊断。
          </div>
        </div>
      </div>
    </div>
  );
}
