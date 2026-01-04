import { Link, useLocation, useNavigate } from "react-router-dom";
import { setAuthToken } from "../api/http";

export default function Layout({ children }: { children: React.ReactNode }) {
  const nav = useNavigate();
  const loc = useLocation();
  const token = localStorage.getItem("token");
  const username = localStorage.getItem("username") || "";
  const role = (localStorage.getItem("role") || "").toUpperCase();

  const roleCN =
    role === "DOCTOR" ? "医生" : role === "NURSE" ? "护士" : role === "ADMIN" ? "管理员" : "";

  // ✅ Landing 页面：只要中间卡片
  const isLanding = loc.pathname === "/";

  function logout() {
    localStorage.removeItem("token");
    setAuthToken(null);
    nav("/");
  }

  const active = (path: string) =>
    loc.pathname.startsWith(path)
      ? { opacity: 1, textDecoration: "underline" }
      : undefined;

  return (
    <div className="container">
      {/* ✅ 非 Landing 页面：显示顶部栏 */}
      {!isLanding && (
        <>
          <div className="topbar">
            {/* ⭐ MODIFIED：左侧占位，保证 brand 真正居中 */}
            <div style={{ width: 110 }} />

            {/* ⭐ MODIFIED：标题放在 topbar 正中 + 字体加大 */}
            <div
              className="brand"
              style={{
                flex: 1,                 // ⭐ NEW：占据中间空间
                textAlign: "center",     // ⭐ NEW：居中
                fontSize: 32,            // ⭐ MODIFIED：适当加大（可调 24/26）
                fontWeight: 900,         // ⭐ MODIFIED：更有力量
                letterSpacing: 0.6,
              }}
            >
              大模型赋能的皮肤病诊断系统
            </div>

            {/* ⭐ MODIFIED：右侧操作区固定宽度，避免把标题挤偏 */}
            <div
              style={{
                width: 150,
                display: "flex",
                justifyContent: "flex-end",
                alignItems: "center",
                gap: 15,
              }}
            >
            {token ? (
              <>
                {/* ⭐ MODIFIED：病例 -> 返回 */}
                <Link to="/cases" style={active("/cases")}>
                  返回
                </Link>

                {/* ⭐ MODIFIED：退出改成和 Link 一样的下划线风格 */}
                <span
                  onClick={logout}
                  style={{
                    cursor: "pointer",
                    textDecoration: "underline",
                    opacity: 1,
                  }}
                >
                  退出
                </span>
              </>
            ) : (

              // ⭐ MODIFIED：未登录时显示：登录 / 注册 / 返回
              <>
                <Link to="/login" style={active("/login")}>
                  登录
                </Link>

                {/* ⭐ NEW：注册 */}
                <Link to="/register" style={active("/register")}>
                  注册
                </Link>

                {/* ⭐ NEW：返回（回到 Landing 首页） */}
                <Link to="/" style={active("/")}>
                  返回
                </Link>
              </>
            )}

            </div>
          </div>

          {/* ⭐ MODIFIED：缩短 topbar 与页面内容距离（原来 14） */}
          <div style={{ height: 6 }} />
        </>
      )}

      {/* 中间页面内容 */}
      {children}

      {/* ✅ 非 Landing 页面：显示底部免责声明 */}
      {!isLanding && (
        <>
          {/* ⭐ MODIFIED：缩短页面内容与免责声明距离（原来 10） */}
          <div style={{ height: 0 }} />
          <div
            className="muted"
            style={{
              position: "fixed",
              left: 24,
              bottom: 20,
              fontSize: 16,
              fontWeight: 600,
              color: "rgba(255,255,255,0.7)",
              lineHeight: 1.5,
            }}
          >
            免责声明：本系统输出仅用于临床决策支持参考，不能替代医生诊断。
          </div>

            {/* ⭐ NEW：登录后右下角问候语 */}
          {token && username && roleCN && (
            <div
              style={{
                position: "fixed",
                right: 24,
                bottom: 16,
                fontSize: 14,
                fontWeight: 800,
                opacity: 0.85,
                color: "rgba(255,255,255,0.85)",
                textShadow: "0 2px 12px rgba(0,0,0,0.35)",
                pointerEvents: "none",
              }}
            >
              {username} {roleCN}，您好！
            </div>
          )}
        </>
      )}
    </div>
  );
}
