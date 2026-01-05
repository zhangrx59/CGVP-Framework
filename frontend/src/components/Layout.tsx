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

  // ⭐ NEW：是否在业务首页 /cases（或其子路由）
  const inCasesModule = loc.pathname.startsWith("/cases");
  const isCasesHome = loc.pathname === "/cases";

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
            {/* 左侧占位，保证 brand 真正居中 */}
            <div style={{ width: 110 }} />

            {/* 标题放在 topbar 正中 */}
            <div
              className="brand"
              style={{
                flex: 1,
                textAlign: "center",
                fontSize: 32,
                fontWeight: 900,
                letterSpacing: 0.6,
              }}
            >
              大模型赋能的皮肤病诊断系统
            </div>

            {/* 右侧操作区固定宽度，避免把标题挤偏 */}
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
                  {/* ⭐ MODIFIED：返回逻辑修复
                      - 仅在 cases 模块内且不在 /cases 首页时显示“返回”
                      - 点击永远回到 /cases（业务首页），不依赖 history
                   */}
                  {inCasesModule && !isCasesHome && (
                    <span
                      onClick={() => nav("/cases")}
                      style={{
                        cursor: "pointer",
                        textDecoration: "underline",
                        opacity: 1,
                      }}
                    >
                      返回
                    </span>
                  )}

                  {/* 退出保持不变 */}
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
                <>
                  <Link to="/login" style={active("/login")}>
                    登录
                  </Link>

                  <Link to="/register" style={active("/register")}>
                    注册
                  </Link>

                  <Link to="/" style={active("/")}>
                    返回
                  </Link>
                </>
              )}
            </div>
          </div>

          <div style={{ height: 6 }} />
        </>
      )}

      {/* 中间页面内容 */}
      {children}

      {/* ✅ 非 Landing 页面：显示底部免责声明 */}
      {!isLanding && (
        <>
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

          {/* 登录后右下角问候语 */}
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
