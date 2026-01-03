import { Link, useLocation, useNavigate } from "react-router-dom";
import { setAuthToken } from "../api/http";

export default function Layout({ children }: { children: React.ReactNode }) {
  const nav = useNavigate();
  const loc = useLocation();
  const token = localStorage.getItem("token");

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
            <div className="brand">大模型赋能的皮肤病诊断系统</div>
            <div className="spacer" />

            {token ? (
              <>
                <Link to="/cases" style={active("/cases")}>
                  病例
                </Link>
                <button onClick={logout}>退出</button>
              </>
            ) : (
              <Link to="/login" style={active("/login")}>
                登录
              </Link>
            )}
          </div>
          <div style={{ height: 14 }} />
        </>
      )}

      {/* 中间页面内容 */}
      {children}

      {/* ✅ 非 Landing 页面：显示底部免责声明 */}
      {!isLanding && (
        <>
          <div style={{ height: 30 }} />
          <div className="muted" style={{ fontSize: 12, fontWeight:800, opacity: 0.8 }}>
            免责声明：本系统输出仅用于临床决策支持参考，不能替代医生诊断。
          </div>
        </>
      )}
    </div>
  );
}
