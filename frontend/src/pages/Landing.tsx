import { useNavigate } from "react-router-dom";

export default function Landing() {
  const nav = useNavigate();

  function logout() {
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    localStorage.removeItem("userId");
    localStorage.removeItem("username");
    nav("/");
  }

  return (
    <div
      style={{
        minHeight: "60vh",
        width: "100%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        position: "relative",
        padding: 24,
      }}
    >
      {/* ⭐ 标题：居中 + 更大 */}
      <div
        style={{
          textAlign: "center",
          fontSize: 42,          // 字体明显加大
          fontWeight: 800,
          letterSpacing: 1,
          color: "rgba(255,255,255,0.96)",
        }}
      >
        大模型赋能的皮肤病诊断系统
      </div>

      {/* ⭐ 按钮区域：与标题距离加大 */}
      <div
        style={{
          marginTop: 60,        // 与 title 的距离明显加大
          display: "flex",
          gap: 36,              // 按钮之间距离加大
        }}
      >
        <button
          className="landing-btn"
          onClick={() => nav("/login")}
          style={{
            padding: "14px 34px",
            fontSize: 18,
            borderRadius: 16,
          }}
        >
          登录
        </button>

        <button
          className="landing-btn"
          onClick={() => nav("/register")}
          style={{
            padding: "14px 34px",
            fontSize: 18,
            borderRadius: 16,
          }}
        >
          注册
        </button>

      </div>

      {/* ⭐ 免责声明：整个页面左下角 */}
      <div
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
    </div>
  );
}
