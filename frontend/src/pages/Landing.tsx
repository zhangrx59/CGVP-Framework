import { useNavigate } from "react-router-dom";

export default function Landing() {
  const nav = useNavigate();

  function onLogin() {
    nav("/login"); // 等价于 http://localhost:5173/login
  }

  function onExit() {
    const ok = window.confirm("确定退出？");
    if (!ok) return;

    // 浏览器安全限制：通常只能关闭由 window.open 打开的窗口/标签页
    // 所以这里做“尽力而为 + 降级”
    window.close();

    // 降级：如果无法关闭（大多数情况下），就跳转到 about:blank 并提示用户手动关闭
    setTimeout(() => {
      // 如果还没关闭，基本说明被拦截
      // 跳转到空白页，模拟“退出”
      window.location.href = "about:blank";
    }, 50);
  }

  return (
    <div className="grid" style={{ gap: 18 }}>
      <div className="card" style={{ padding: 26 }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 28, fontWeight: 800, letterSpacing: 1 }}>
            大模型赋能的皮肤病诊断系统
          </div>

          <div className="row" style={{ justifyContent: "center", marginTop: 18 }}>
            <button onClick={onLogin} style={{ minWidth: 40 }}>
              登录
            </button>
            <button onClick={onExit} style={{ minWidth: 40 }}>
              退出
            </button>
          </div>

          <div className="muted" style={{marginTop: 18, fontWeight: 800, fontSize: 13 }}>
            免责声明：本系统输出仅用于临床决策支持参考，不能替代医生诊断。
          </div>
        </div>
      </div>
    </div>
  );
}
