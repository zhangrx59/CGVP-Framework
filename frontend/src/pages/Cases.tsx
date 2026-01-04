import { useNavigate } from "react-router-dom";

export default function Cases() {
  const nav = useNavigate();

  return (
    <div
      style={{
        width: "100%",
        minHeight: "60vh",
        display: "flex",
        justifyContent: "center",   // ⭐ 整体居中
        alignItems: "center",
      }}
    >
      {/* ⭐ 两个按钮并排的容器 */}
      <div
        style={{
          display: "flex",
          gap: 48,                 // ⭐ 左右间距，显得稳重
        }}
      >
        <button
          className="landing-btn"
          onClick={() => nav("/cases/create")}
          style={{
            padding: "14px 34px",
            fontSize: 18,
            borderRadius: 16,
            minWidth: 220,          // ⭐ 保证两个按钮等宽
          }}
        >
          创建病例
        </button>

        <button
          className="landing-btn"
          onClick={() => nav("/cases/all")}
          style={{
            padding: "14px 34px",
            fontSize: 18,
            borderRadius: 16,
            minWidth: 220,          // ⭐ 与上面完全一致
          }}
        >
          查看病例
        </button>
      </div>
    </div>
  );
}
