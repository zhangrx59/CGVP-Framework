import { useNavigate } from "react-router-dom";

export default function Cases() {
  const nav = useNavigate();

  return (
    <div
      style={{
        width: "100%",
        minHeight: "60vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, minmax(220px, 1fr))",
          gap: 26,
          alignItems: "center",
          justifyItems: "center",
        }}
      >
        <button
          className="landing-btn"
          onClick={() => nav("/cases/create")}
          style={{ padding: "14px 34px", fontSize: 18, borderRadius: 16, minWidth: 220 }}
        >
          创建病例
        </button>

        <button
          className="landing-btn"
          onClick={() => nav("/cases/all")}
          style={{ padding: "14px 34px", fontSize: 18, borderRadius: 16, minWidth: 220 }}
        >
          查看病例
        </button>

        <button
          className="landing-btn"
          onClick={() => nav("/cases/edit")}
          style={{ padding: "14px 34px", fontSize: 18, borderRadius: 16, minWidth: 220 }}
        >
          修改病例
        </button>

        <button
          className="landing-btn"
          onClick={() => nav("/cases/infer")}
          style={{ padding: "14px 34px", fontSize: 18, borderRadius: 16, minWidth: 220 }}
        >
          推理病例
        </button>
      </div>
    </div>
  );
}
