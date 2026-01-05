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
      {/* ⭐ NEW：2×2 网格布局 */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, minmax(220px, 1fr))",
          gap: 22,
          maxWidth: 520,
          width: "100%",
        }}
      >
        {/* 第一排 */}
        <button
          className="card"
          onClick={() => nav("/cases/create")}
          style={{
            padding: "14px 34px",
            fontSize: 18,
            borderRadius: 16,
          }}
        >
          创建病例
        </button>

        <button
          className="card"
          onClick={() => nav("/cases/all")}
          style={{
            padding: "14px 34px",
            fontSize: 18,
            borderRadius: 16,
          }}
        >
          查看病例
        </button>

        {/* 第二排 */}
        <button
          className="card"
          onClick={() => nav("/cases/edit")}
          style={{
            padding: "14px 34px",
            fontSize: 18,
            borderRadius: 16,
          }}
        >
          修改病例
        </button>

        <button
          className="card"
          onClick={() => nav("/cases/infer")}
          style={{
            padding: "14px 34px",
            fontSize: 18,
            borderRadius: 16,
          }}
        >
          推理病例
        </button>
      </div>
    </div>
  );
}
