import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listAllCases, deleteCase } from "../api/cases"; // ⭐ MODIFIED：引入 deleteCase
import type { CaseView } from "../api/cases"; // ✅ type-only

export default function AllCases() {
  const nav = useNavigate();
  const [items, setItems] = useState<CaseView[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const role = (localStorage.getItem("role") || "").toUpperCase();
  const canDelete = role === "DOCTOR" || role === "ADMIN"; // ⭐ NEW

  useEffect(() => {

    (async () => {
      try {
        const data = await listAllCases();
        setItems(data);
      } catch (e: any) {
        setErr(e?.response?.data?.message || e?.message || "加载失败");
      } finally {
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ⭐ NEW：删除逻辑（最小化，不引入额外状态管理）
  async function onDelete(id: number) {
    const ok = window.confirm(`确认删除病例 #${id} 吗？此操作不可恢复。`);
    if (!ok) return;

    try {
      await deleteCase(id);
      // ⭐ NEW：本地移除，避免重新请求
      setItems((prev) => prev.filter((x) => x.id !== id));
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "删除失败");
    }
  }

  if (loading) return <div>加载中...</div>;

  if (err) {
    return (
      <div style={{ maxWidth: 720 }}>
        <h3>所有病例</h3>
        <div style={{ color: "crimson" }}>{err}</div>
        <div style={{ height: 12 }} />
        <button onClick={() => nav("/cases")}>返回</button>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 900 }}>
      <h3>所有病例</h3>

      <div style={{ display: "grid", gap: 10 }}>
        {items.map((c) => (
          <div
            key={c.id}
            className="card"
            style={{ cursor: "pointer" }}
            onClick={() => nav(`/cases/${c.id}`)}
          >
            <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "center" }}>
              <div style={{ fontWeight: 900 }}>
                #{c.id} {c.patientName || "(未填写姓名)"}
              </div>

              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <div className="muted" style={{ fontSize: 12, opacity: 0.85 }}>
                  {c.dept ? `科室：${c.dept}` : ""} {c.status ? `  状态：${c.status}` : ""}
                </div>

                {/* ⭐ NEW：删除按钮，仅 DOCTOR/ADMIN */}
                {canDelete && (
                  <button
                    style={{ padding: "6px 10px" }}
                    onClick={(e) => {
                      e.stopPropagation(); // ⭐ NEW：防止触发行点击跳转
                      onDelete(c.id);
                    }}
                  >
                    删除
                  </button>
                )}
              </div>
            </div>

            <div className="muted" style={{ marginTop: 6 }}>
              性别：{c.patientSex || "-"}　年龄：{c.patientAge ?? "-"}
            </div>

            <div style={{ marginTop: 8 }}>
              <b>基本信息：</b>
              {c.chiefComplaint}
            </div>

            {/* ⭐ NEW：显示病史（history） */}
            <div style={{ marginTop: 6 }}>
            <b>病史：</b>
            {c.history ? c.history : "—"}
             </div>


          </div>
        ))}
      </div>

      {items.length === 0 && (
        <div className="muted" style={{ marginTop: 12 }}>
          暂无病例
        </div>
      )}
    </div>
  );
}
