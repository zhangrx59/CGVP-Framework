import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listAllCases, deleteCase } from "../api/cases";
import type { CaseView } from "../api/cases";

export default function ModifyCases() {
  const nav = useNavigate();
  const [items, setItems] = useState<CaseView[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const role = (localStorage.getItem("role") || "").toUpperCase();
  const canEdit = role === "DOCTOR" || role === "ADMIN";
  const canDelete = role === "DOCTOR" || role === "ADMIN";

  useEffect(() => {
    (async () => {
      setLoading(true);
      setErr(null);
      try {
        const data = await listAllCases();
        setItems(data);
      } catch (e: any) {
        setErr(e?.response?.data?.message || e?.message || "加载失败");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  async function onDelete(id: number) {
    if (!canDelete) {
      alert("无权限");
      return;
    }

    const ok = window.confirm(`确认删除病例 #${id} 吗？此操作不可恢复。`);
    if (!ok) return;

    try {
      await deleteCase(id);
      setItems((prev) => prev.filter((x) => x.id !== id));
    } catch (e: any) {
      alert(e?.response?.data?.message || e?.message || "删除失败");
    }
  }

  if (loading) return <div style={{ padding: 24 }}>加载中...</div>;

  if (err) {
    return (
      <div style={{ padding: 24 }}>
        <div style={{ color: "crimson" }}>{err}</div>
        <button onClick={() => nav("/")}>返回首页</button>
      </div>
    );
  }

  return (
    <div style={{ padding: 24 }}>
      <h3>可修改病例列表</h3>

      <div style={{ display: "grid", gap: 10 }}>
        {items.map((c) => (
          <div
            key={c.id}
            className="card"
            style={{ cursor: "pointer" }}
            onClick={() =>
              nav(`/cases/${c.id}`, {
                state: { backTo: "/cases/edit" }, // ✅ NEW
              })
            }
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: 10,
                alignItems: "center",
              }}
            >
              <div style={{ fontWeight: 900 }}>
                #{c.id} {c.patientName || "(未填写姓名)"}
              </div>

              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <div className="muted" style={{ fontSize: 12, opacity: 0.85 }}>
                  {c.dept ? `科室：${c.dept}` : ""}
                  {c.status ? `  状态：${c.status}` : ""}
                </div>

                {/* 修改：所有人都显示，护士点了提示无权限 */}
                <button
                  style={{ padding: "6px 10px" }}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!canEdit) {
                      alert("无权限");
                      return;
                    }
                    nav(`/cases/${c.id}/edit`, {
                      state: { backTo: "/cases/edit" }, // ✅ NEW
                    });
                  }}
                >
                  修改
                </button>

                {/* 删除：仅 DOCTOR/ADMIN 能用（按钮也可选择隐藏，你之前就是隐藏，我这里不强行改） */}
                {canDelete && (
                  <button
                    style={{ padding: "6px 10px" }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(c.id);
                    }}
                  >
                    删除
                  </button>
                )}
              </div>
            </div>

            <div style={{ marginTop: 8, opacity: 0.9 }}>
              <div style={{ fontSize: 13 }}>
                <b>性别：</b>
                {c.patientSex ?? "-"} <b style={{ marginLeft: 12 }}>年龄：</b>
                {c.patientAge ?? "-"}
              </div>

              <div style={{ fontSize: 13, marginTop: 6 }}>
                <b>基本信息：</b>
                {c.chiefComplaint ?? "-"}
              </div>

              <div style={{ fontSize: 13, marginTop: 6 }}>
                <b>病史：</b>
                {c.history ?? "-"}
              </div>
            </div>
          </div>
        ))}
      </div>

      {items.length === 0 && (
        <div className="muted" style={{ marginTop: 12 }}>
          暂无病例
        </div>
      )}

      <div style={{ marginTop: 16 }}>
        <button onClick={() => nav("/")}>返回首页</button>
      </div>
    </div>
  );
}
