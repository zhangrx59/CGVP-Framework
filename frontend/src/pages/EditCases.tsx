import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listAllCases } from "../api/cases";
import { deleteCase } from "../api/cases";
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
    if (role !== "DOCTOR" && role !== "ADMIN") {
      setErr("无权限：只有医生/管理员可以修改病例");
      setLoading(false);
      return;
    }

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

  async function onDelete(id: number) {
    const ok = window.confirm(`确认删除病例 #${id} 吗？此操作不可恢复。`);
    if (!ok) return;

    try {
      await deleteCase(id);
      setItems((prev) => prev.filter((x) => x.id !== id));
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "删除失败");
    }
  }

  if (loading) return <div>加载中.</div>;

  if (err) {
    return (
      <div style={{ maxWidth: 720 }}>
        <h3>修改病例</h3>
        <div style={{ color: "crimson" }}>{err}</div>
        <div style={{ height: 12 }} />
        <button onClick={() => nav("/cases")}>返回</button>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 900 }}>
      <h3>修改病例</h3>

      <div style={{ display: "grid", gap: 10 }}>
        {items.map((c) => (
          <div
            key={c.id}
            className="card"
            style={{ cursor: "pointer" }}
            onClick={() => nav(`/cases/${c.id}`)} // 保持“查看详情”行为不变
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
                  {c.dept ? `科室：${c.dept}` : ""}{" "}
                  {c.status ? `  状态：${c.status}` : ""}
                </div>

                {/* ⭐ NEW：修改按钮（最小化增加） */}
                {canEdit && (
                  <button
                    style={{ padding: "6px 10px" }}
                    onClick={(e) => {
                      e.stopPropagation();
                      nav(`/cases/${c.id}/edit`);
                    }}
                  >
                    修改
                  </button>
                )}

                {/* 删除按钮保持一致 */}
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

            <div className="muted" style={{ marginTop: 6 }}>
              性别：{c.patientSex || "-"}　年龄：{c.patientAge ?? "-"}
            </div>

            <div style={{ marginTop: 8 }}>
              <b>基本信息：</b>
              {c.chiefComplaint}
            </div>

            {/* 你之前提到“病史要显示”，这里也顺手保持一致（如果不想要可删这一段） */}
            <div style={{ marginTop: 6 }}>
              <b>病史：</b>
              {c.history ?? "-"}
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
