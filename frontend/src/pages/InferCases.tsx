import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listAllCases } from "../api/cases";
import type { CaseView } from "../api/cases";

export default function InferCases() {
  const nav = useNavigate();

  const role = (localStorage.getItem("role") || "").toUpperCase();
  const canInfer = role === "DOCTOR" || role === "ADMIN";

  const [items, setItems] = useState<CaseView[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!canInfer) {
      setErr("无权限：只有医生/管理员可以推理病例");
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

  if (loading) return <div>加载中...</div>;

  if (err) {
    return (
      <div style={{ maxWidth: 900 }}>
        <h3>推理病例</h3>
        <div style={{ color: "crimson" }}>{err}</div>
        <div style={{ height: 12 }} />
        <button onClick={() => nav("/cases")}>返回</button>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 980 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
        <h3 style={{ margin: 0 }}>推理病例</h3>
        <button onClick={() => nav("/cases")}>返回</button>
      </div>

      <div style={{ height: 12 }} />

      <div style={{ display: "grid", gap: 10 }}>
        {items.map((c) => (
          <div
            key={c.id}
            className="card"
            style={{ cursor: "pointer" }}
            onClick={() => nav(`/cases/${c.id}`)} // 点卡片也进详情
          >
            <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "center" }}>
              <div style={{ fontWeight: 900 }}>
                #{c.id} {c.patientName || "(未填写姓名)"}
              </div>

              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <div className="muted" style={{ fontSize: 12, opacity: 0.85 }}>
                  {c.dept ? `科室：${c.dept}` : ""} {c.status ? `  状态：${c.status}` : ""}
                </div>

                {/* ✅ MODIFIED：推理 -> 详情 */}
                <button
                  style={{ padding: "6px 10px" }}
                  onClick={(e) => {
                    e.stopPropagation();
                    nav(`/cases/${c.id}`); // 进入第二张图（病例详情页）
                  }}
                >
                  详情
                </button>
              </div>
            </div>

            <div className="muted" style={{ marginTop: 6 }}>
              性别：{c.patientSex || "-"}　年龄：{c.patientAge ?? "-"}
            </div>

            <div style={{ marginTop: 8 }}>
              <b>基本信息：</b>
              {c.chiefComplaint}
            </div>

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
