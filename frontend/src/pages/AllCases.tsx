import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listAllCases } from "../api/cases";
import type { CaseView } from "../api/cases";

export default function AllCases() {
  const nav = useNavigate();
  const [items, setItems] = useState<CaseView[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const role = (localStorage.getItem("role") || "").toUpperCase();
    if (role !== "DOCTOR" && role !== "ADMIN") {
      setErr("无权限：只有医生/管理员可以查看所有病例");
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
  }, []);

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
      <h3>所有病例（医生/管理员）</h3>

      <div style={{ display: "grid", gap: 10 }}>
        {items.map((c) => (
          <div
            key={c.id}
            className="card"
            style={{ cursor: "pointer" }}
            onClick={() => nav(`/cases/${c.id}`)}
          >
            <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
              <div style={{ fontWeight: 900 }}>
                #{c.id} {c.patientName || "(未填写姓名)"}
              </div>
              <div className="muted" style={{ fontSize: 12, opacity: 0.85 }}>
                {c.dept ? `科室：${c.dept}` : ""} {c.status ? `  状态：${c.status}` : ""}
              </div>
            </div>

            <div className="muted" style={{ marginTop: 6 }}>
              性别：{c.patientSex || "-"}　年龄：{c.patientAge ?? "-"}
            </div>

            <div style={{ marginTop: 8 }}>
              <b>主诉：</b>{c.chiefComplaint}
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
