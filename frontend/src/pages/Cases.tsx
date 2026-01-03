import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createCase } from "../api/cases";

export default function Cases() {
  const nav = useNavigate();
  const [chiefComplaint, setChiefComplaint] = useState("43-year-old male with a skin lesion...");
  const [history, setHistory] = useState("");
  const [patientSex, setPatientSex] = useState("M");
  const [patientAge, setPatientAge] = useState<number>(43);
  const [err, setErr] = useState<string>("");

  async function onCreate() {
    setErr("");
    try {
      const c = await createCase({ chiefComplaint, history, patientSex, patientAge });
      nav(`/cases/${c.id}`);
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "创建病例失败");
    }
  }

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <h3>新建病例</h3>
      <input value={patientSex} onChange={(e) => setPatientSex(e.target.value)} placeholder="patientSex (M/F/U)" />
      <input value={patientAge} type="number" onChange={(e) => setPatientAge(Number(e.target.value))} />
      <textarea rows={3} value={chiefComplaint} onChange={(e) => setChiefComplaint(e.target.value)} />
      <textarea rows={2} value={history} onChange={(e) => setHistory(e.target.value)} />
      <button onClick={onCreate}>创建并进入详情</button>
      {err && <div style={{ color: "crimson" }}>{err}</div>}
    </div>
  );
}
