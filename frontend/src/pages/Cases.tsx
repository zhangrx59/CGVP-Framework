import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createCase } from "../api/cases";

export default function Cases() {
  const nav = useNavigate();

  const [patientName, setPatientName] = useState("");
  const [gender, setGender] = useState("");
  const [age, setAge] = useState<number>(0);
  const [chiefComplaint, setChiefComplaint] = useState("");
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setErr(null);

    try {
      const role = localStorage.getItem("role") || ""; // ⭐ NEW

      const c = await createCase({
        patientName,
        gender,
        age,
        chiefComplaint,
      });

      // ⭐ MODIFIED：护士创建后跳上传页；医生/管理员跳详情页
      if (role.toUpperCase() === "NURSE") {
        nav(`/cases/${c.id}/upload`);
      } else {
        nav(`/cases/${c.id}`);
      }
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "创建失败");
    }
  }

  return (
    <div style={{ maxWidth: 520 }}>
      <h3>创建病例</h3>

      <div style={{ display: "grid", gap: 8 }}>
        <input
          placeholder="患者姓名"
          value={patientName}
          onChange={(e) => setPatientName(e.target.value)}
        />
        <input
          placeholder="性别"
          value={gender}
          onChange={(e) => setGender(e.target.value)}
        />
        <input
          placeholder="年龄"
          type="number"
          value={age}
          onChange={(e) => setAge(Number(e.target.value))}
        />
        <textarea
          placeholder="主诉"
          value={chiefComplaint}
          onChange={(e) => setChiefComplaint(e.target.value)}
        />
        <button onClick={submit}>创建</button>
      </div>

      {err && <div style={{ color: "crimson", marginTop: 10 }}>{err}</div>}
    </div>
  );
}
