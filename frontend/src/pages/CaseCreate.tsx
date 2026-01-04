import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createCase } from "../api/cases";

export default function CaseCreate() {
  const nav = useNavigate();

  const [patientName, setPatientName] = useState("");
  const [gender, setGender] = useState("");
  const [age, setAge] = useState<string>(""); // ⭐ MODIFIED：不默认 0，改成字符串空值
  const [chiefComplaint, setChiefComplaint] = useState("");
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setErr(null);

    // ⭐ NEW：前端防呆校验
    const g = gender.trim();
    const a = age.trim();

    if (g !== "男" && g !== "女") {
      setErr("性别只能填写：男 或 女");
      return;
    }

    if (!/^[1-9]\d*$/.test(a)) {
      setErr("年龄必须是正整数");
      return;
    }

    if (!chiefComplaint.trim()) {
      setErr("基本信息（主诉）不能为空");
      return;
    }

    try {
      const role = (localStorage.getItem("role") || "").toUpperCase();

      const c = await createCase({
        patientName: patientName.trim() || undefined,
        gender: g,
        age: Number(a),
        chiefComplaint: chiefComplaint.trim(),
      });

      // ⭐ 保持你原来的逻辑：护士去上传页；医生/管理员去详情页
      if (role === "NURSE") {
        nav(`/cases/${c.id}/upload`);
      } else {
        nav(`/cases/${c.id}`);
      }
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "创建失败");
    }
  }

return (
  // ⭐ NEW：外层只负责“居中定位”
  <div
    style={{
      width: "100%",
      display: "flex",
      justifyContent: "center", // ⭐ 水平居中
    }}
  >
    {/* ⭐ 原来的表单容器，完全不动 */}
    <div style={{ maxWidth: 520, width: "100%" }}>
      <h3>创建病例</h3>

      <div style={{ display: "grid", gap: 8 }}>
        <input
          placeholder="患者姓名:（张三）"
          value={patientName}
          onChange={(e) => setPatientName(e.target.value)}
        />
        <input
          placeholder="性别:（男/女）"
          value={gender}
          onChange={(e) => setGender(e.target.value)}
        />
        <input
          placeholder="年龄："
          type="number"
          value={age}
          onChange={(e) => setAge(e.target.value)}
        />
        <textarea
          placeholder="基本信息（不能为空）"
          value={chiefComplaint}
          onChange={(e) => setChiefComplaint(e.target.value)}
        />
        <button onClick={submit}>创建</button>
      </div>

      {err && <div style={{ color: "crimson", marginTop: 10 }}>{err}</div>}
    </div>
  </div>
);

}
