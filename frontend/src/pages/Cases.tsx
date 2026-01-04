import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createCase } from "../api/cases";

export default function Cases() {
  const nav = useNavigate();

  const [patientName, setPatientName] = useState("");
  const [gender, setGender] = useState("");
  // ⭐ MODIFIED：用 string 存输入值，默认空字符串 -> 输入框不显示 0
  const [age, setAge] = useState<string>(""); // ✅ 不再是 0
  const [chiefComplaint, setChiefComplaint] = useState("");
  const [err, setErr] = useState<string | null>(null);

  function validate(): { ok: true; ageNum: number } | { ok: false } {
    const g = gender.trim();
    const cc = chiefComplaint.trim();

    // 性别只能男/女
    if (g !== "男" && g !== "女") {
      setErr("错误：性别只能填写：男 或 女");
      return { ok: false };
    }

    // 主诉不能为空
    if (!cc) {
      setErr("错误：病例信息不能为空");
      return { ok: false };
    }

    // 年龄必须是正整数
    // 允许用户输入空，但提交时必须有值
    if (!age.trim()) {
      setErr("错误：年龄不能为空");
      return { ok: false };
    }

    // 只允许纯数字（防止 1.5 / -3 / e 等）
    if (!/^[1-9]\d*$/.test(age.trim())) {
      setErr("错误：年龄必须是正整数");
      return { ok: false };
    }

    const ageNum = Number(age.trim());
    if (!Number.isSafeInteger(ageNum) || ageNum <= 0) {
      setErr("错误：年龄必须是正整数");
      return { ok: false };
    }

    setErr(null);
    return { ok: true, ageNum };
  }

  async function submit() {
    const v = validate();
    if (!v.ok) return;

    try {
      const role = (localStorage.getItem("role") || "").toUpperCase();

      const c = await createCase({
        patientName: patientName.trim(),
        gender: gender.trim(),
        age: v.ageNum, // ✅ 提交给后端的是 number
        chiefComplaint: chiefComplaint.trim(),
      });

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
    <div style={{ maxWidth: 520 }}>
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
          // ⭐ MODIFIED：value 改成 string，默认空就不会显示 0
          value={age}
          // ⭐ MODIFIED：保持字符串输入；允许用户清空
          onChange={(e) => setAge(e.target.value)}
          // ⭐ NEW：一些输入体验优化（不影响校验）
          min={1}
          step={1}
        />

        <textarea
          placeholder="病例信息（不能为空）"
          value={chiefComplaint}
          onChange={(e) => setChiefComplaint(e.target.value)}
        />

        <button onClick={submit}>创建</button>
      </div>

      {err && <div style={{ color: "crimson", marginTop: 10 }}>{err}</div>}
    </div>
  );
}
