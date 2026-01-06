import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createCase, uploadCaseImage } from "../api/cases";

export default function CaseCreate() {
  const nav = useNavigate();

  const [patientName, setPatientName] = useState("");
  const [gender, setGender] = useState("");
  const [age, setAge] = useState<string>("");

  const [chiefComplaint, setChiefComplaint] = useState(""); // 基本信息
  const [history, setHistory] = useState(""); // ✅ NEW：病史

  const [err, setErr] = useState<string | null>(null);

  const [imgFile, setImgFile] = useState<File | null>(null);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const previewUrl = useMemo(() => {
    if (!imgFile) return "";
    return URL.createObjectURL(imgFile);
  }, [imgFile]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function submit() {
    setErr(null);
    setUploadMsg(null);

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
      setErr("基本信息不能为空");
      return;
    }

    try {
      const role = (localStorage.getItem("role") || "").toUpperCase();

      // ✅ MODIFIED：把 history 一并提交给后端（代替 meta.json 文件）
      const c = await createCase({
        patientName: patientName.trim() || undefined,
        gender: g,
        age: Number(a),
        chiefComplaint: chiefComplaint.trim(),
        history: history.trim() || undefined, // ✅ NEW
      });

      if (imgFile) {
        setUploading(true);
        try {
          await uploadCaseImage(c.id, imgFile);
          setUploadMsg("图片上传成功（已作为该病例最新病例图）");
        } catch (e: any) {
          setUploadMsg(e?.response?.data?.message || e?.message || "图片上传失败（病例已创建）");
        } finally {
          setUploading(false);
        }
      }

      if (role === "NURSE" && !imgFile) {
        nav(`/cases/${c.id}/upload`);
      } else {
        nav(`/cases/${c.id}`);
      }
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "创建失败");
    }
  }

  // ✅ NEW：给护士一个“可复制样例”
  const basicInfoHint =
    "父籍贯: 河北；母籍贯: 河北；是否吸烟: 否；是否饮酒: 否；农药: 否；生活环境是否有自来水: 是；生活环境是否有下水道: 是；皮肤光型: III；区域: face；直径1: 8；直径2: 6；瘙痒: 否；是否长大: 是；疼痛: 否；形态变化: 是；出血: 否；是否隆起: 是；";
  const historyHint = "皮肤癌病史: 否；癌症病史: 否；";

  return (
    <div style={{ width: "100%", display: "flex", justifyContent: "center" }}>
      <div style={{ maxWidth: 520, width: "100%" }}>
        <h3>创建病例</h3>

        <div style={{ display: "grid", gap: 10 }}>
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
            value={age}
            onChange={(e) => setAge(e.target.value)}
            inputMode="numeric"
          />

          {/* ✅ MODIFIED：明确说明这是“基本信息 kv”，用于代替 meta.json */}
          <textarea
            placeholder={`基本信息（不能为空，按“key: value；”填写）\n例如：${basicInfoHint}`}
            value={chiefComplaint}
            onChange={(e) => setChiefComplaint(e.target.value)}
            rows={5}
          />

          {/* ✅ NEW：病史输入框 */}
          <textarea
            placeholder={`病史（可选，按“key: value；”填写）\n例如：${historyHint}`}
            value={history}
            onChange={(e) => setHistory(e.target.value)}
            rows={3}
          />

          {/* 上传图片卡片（保持不动） */}
          <div className="card" style={{ padding: 14 }}>
            <div style={{ fontWeight: 900, marginBottom: 10 }}>上传病例图片（可选）</div>

            <div className="muted" style={{ fontSize: 12, opacity: 0.85, marginBottom: 10 }}>
              选择一张病理/皮肤图片作为该病例的病例图（后端采用覆盖上传，仅保留一张）。
            </div>

            <input
              type="file"
              accept="image/*"
              disabled={uploading}
              onChange={(e) => setImgFile(e.target.files?.[0] || null)}
            />

            {previewUrl && (
              <div style={{ marginTop: 10 }}>
                <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
                  预览：
                </div>
                <img
                  src={previewUrl}
                  alt="preview"
                  style={{
                    width: "100%",
                    maxHeight: 260,
                    objectFit: "contain",
                    borderRadius: 12,
                    border: "1px solid rgba(255,255,255,0.18)",
                  }}
                />
              </div>
            )}

            {uploadMsg && <div className="muted" style={{ marginTop: 10 }}>{uploadMsg}</div>}
          </div>

          <button onClick={submit} disabled={uploading}>
            {uploading ? "上传中..." : "创建"}
          </button>

          {err && <div className="error">{err}</div>}
        </div>
      </div>
    </div>
  );
}
