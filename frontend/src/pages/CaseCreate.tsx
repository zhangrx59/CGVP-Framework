import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createCase, uploadCaseImage } from "../api/cases";
import { MetaForm, defaultMetaState, buildBasicInfoText, buildHistoryText } from "../components/MetaForm"; // ✅ NEW

export default function CaseCreate() {
  const nav = useNavigate();

  const [patientName, setPatientName] = useState("");
  const [gender, setGender] = useState("");
  const [age, setAge] = useState<string>("");

  const [metaForm, setMetaForm] = useState(defaultMetaState); // ✅ NEW

  const [err, setErr] = useState<string | null>(null);

  const [imgFile, setImgFile] = useState<File | null>(null);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const previewUrl = useMemo(() => (imgFile ? URL.createObjectURL(imgFile) : ""), [imgFile]);

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

    // ✅ NEW：把表单自动拼成 chiefComplaint/history
    const chiefComplaint = buildBasicInfoText(metaForm);
    const history = buildHistoryText(metaForm);

    if (!chiefComplaint.trim()) {
      setErr("基本信息不能为空（请至少填写一些基本信息字段）");
      return;
    }

    try {
      const role = (localStorage.getItem("role") || "").toUpperCase();

      const c = await createCase({
        patientName: patientName.trim() || undefined,
        patientSex: g,
        patientAge: Number(a),
        chiefComplaint,
        history: history.trim() ? history : undefined,
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

      if (role === "NURSE" && !imgFile) nav(`/cases/${c.id}/upload`);
      else nav(`/cases/${c.id}`);
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "创建失败");
    }
  }

  return (
    <div style={{ width: "100%", display: "flex", justifyContent: "center" }}>
      <div style={{ maxWidth: 820, width: "100%", padding: 18 }}>
        <h2 style={{ marginTop: 0 }}>创建病例</h2>

        {err && <div style={{ color: "crimson", marginBottom: 12 }}>{err}</div>}

        <div className="card" style={{ padding: 14, marginBottom: 14 }}>
          <div style={{ display: "grid", gap: 10 }}>
            <input
              className="input"
              placeholder="患者姓名：（张三）"
              value={patientName}
              onChange={(e) => setPatientName(e.target.value)}
            />
            <input
              className="input"
              placeholder="性别：（男/女）"
              value={gender}
              onChange={(e) => setGender(e.target.value)}
            />
            <input
              className="input"
              placeholder="年龄："
              value={age}
              onChange={(e) => setAge(e.target.value)}
              inputMode="numeric"
            />
          </div>
        </div>

        {/* ✅ NEW：结构化表单 */}
        <MetaForm value={metaForm} onChange={setMetaForm} />

        <div style={{ height: 14 }} />

        <div className="card" style={{ padding: 14 }}>
          <div style={{ fontWeight: 900, marginBottom: 10 }}>上传病例图片（可选）</div>
          <div style={{ opacity: 0.9, marginBottom: 10 }}>
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
              <div style={{ fontSize: 12, opacity: 0.85, marginBottom: 6 }}>预览：</div>
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

          {uploadMsg && <div style={{ marginTop: 10, opacity: 0.95 }}>{uploadMsg}</div>}
        </div>

        <div style={{ height: 14 }} />

        <button className="btn" style={{ width: "100%" }} onClick={submit} disabled={uploading}>
          {uploading ? "上传中..." : "创建"}
        </button>
      </div>
    </div>
  );
}
