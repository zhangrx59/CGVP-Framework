import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { getCase, updateCase, uploadCaseImage } from "../api/cases"; // ⭐ NEW：引入 uploadCaseImage

export default function EditCaseForm() {
  const nav = useNavigate();
  const { id } = useParams();

  const caseId = useMemo(() => Number(id), [id]);

  const [err, setErr] = useState<string>("");

  const [patientName, setPatientName] = useState("");
  const [patientSex, setPatientSex] = useState("");
  const [patientAge, setPatientAge] = useState<string>("");
  const [chiefComplaint, setChiefComplaint] = useState("");
  const [history, setHistory] = useState("");

  // ⭐ NEW：上传图片相关状态
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [uploadMsg, setUploadMsg] = useState<string>("");
  const [uploading, setUploading] = useState(false);

  // ⭐ NEW：本地预览（可选，但很有用）
  const previewUrl = useMemo(() => {
    if (!imgFile) return "";
    return URL.createObjectURL(imgFile);
  }, [imgFile]);

  // ⭐ NEW：释放预览 URL，避免内存泄漏
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!Number.isFinite(caseId)) {
      setErr("路由参数错误：caseId 不是合法数字");
      return;
    }

    (async () => {
      try {
        const c: any = await getCase(caseId);

        setPatientName(c?.patientName || "");
        setPatientSex(c?.patientSex || "");
        setPatientAge(c?.patientAge == null ? "" : String(c.patientAge));
        setChiefComplaint(c?.chiefComplaint || "");
        setHistory(c?.history || "");
      } catch (e: any) {
        setErr(e?.response?.data?.message || e?.message || "获取病例失败");
      }
    })();
  }, [caseId]);

  async function onSave() {
    setErr("");

    const sex = patientSex.trim();
    if (sex !== "男" && sex !== "女") return setErr("性别只能是男或女");

    const ageNum = Number(patientAge);
    if (!Number.isInteger(ageNum) || ageNum <= 0) return setErr("年龄必须为正整数");

    if (!chiefComplaint.trim()) return setErr("基本信息不能为空");

    try {
      await updateCase(caseId, {
        patientName: patientName.trim() || undefined,
        patientSex: sex,
        patientAge: ageNum,
        chiefComplaint: chiefComplaint.trim(),
        history: history.trim() || undefined,
      });

      alert("修改成功");
      nav(`/cases/${caseId}`);
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "修改失败");
    }
  }

  // ⭐ NEW：上传图片（覆盖上传：后端会按你的业务逻辑只保留一张）
  async function onUploadImage() {
    if (!Number.isFinite(caseId)) return;
    if (!imgFile) {
      setUploadMsg("请先选择一张图片");
      return;
    }

    setUploadMsg("");
    setUploading(true);
    try {
      await uploadCaseImage(caseId, imgFile);
      setUploadMsg("上传成功（已覆盖为最新病例图）");
      setImgFile(null); // 清空选择
    } catch (e: any) {
      setUploadMsg(e?.response?.data?.message || e?.message || "上传失败");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div style={{ maxWidth: 720 }}>
      <h3>修改病例 #{Number.isFinite(caseId) ? caseId : "-"}</h3>

      {/* 原表单：不动 */}
      <div style={{ display: "grid", gap: 12 }}>
        <input
          value={patientName}
          onChange={(e) => setPatientName(e.target.value)}
          placeholder="患者姓名"
        />

        <input
          value={patientSex}
          onChange={(e) => setPatientSex(e.target.value)}
          placeholder="性别（男/女）"
        />

        <input
          type="number"
          value={patientAge}
          onChange={(e) => setPatientAge(e.target.value)}
          placeholder="年龄（正整数）"
          min={1}
          step={1}
        />

        <textarea
          value={chiefComplaint}
          onChange={(e) => setChiefComplaint(e.target.value)}
          placeholder={
            "基本信息（不能为空，按“key: value；”填写）\n" +
            "例：父籍贯: 河北；母籍贯: 河北；是否吸烟: 否；是否饮酒: 否；农药: 否；生活环境是否有自来水: 是；生活环境是否有下水道: 是；皮肤光型: III；区域: face；直径1: 8；直径2: 6；瘙痒: 否；是否长大: 是；疼痛: 否；形态变化: 是；出血: 否；是否隆起: 是；"
          }
          rows={5}
        />

        <textarea
          value={history}
          onChange={(e) => setHistory(e.target.value)}
          placeholder={"病史（可选，按“key: value；”填写）\n例：皮肤癌病史: 否；癌症病史: 否；"}
          rows={3}
        />


        <button onClick={onSave}>保存修改</button>
      </div>

      {/* ⭐ NEW：上传新图片卡片 */}
      <div style={{ height: 14 }} />
      <div className="card" style={{ padding: 14 }}>
        <div style={{ fontWeight: 900, marginBottom: 10 }}>上传新图片（覆盖病例图）</div>

        <div className="muted" style={{ fontSize: 12, opacity: 0.85, marginBottom: 10 }}>
          选择图片后点击上传，将覆盖该病例的最新病理图片。
        </div>

        <input
          type="file"
          accept="image/*"
          onChange={(e) => setImgFile(e.target.files?.[0] || null)}
          disabled={uploading}
        />

        {/* ⭐ NEW：预览 */}
        {previewUrl && (
          <div style={{ marginTop: 10 }}>
            <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>预览：</div>
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

        <div style={{ display: "flex", gap: 10, marginTop: 12, alignItems: "center" }}>
          <button onClick={onUploadImage} disabled={!imgFile || uploading}>
            {uploading ? "上传中..." : "上传图片"}
          </button>

          {uploadMsg && <div className="muted">{uploadMsg}</div>}
        </div>
      </div>

      {err && <div className="error" style={{ marginTop: 12 }}>{err}</div>}
    </div>
  );
}
