import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createCase, uploadCaseImage } from "../api/cases"; // ⭐ NEW：引入 uploadCaseImage

export default function CaseCreate() {
  const nav = useNavigate();

  const [patientName, setPatientName] = useState("");
  const [gender, setGender] = useState("");
  const [age, setAge] = useState<string>(""); // 保持你之前的写法：字符串
  const [chiefComplaint, setChiefComplaint] = useState("");
  const [err, setErr] = useState<string | null>(null);

  // ⭐ NEW：上传图片相关 state
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  // ⭐ NEW：本地预览 URL（可选但很实用）
  const previewUrl = useMemo(() => {
    if (!imgFile) return "";
    return URL.createObjectURL(imgFile);
  }, [imgFile]);

  // ⭐ NEW：释放预览 URL，避免内存泄露
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function submit() {
    setErr(null);
    setUploadMsg(null);

    // ⭐ 你原有的前端防呆校验（保留）
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

      // 1) 先创建病例
      const c = await createCase({
        patientName: patientName.trim() || undefined,
        gender: g,
        age: Number(a),
        chiefComplaint: chiefComplaint.trim(),
      });

      // ⭐ NEW：2) 如果用户在创建时选了图片，则自动上传（覆盖上传）
      if (imgFile) {
        setUploading(true);
        try {
          await uploadCaseImage(c.id, imgFile);
          setUploadMsg("图片上传成功（已作为该病例最新病例图）");
        } catch (e: any) {
          // 图片上传失败：不影响病例创建成功，但提示用户
          setUploadMsg(e?.response?.data?.message || e?.message || "图片上传失败（病例已创建）");
        } finally {
          setUploading(false);
        }
      }

      // 3) 保持你原来的导航逻辑（但做一个很小的优化）
      // - 护士：如果没选图片，仍去上传页；如果选了图片并已上传，就直接去详情页
      if (role === "NURSE" && !imgFile) {
        nav(`/cases/${c.id}/upload`);
      } else {
        nav(`/cases/${c.id}`);
      }
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "创建失败");
    }
  }

  return (
    // 外层负责居中
    <div
      style={{
        width: "100%",
        display: "flex",
        justifyContent: "center",
      }}
    >
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

          <textarea
            placeholder="基本信息（不能为空）"
            value={chiefComplaint}
            onChange={(e) => setChiefComplaint(e.target.value)}
            rows={3}
          />

          {/* ⭐ NEW：上传病例图片卡片（可选） */}
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
        </div>

        {err && <div style={{ color: "crimson", marginTop: 10 }}>{err}</div>}
      </div>
    </div>
  );
}
