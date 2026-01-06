// src/pages/EditCaseForm.tsx
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { getCase, updateCase, uploadCaseImage } from "../api/cases";

// ✅ CHANGED：复用创建病例同一套结构化表单（避免两套字段不一致）
import {
  MetaForm,
  defaultMetaState,
  buildBasicInfoText,
  buildHistoryText,
  parseKvText,
  type MetaFormState,
} from "../components/MetaForm";

export default function EditCaseForm() {
  const nav = useNavigate();
  const loc = useLocation();
  const { id } = useParams();
  const caseId = useMemo(() => Number(id), [id]);

  const backTo = (loc.state as any)?.backTo || "/cases/edit";

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [patientName, setPatientName] = useState("");
  const [patientSex, setPatientSex] = useState("");
  const [patientAge, setPatientAge] = useState<string>("");

  // ✅ CHANGED：改为复用 MetaFormState（与创建病例一致）
  const [metaForm, setMetaForm] = useState<MetaFormState>(defaultMetaState);

  // 图片覆盖上传
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!caseId || Number.isNaN(caseId)) {
      setErr("病例 ID 非法");
      setLoading(false);
      return;
    }

    (async () => {
      setLoading(true);
      setErr(null);
      setMsg(null);

      try {
        const c = await getCase(caseId);

        setPatientName(c.patientName || "");
        setPatientSex(c.patientSex || "");
        setPatientAge(c.patientAge != null ? String(c.patientAge) : "");

        // ✅ CHANGED：用 MetaForm.tsx 的 parseKvText 回填（与创建/后端拼接规则一致）
        const basic = parseKvText(c.chiefComplaint);
        const hx = parseKvText(c.history);

        // ✅ CHANGED：把中文 key 映射回 MetaFormState 字段
        setMetaForm((prev) => ({
          ...prev,
          fatherOrigin: basic["父籍贯"] ?? prev.fatherOrigin,
          motherOrigin: basic["母籍贯"] ?? prev.motherOrigin,

          smoke: (basic["是否吸烟"] as any) ?? prev.smoke,
          drink: (basic["是否饮酒"] as any) ?? prev.drink,
          pesticide: (basic["农药"] as any) ?? prev.pesticide,
          tapWater: (basic["生活环境是否有自来水"] as any) ?? prev.tapWater,
          sewer: (basic["生活环境是否有下水道"] as any) ?? prev.sewer,

          phototype: basic["皮肤光型"] ?? prev.phototype,
          region: basic["区域"] ?? prev.region,
          d1: basic["直径1"] ?? prev.d1,
          d2: basic["直径2"] ?? prev.d2,

          itch: (basic["瘙痒"] as any) ?? prev.itch,
          grow: (basic["是否长大"] as any) ?? prev.grow,
          pain: (basic["疼痛"] as any) ?? prev.pain,
          morph: (basic["形态变化"] as any) ?? prev.morph,
          bleed: (basic["出血"] as any) ?? prev.bleed,
          elevate: (basic["是否隆起"] as any) ?? prev.elevate,

          skinCancerHx: (hx["皮肤癌病史"] as any) ?? prev.skinCancerHx,
          cancerHx: (hx["癌症病史"] as any) ?? prev.cancerHx,
        }));
      } catch (e: any) {
        setErr(e?.response?.data?.message || e?.message || "加载失败");
      } finally {
        setLoading(false);
      }
    })();
  }, [caseId]);

  async function onSave() {
    setErr(null);
    setMsg(null);

    const sex = patientSex.trim();
    const age = patientAge.trim();

    if (sex !== "男" && sex !== "女") {
      setErr("性别只能填写：男 或 女");
      return;
    }
    if (!/^[1-9]\d*$/.test(age)) {
      setErr("年龄必须是正整数");
      return;
    }

    // ✅ CHANGED：与创建病例保持一致——用同一套 buildXXXText 拼接
    const chiefComplaint = buildBasicInfoText(metaForm);
    const history = buildHistoryText(metaForm);

    if (!chiefComplaint.trim()) {
      setErr("基本信息不能为空（请至少填写一些基本信息字段）");
      return;
    }

    setSaving(true);
    try {
      await updateCase(caseId, {
        patientName: patientName.trim() ? patientName.trim() : undefined,
        patientSex: sex,
        patientAge: Number(age),
        chiefComplaint,
        history: history.trim() ? history : undefined,
      });

      // ⭐ NEW：写入一次性提示，详情页会显示（CaseDetail.tsx 里已有 flash 读取逻辑）
      sessionStorage.setItem("flash_msg", "病例修改成功");

      // ✅ 保持你原有逻辑：保存成功跳到病例详情页，且带 backTo
      nav(`/cases/${caseId}`, { state: { backTo } });
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "保存失败");
    } finally {
      setSaving(false);
    }
  }

  async function onUploadCover() {
    if (!imgFile) {
      setUploadMsg("请选择要覆盖的图片");
      return;
    }
    setUploadMsg(null);
    setUploading(true);

    try {
      await uploadCaseImage(caseId, imgFile);

      // ⭐ NEW：同样写入一次性提示，详情页可显示
      sessionStorage.setItem("flash_msg", "病例图片更新成功");

      setUploadMsg("图片已覆盖上传成功");
      setImgFile(null);

      // ✅ CHANGED：可选——上传完直接跳详情页（更符合“修改后跳详情”的体验）
      // 如果你希望留在本页不跳转，把下面这行删掉即可
      nav(`/cases/${caseId}`, { state: { backTo } });
    } catch (e: any) {
      setUploadMsg(e?.response?.data?.message || e?.message || "上传失败");
    } finally {
      setUploading(false);
    }
  }

  if (loading) return <div style={{ padding: 24 }}>加载中...</div>;

  if (err) {
    return (
      <div style={{ padding: 24 }}>
        <div style={{ color: "crimson" }}>{err}</div>
        <div style={{ height: 10 }} />
        <button onClick={() => nav(backTo)}>返回列表</button>
      </div>
    );
  }

  return (
    <div style={{ width: "100%", display: "flex", justifyContent: "center" }}>
      <div style={{ maxWidth: 820, width: "100%", padding: 18 }}>
        <h2 style={{ marginTop: 0 }}>修改病例 #{caseId}</h2>

        {msg && <div style={{ color: "limegreen", marginBottom: 12 }}>{msg}</div>}
        {err && <div style={{ color: "crimson", marginBottom: 12 }}>{err}</div>}

        {/* ✅ CHANGED：基本信息卡片风格与创建病例一致 */}
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
              value={patientSex}
              onChange={(e) => setPatientSex(e.target.value)}
            />
            <input
              className="input"
              placeholder="年龄："
              value={patientAge}
              onChange={(e) => setPatientAge(e.target.value)}
              inputMode="numeric"
            />
          </div>
        </div>

        {/* ✅ CHANGED：核心——复用创建病例同一套 MetaForm 组件 */}
        <MetaForm value={metaForm} onChange={setMetaForm} />

        <div style={{ height: 14 }} />

        <div style={{ display: "flex", gap: 10 }}>
          <button className="btn" disabled={saving} onClick={onSave}>
            {saving ? "保存中..." : "保存修改"}
          </button>
          <button onClick={() => nav(backTo)}>返回列表</button>
        </div>

        <div style={{ height: 14 }} />

        {/* ✅ CHANGED：图片覆盖上传保持原功能，但样式与创建页一致 */}
        <div className="card" style={{ padding: 14 }}>
          <div style={{ fontWeight: 900, marginBottom: 10 }}>更新病例图片（覆盖式）</div>
          <div style={{ opacity: 0.9, marginBottom: 10 }}>
            选择一张新图片覆盖当前病例图（后端可能仅保留最新一张）。
          </div>

          <input
            type="file"
            accept="image/*"
            disabled={uploading}
            onChange={(e) => setImgFile(e.target.files?.[0] || null)}
          />

          <div style={{ marginTop: 10, display: "flex", gap: 10, alignItems: "center" }}>
            <button disabled={uploading} onClick={onUploadCover}>
              {uploading ? "上传中..." : "上传并覆盖"}
            </button>
            {uploadMsg && <div style={{ opacity: 0.95 }}>{uploadMsg}</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
