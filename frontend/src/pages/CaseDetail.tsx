import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  deleteCase, // ✅ 你已实现
  getCase,
  updateCase, // ⭐ NEW
  uploadCaseImage,
} from "../api/cases";
import { getInferenceByJobId, getJob, startInfer } from "../api/infer";
import ReportViewer from "../components/ReportViewer";
import StatusBadge from "../components/StatusBadge";

export default function CaseDetail() {
  const { id } = useParams();
  const caseId = useMemo(() => Number(id), [id]);
  const nav = useNavigate();

  const [caze, setCaze] = useState<any>(null);
  const [file, setFile] = useState<File | null>(null);

  const [jobId, setJobId] = useState<number | null>(null);
  const [status, setStatus] = useState<string>("");
  const [err, setErr] = useState<string>("");
  const [reportText, setReportText] = useState<string>("");

  // ⭐ NEW：角色判断（只影响按钮显示，后端仍会做 403 校验）
  const role = (localStorage.getItem("role") || "").toUpperCase();
  const canEdit = role === "DOCTOR" || role === "ADMIN";
  const canDelete = role === "DOCTOR" || role === "ADMIN";

  // ⭐ NEW：编辑状态 + 表单字段
  const [editing, setEditing] = useState(false);
  const [patientName, setPatientName] = useState("");
  const [patientSex, setPatientSex] = useState("");
  const [patientAge, setPatientAge] = useState<string>(""); // 用 string 避免默认 0
  const [chiefComplaint, setChiefComplaint] = useState("");
  const [history, setHistory] = useState("");

  // 加载病例
  useEffect(() => {
    (async () => {
      try {
        const c = await getCase(caseId);
        setCaze(c);

        // ⭐ NEW：把数据灌入表单（便于编辑）
        setPatientName(c.patientName || "");
        setPatientSex(c.patientSex || "");
        setPatientAge(c.patientAge == null ? "" : String(c.patientAge));
        setChiefComplaint(c.chiefComplaint || "");
        setHistory(c.history || "");
      } catch (e: any) {
        setErr(e?.message || "获取病例失败");
      }
    })();
  }, [caseId]);

  // 上传图片
  async function onUpload() {
    if (!file) return alert("请选择图片");
    setErr("");
    try {
      await uploadCaseImage(caseId, file);
      alert("上传成功");
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "上传失败");
    }
  }

  // 开始推理
  async function onInfer() {
    setErr("");
    setReportText("");
    try {
      const resp = await startInfer(caseId);
      setJobId(resp.jobId);
      setStatus(resp.status);
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "启动推理失败");
    }
  }

  // ⭐ NEW：保存修改（PUT /cases/{id}）
  async function onSaveEdit() {
    setErr("");

    // 最小前端防呆：跟你创建病例那块风格一致
    const sex = patientSex.trim();
    if (sex !== "男" && sex !== "女") return setErr("性别只能是男或女");

    const ageNum = Number(patientAge);
    if (!Number.isInteger(ageNum) || ageNum <= 0) return setErr("年龄必须为正整数");

    if (!chiefComplaint.trim()) return setErr("主诉不能为空");

    try {
      const updated = await updateCase(caseId, {
        patientName: patientName.trim() || undefined,
        patientSex: sex,
        patientAge: ageNum,
        chiefComplaint: chiefComplaint.trim(),
        history: history.trim() || undefined,
      });

      setCaze(updated); // ⭐ NEW：用后端回包刷新视图
      setEditing(false);
      alert("修改成功");
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "修改失败");
    }
  }

  // 删除病例（你已实现，这里只是放进完整文件里）
  async function onDelete() {
    if (!canDelete) return;
    const ok = window.confirm(`确认删除病例 #${caseId} 吗？此操作不可恢复。`);
    if (!ok) return;

    setErr("");
    try {
      await deleteCase(caseId);
      alert("删除成功");
      nav("/cases/all");
    } catch (e: any) {
      setErr(e?.response?.data?.message || e?.message || "删除失败");
    }
  }

  // 轮询 job 状态
  useEffect(() => {
    if (!jobId) return;

    let stop = false;
    const timer = setInterval(async () => {
      if (stop) return;
      try {
        const j = await getJob(jobId);
        setStatus(j.status);

        if (j.status === "SUCCEEDED") {
          stop = true;
          clearInterval(timer);
          const r = await getInferenceByJobId(jobId);
          setReportText(r.reportText || r.rawResult || "");
        }

        if (j.status === "FAILED") {
          stop = true;
          clearInterval(timer);
          setErr(j.lastError || "推理失败");
        }
      } catch (e: any) {
        setErr(e?.message || "轮询失败");
      }
    }, 1200);

    return () => {
      stop = true;
      clearInterval(timer);
    };
  }, [jobId]);

  return (
    <div className="grid">
      {/* 病例信息卡片 */}
      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div>
            <div className="muted" style={{ fontSize: 12 }}>
              病例详情
            </div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>Case #{caseId}</div>
          </div>

          <div className="row">
            <span className="muted">推理状态</span>
            <StatusBadge status={status} />
          </div>
        </div>

        {/* ⭐ NEW：编辑/删除入口（仅 DOCTOR/ADMIN） */}
        <div style={{ marginTop: 10 }} className="row" >
          {canEdit && !editing && <button onClick={() => { setErr(""); setEditing(true); }}>编辑病例</button>}
          {canEdit && editing && (
            <>
              <button onClick={onSaveEdit}>保存修改</button>
              <button onClick={() => { setErr(""); setEditing(false); }}>取消</button>
            </>
          )}
          {canDelete && <button onClick={onDelete}>删除病例</button>}
        </div>

        {/* 查看/编辑区域 */}
        {caze && (
          <div style={{ marginTop: 10 }} className="kv">
            {!editing ? (
              <>
                <b>患者姓名</b>
                <span>{caze.patientName ?? "-"}</span>

                <b>性别/年龄</b>
                <span>
                  {caze.patientSex ?? "?"} / {caze.patientAge ?? "?"}
                </span>

                <b>主诉</b>
                <span>{caze.chiefComplaint}</span>

                <b>病史</b>
                <span>{caze.history ?? "-"}</span>
              </>
            ) : (
              <>
                <b>患者姓名</b>
                <span>
                  <input value={patientName} onChange={(e) => setPatientName(e.target.value)} />
                </span>

                <b>性别（男/女）</b>
                <span>
                  <input value={patientSex} onChange={(e) => setPatientSex(e.target.value)} />
                </span>

                <b>年龄（正整数）</b>
                <span>
                  <input
                    type="number"
                    value={patientAge}
                    onChange={(e) => setPatientAge(e.target.value)}
                    min={1}
                    step={1}
                  />
                </span>

                <b>主诉</b>
                <span>
                  <textarea value={chiefComplaint} onChange={(e) => setChiefComplaint(e.target.value)} />
                </span>

                <b>病史</b>
                <span>
                  <textarea value={history} onChange={(e) => setHistory(e.target.value)} />
                </span>
              </>
            )}
          </div>
        )}
      </div>

      {/* 上传图片 */}
      <div className="card">
        <div className="cardTitle">上传图片</div>
        <div className="row">
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <button onClick={onUpload} disabled={!file}>
            上传
          </button>
        </div>
        <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          要求：png/jpg；建议清晰近景。上传成功后再点击推理。
        </div>
      </div>

      {/* 推理 */}
      <div className="card">
        <div className="cardTitle">推理任务</div>
        <div className="row">
          <button onClick={onInfer}>开始推理</button>
          <div className="muted">jobId: {jobId ?? "-"}</div>
        </div>
      </div>

      {/* 错误 */}
      {err && <div className="error">{err}</div>}

      {/* 报告 */}
      <div className="card">
        {reportText ? <ReportViewer text={reportText} /> : <div className="muted">暂无报告</div>}
      </div>
    </div>
  );
}
