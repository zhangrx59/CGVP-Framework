import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { getCase, uploadCaseImage } from "../api/cases";
import { getInferenceByJobId, getJob, startInfer } from "../api/infer";
import ReportViewer from "../components/ReportViewer";
import StatusBadge from "../components/StatusBadge";

export default function CaseDetail() {
  const { id } = useParams();
  const caseId = useMemo(() => Number(id), [id]);

  const [caze, setCaze] = useState<any>(null);
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<number | null>(null);
  const [status, setStatus] = useState<string>("");
  const [err, setErr] = useState<string>("");
  const [reportText, setReportText] = useState<string>("");

  useEffect(() => {
    (async () => {
      try {
        const c = await getCase(caseId);
        setCaze(c);
      } catch (e: any) {
        setErr(e?.message || "获取病例失败");
      }
    })();
  }, [caseId]);

  async function onUpload() {
    if (!file) return alert("请选择图片");
    setErr("");
    await uploadCaseImage(caseId, file);
    alert("上传成功");
  }

  async function onInfer() {
    setErr("");
    setReportText("");
    const resp = await startInfer(caseId);
    setJobId(resp.jobId);
    setStatus(resp.status);
  }

  // 轮询 job
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
      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div>
            <div className="muted" style={{ fontSize: 12 }}>病例详情</div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>Case #{caseId}</div>
          </div>
          <div className="row">
            <span className="muted">推理状态</span>
            <StatusBadge status={status} />
          </div>
        </div>

        {caze && (
          <div style={{ marginTop: 10 }} className="kv">
            <b>性别/年龄</b>
            <span>{caze.patientSex ?? "?"} / {caze.patientAge ?? "?"}</span>

            <b>主诉</b>
            <span>{caze.chiefComplaint}</span>
          </div>
        )}
      </div>

      <div className="card">
        <div className="cardTitle">上传图片</div>
        <div className="row">
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <button onClick={onUpload} disabled={!file}>上传</button>
        </div>
        <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          要求：png/jpg；建议清晰近景。上传成功后再点击推理。
        </div>
      </div>

      <div className="card">
        <div className="cardTitle">推理任务</div>
        <div className="row">
          <button onClick={onInfer}>开始推理</button>
          <div className="muted">jobId: {jobId ?? "-"}</div>
        </div>
      </div>

      {err && <div className="error">{err}</div>}

      <div className="card">
        {reportText ? <ReportViewer text={reportText} /> : <div className="muted">暂无报告</div>}
      </div>
    </div>
  );
}
