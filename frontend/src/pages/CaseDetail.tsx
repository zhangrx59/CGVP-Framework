import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import {
  getCase,
  listCaseImages,
  fetchCaseImageBlob,
  type CaseView,
  type CaseImageView,
} from "../api/cases";

import { startInfer, getJob, getInferenceByJobId } from "../api/infer";
import type { InferenceJob, InferenceResultView } from "../api/infer";
import StatusBadge from "../components/StatusBadge";
import ReportViewer from "../components/ReportViewer";

export default function CaseDetail() {
  const { id } = useParams();
  const caseId = useMemo(() => Number(id), [id]);
  const nav = useNavigate();

  const role = (localStorage.getItem("role") || "").toUpperCase();
  const canInfer = role === "DOCTOR" || role === "ADMIN";

  const [caze, setCaze] = useState<CaseView | null>(null);
  const [images, setImages] = useState<CaseImageView[]>([]);
  const [imgUrls, setImgUrls] = useState<Record<number, string>>({});

  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // ⭐ NEW：推理相关状态
  const [job, setJob] = useState<InferenceJob | null>(null);
  const [inferErr, setInferErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [result, setResult] = useState<InferenceResultView | null>(null);

  // ✅ 1) 加载病例详情 + 图片列表
  useEffect(() => {
    if (!Number.isFinite(caseId) || caseId <= 0) {
      setErr("病例 ID 非法");
      setLoading(false);
      return;
    }

    (async () => {
      setErr(null);
      setLoading(true);
      try {
        const [c, imgs] = await Promise.all([getCase(caseId), listCaseImages(caseId)]);
        setCaze(c);
        setImages(imgs || []);
      } catch (e: any) {
        setErr(e?.response?.data?.message || e?.message || "加载失败");
      } finally {
        setLoading(false);
      }
    })();
  }, [caseId]);

  // ✅ 2) 图片：blob URL
  useEffect(() => {
    const cleanup = (m: Record<number, string>) => {
      Object.values(m).forEach((u) => {
        try {
          URL.revokeObjectURL(u);
        } catch {
          // ignore
        }
      });
    };

    if (!images || images.length === 0) {
      setImgUrls((prev) => {
        cleanup(prev);
        return {};
      });
      return;
    }

    let alive = true;

    (async () => {
      const next: Record<number, string> = {};
      for (const img of images) {
        try {
          const blob = await fetchCaseImageBlob(caseId, img.id);
          next[img.id] = URL.createObjectURL(blob);
        } catch {
          // 单张失败不影响其它
        }
      }

      if (!alive) {
        cleanup(next);
        return;
      }

      setImgUrls((prev) => {
        cleanup(prev);
        return next;
      });
    })();

    return () => {
      alive = false;
      setImgUrls((prev) => {
        cleanup(prev);
        return {};
      });
    };
  }, [caseId, images]);

  // ⭐ NEW：轮询推理任务
  useEffect(() => {
    if (!job?.id) return;

    let alive = true;

    const tick = async () => {
      try {
        const j = await getJob(job.id);
        if (!alive) return;
        setJob(j);

        if (j.status === "SUCCEEDED") {
          const r = await getInferenceByJobId(j.id);
          if (!alive) return;
          setResult(r);
        }
      } catch (e: any) {
        if (!alive) return;
        setInferErr(e?.response?.data?.message || e?.message || "轮询任务失败");
      }
    };

    tick();

    const t = window.setInterval(() => {
      if (job.status === "RUNNING" || job.status === "QUEUED") tick();
    }, 1200);

    return () => {
      alive = false;
      window.clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id]);

  // ⭐ NEW：点击推理
  async function onStartInfer() {
    if (!canInfer) return;

    setInferErr(null);
    setStarting(true);
    setResult(null);

    try {
      const resp = await startInfer(caseId); // POST /cases/{id}/infer
      setJob({ id: resp.jobId, caseId, status: resp.status });
    } catch (e: any) {
      setInferErr(e?.response?.data?.message || e?.message || "发起推理失败");
    } finally {
      setStarting(false);
    }
  }

  if (loading) return <div>加载中...</div>;

  if (err) {
    return (
      <div style={{ maxWidth: 820 }}>
        <h3>查看病例</h3>
        <div style={{ color: "crimson" }}>{err}</div>
        <div style={{ height: 12 }} />
        <button onClick={() => nav("/cases/all")}>返回</button>
      </div>
    );
  }

  if (!caze) return null;

  const reportText = result?.reportText || result?.rawResult || "";

  return (
    <div style={{ maxWidth: 980 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
        <h3 style={{ margin: 0 }}>查看病例 #{caze.id}</h3>

        {/* ✅ 右上角：返回列表 + 推理按钮（符合你要求） */}
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <button onClick={() => nav("/cases/all")}>返回列表</button>

          {job?.status && <StatusBadge status={job.status} />}

          {canInfer && (
            <button
              onClick={onStartInfer}
              disabled={starting || job?.status === "RUNNING" || job?.status === "QUEUED"}
            >
              {starting ? "推理中..." : "推理"}
            </button>
          )}
        </div>
      </div>

      {inferErr && <div style={{ marginTop: 10, color: "crimson" }}>{inferErr}</div>}
      {job?.lastError && <div style={{ marginTop: 10, color: "crimson" }}>任务错误：{job.lastError}</div>}

      <div style={{ height: 12 }} />

      {/* 病例信息卡片（不变） */}
      <div className="card">
        <div style={{ display: "grid", gap: 6 }}>
          <div>
            <b>患者：</b>
            {caze.patientName || "(未填写姓名)"}
          </div>
          <div>
            <b>性别：</b>
            {caze.patientSex || "-"}　<b>年龄：</b>
            {caze.patientAge ?? "-"}
          </div>
          <div>
            <b>科室：</b>
            {caze.dept || "-"}　<b>状态：</b>
            {caze.status || "-"}
          </div>
          <div>
            <b>基本信息：</b>
            {caze.chiefComplaint}
          </div>
          <div>
            <b>病史：</b>
            {caze.history ?? "-"}
          </div>
        </div>
      </div>

      <div style={{ height: 14 }} />

      {/* 病理图片卡片（不变） */}
      <div className="card">
        <div style={{ fontWeight: 900, marginBottom: 10 }}>病理图片</div>

        {images.length === 0 ? (
          <div className="muted" style={{ fontSize: 13 }}>
            暂无上传图片
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
              gap: 12,
            }}
          >
            {images.map((img) => {
              const src = imgUrls[img.id];
              return (
                <div key={img.id} className="card" style={{ padding: 10 }}>
                  {src ? (
                    <img
                      src={src}
                      alt={img.fileName || `image-${img.id}`}
                      style={{
                        width: "100%",
                        height: 180,
                        objectFit: "cover",
                        borderRadius: 10,
                        display: "block",
                      }}
                    />
                  ) : (
                    <div
                      className="muted"
                      style={{
                        height: 180,
                        borderRadius: 10,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "rgba(255,255,255,0.04)",
                      }}
                    >
                      图片加载中...
                    </div>
                  )}

                  <div style={{ height: 8 }} />
                  <div style={{ fontSize: 12, fontWeight: 800, wordBreak: "break-all" }}>
                    {img.fileName || `image-${img.id}`}
                  </div>
                  <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
                    {img.contentType || ""}{" "}
                    {typeof img.fileSize === "number" ? `· ${Math.round(img.fileSize / 1024)}KB` : ""}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ⭐ NEW：推理成功后展示结果（其它不变） */}
      {job?.status === "SUCCEEDED" && (
        <>
          <div style={{ height: 14 }} />
          <div className="card">
            <div style={{ fontWeight: 900, marginBottom: 10 }}>推理结果</div>
            <ReportViewer text={reportText} />
          </div>
        </>
      )}
    </div>
  );
}
