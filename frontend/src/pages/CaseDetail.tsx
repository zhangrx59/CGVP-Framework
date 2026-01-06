// src/pages/CaseDetail.tsx
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";

import {
  getCase,
  listCaseImages,
  fetchCaseImageBlob,
  type CaseView,
  type CaseImageView,
} from "../api/cases";

import {
  startInfer,
  getJob,
  getInferenceByJobId,
  type InferenceJob,
  type InferenceResultView,
  type StartInferResp,
} from "../api/infer";

import StatusBadge from "../components/StatusBadge";
import ReportViewer from "../components/ReportViewer";

export default function CaseDetail() {
  const { id } = useParams();
  const caseId = useMemo(() => Number(id), [id]);
  const nav = useNavigate();
  const location = useLocation();
  const backTo = (location.state as any)?.backTo || "/cases/edit";

  const role = (localStorage.getItem("role") || "").toUpperCase();

  // ✅ CHANGED：对齐后端（你后端 InferController 是 DOCTOR 才能访问 jobs/inferences）
  const canInfer = role === "DOCTOR"; // ✅ CHANGED

  const [flash, setFlash] = useState<string | null>(null);
  useEffect(() => {
    const msg = sessionStorage.getItem("flash_msg");
    if (msg) {
      setFlash(msg);
      sessionStorage.removeItem("flash_msg");
    }
  }, []);

  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [caze, setCaze] = useState<CaseView | null>(null);

  // ===== 图片相关 =====
  const [images, setImages] = useState<CaseImageView[]>([]);
  const [imgUrls, setImgUrls] = useState<Record<number, string>>({});
  const [imgLoading, setImgLoading] = useState(false); // ⭐ NEW
  const [imgLoadErr, setImgLoadErr] = useState<string | null>(null); // ⭐ NEW

  // ===== 推理相关 =====
  const [starting, setStarting] = useState(false);
  const [inferErr, setInferErr] = useState<string | null>(null);

  const [jobId, setJobId] = useState<number | null>(null); // ⭐ NEW：核心：startInfer 返回 jobId
  const [job, setJob] = useState<InferenceJob | null>(null);
  const [result, setResult] = useState<InferenceResultView | null>(null);
  const [resultLoading, setResultLoading] = useState(false);

  // ✅ 释放 blob url
  useEffect(() => {
    return () => {
      Object.values(imgUrls).forEach((u) => URL.revokeObjectURL(u));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 1) 加载病例详情
  useEffect(() => {
    if (!Number.isFinite(caseId)) {
      setErr("路由参数错误：caseId 不是合法数字");
      setLoading(false);
      return;
    }

    (async () => {
      setLoading(true);
      setErr(null);
      try {
        const c = await getCase(caseId);
        setCaze(c);
      } catch (e: any) {
        setErr(e?.response?.data?.message || e?.message || "获取病例失败");
      } finally {
        setLoading(false);
      }
    })();
  }, [caseId]);

  // 2) 加载图片（✅ FIX：fetchCaseImageBlob 必须传 (caseId, imageId)）
  useEffect(() => {
    if (!Number.isFinite(caseId)) return;

    (async () => {
      setImgLoading(true); // ⭐ NEW
      setImgLoadErr(null); // ⭐ NEW

      try {
        const imgs = await listCaseImages(caseId);
        setImages(imgs);

        // ✅ CHANGED：清理旧 blob
        setImgUrls((prev) => {
          Object.values(prev).forEach((u) => URL.revokeObjectURL(u));
          return {};
        });

        const urlMap: Record<number, string> = {};
        for (const img of imgs) {
          try {
            const blob = await fetchCaseImageBlob(caseId, img.id); // ✅ FIX
            urlMap[img.id] = URL.createObjectURL(blob);
          } catch (e) {
            console.warn("[CaseDetail] image blob failed:", { caseId, imageId: img.id }, e); // ⭐ NEW
          }
        }

        setImgUrls(urlMap);
      } catch (e: any) {
        setImgLoadErr(e?.response?.data?.message || e?.message || "图片加载失败"); // ⭐ NEW
      } finally {
        setImgLoading(false); // ⭐ NEW
      }
    })();
  }, [caseId]);

  // 3) ⭐ NEW：轮询 job 状态（QUEUED/RUNNING -> SUCCEEDED/FAILED）
  useEffect(() => {
    if (!jobId) return;

    let stopped = false;
    let timer: any = null;

    const tick = async () => {
      try {
        const j = await getJob(jobId);
        if (stopped) return;
        setJob(j);

        // ✅ CHANGED：结束就停止轮询
        const s = (j.status || "").toUpperCase();
        if (s === "SUCCEEDED" || s === "FAILED") {
          if (timer) clearInterval(timer);
          timer = null;
        }
      } catch {
        // ignore
      }
    };

    // 立即拉一次
    tick();
    timer = setInterval(tick, 1200);

    return () => {
      stopped = true;
      if (timer) clearInterval(timer);
    };
  }, [jobId]);

  // 4) 当 job 成功，拉取结果（✅ FIX：使用 jobId 拉结果，不依赖 startInfer 返回 job）
  useEffect(() => {
    if (!job?.id) return;
    if ((job.status || "").toUpperCase() !== "SUCCEEDED") return;

    (async () => {
      setResultLoading(true);
      try {
        const r = await getInferenceByJobId(job.id);
        setResult(r);
      } catch {
        // ignore
      } finally {
        setResultLoading(false);
      }
    })();
  }, [job?.id, job?.status]);

  async function onStartInfer() {
    if (!caze) return;

    setInferErr(null);
    if (!canInfer) {
      setInferErr("无权限：只有医生可以推理"); // ✅ CHANGED
      return;
    }

    setStarting(true);
    try {
      const resp: StartInferResp = await startInfer((caze as any).id);

      // ⭐ NEW：以 jobId 为主驱动状态机
      setJobId(resp.jobId);

      // ⭐ NEW：先放一个最小 job 让 UI 立刻显示 QUEUED badge（后续轮询会覆盖成真实 job）
      setJob({
        id: resp.jobId,
        caseId: (caze as any).id,
        createdBy: Number(localStorage.getItem("userId") || 0),
        status: resp.status || "QUEUED",
      });

      // ⭐ NEW：清空旧报告，避免残留
      setResult(null);
    } catch (e: any) {
      setInferErr(e?.response?.data?.message || e?.message || "启动推理失败");
    } finally {
      setStarting(false);
    }
  }

  if (loading) {
    return (
      <div style={{ padding: 24 }}>
        <div style={{ fontWeight: 800, fontSize: 18 }}>加载中...</div>
      </div>
    );
  }

  if (err) {
    return (
      <div style={{ padding: 24 }}>
        <h3>查看病例</h3>
        <div style={{ color: "crimson" }}>{err}</div>
        <div style={{ height: 12 }} />
        <button onClick={() => nav(backTo)}>返回</button>
      </div>
    );
  }

  if (!caze) return null;

  const reportText = result?.reportText || result?.rawResult || "";

  return (
    <div style={{ maxWidth: 980 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
        <h3 style={{ margin: 0 }}>查看病例 #{(caze as any).id}</h3>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <button onClick={() => nav(backTo)}>返回列表</button>

          {job?.status && <StatusBadge status={job.status} />}

          {canInfer && (
            <button
              onClick={onStartInfer}
              disabled={starting || (job?.status || "").toUpperCase() === "RUNNING" || (job?.status || "").toUpperCase() === "QUEUED"}
            >
              {starting ? "推理中..." : "推理"}
            </button>
          )}
        </div>
      </div>

      {flash && (
        <div
          style={{
            marginTop: 10,
            padding: "10px 12px",
            borderRadius: 12,
            background: "rgba(0, 255, 120, 0.10)",
            border: "1px solid rgba(0, 255, 120, 0.25)",
            fontWeight: 700,
          }}
        >
          {flash}
        </div>
      )}

      {inferErr && <div style={{ marginTop: 10, color: "crimson", fontWeight: 700 }}>{inferErr}</div>}

      <div style={{ height: 14 }} />

      {/* 病例信息 */}
      <div className="card" style={{ padding: 14 }}>
        <div style={{ display: "grid", gap: 8 }}>
          <div>
            <b>患者：</b>
            {(caze as any).patientName || "-"}
          </div>
          <div>
            <b>性别：</b>
            {(caze as any).patientSex || "-"}　<b>年龄：</b>
            {(caze as any).patientAge ?? "-"}
          </div>
          <div>
            <b>科室：</b>
            {(caze as any).dept || (caze as any).department || "-"}　<b>状态：</b>
            {(caze as any).status || "-"}
          </div>
          <div>
            <b>基本信息：</b>
            {(caze as any).chiefComplaint || "-"}
          </div>
          <div>
            <b>病史：</b>
            {(caze as any).history || "-"}
          </div>
        </div>
      </div>

      <div style={{ height: 14 }} />

      {/* 病例图片 */}
      <div className="card" style={{ padding: 14 }}>
        <div style={{ fontWeight: 800, marginBottom: 8 }}>病例图片</div>

        {imgLoadErr && <div style={{ color: "crimson", marginBottom: 8 }}>{imgLoadErr}</div>}

        {images.length === 0 ? (
          <div style={{ opacity: 0.75 }}>{imgLoading ? "图片加载中..." : "暂无图片"}</div>
        ) : (
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {images.map((img) => {
              const url = imgUrls[img.id];
              return (
                <div key={img.id} style={{ width: 260 }}>
                  {url ? (
                    <img
                      src={url}
                      alt="case"
                      style={{
                        width: "100%",
                        height: 180,
                        objectFit: "contain",
                        borderRadius: 12,
                        border: "1px solid rgba(255,255,255,0.2)",
                      }}
                    />
                  ) : (
                    <div
                      style={{
                        width: "100%",
                        height: 180,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        borderRadius: 12,
                        border: "1px solid rgba(255,255,255,0.2)",
                        opacity: 0.75,
                      }}
                    >
                      {imgLoading ? "图片加载中..." : "图片加载失败"}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div style={{ height: 14 }} />

      {/* 推理报告 */}
      <div className="card" style={{ padding: 14 }}>
        <div style={{ fontWeight: 800, marginBottom: 8 }}>推理报告</div>

        {job?.status && (job.status || "").toUpperCase() === "FAILED" && (
          <div style={{ color: "crimson", fontWeight: 800, marginBottom: 8 }}>推理失败（FAILED）</div>
        )}

        {resultLoading ? (
          <div style={{ opacity: 0.8 }}>加载推理结果中...</div>
        ) : reportText ? (
          <ReportViewer text={reportText} />
        ) : (
          <div style={{ opacity: 0.75 }}>
            {job?.status && (job.status || "").toUpperCase() === "SUCCEEDED"
              ? "推理已完成，但未获取到报告（请检查 /inferences/{jobId}）"
              : "暂无报告"}
          </div>
        )}
      </div>
    </div>
  );
}
