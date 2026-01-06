import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";

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

  // ✅ NEW：支持从 /cases/edit 返回；默认回 /cases/edit（满足你现在的需求）
  const location = useLocation();
  const backTo = (location.state as any)?.backTo || "/cases/edit";

  // ✅ NEW：跨页面提示（修改成功/图片更新成功）
  const [flash, setFlash] = useState<string | null>(null);
  useEffect(() => {
    const msg = sessionStorage.getItem("flash_msg");
    if (msg) {
      setFlash(msg);
      sessionStorage.removeItem("flash_msg");
    }
  }, []);

  const role = (localStorage.getItem("role") || "").toUpperCase();
  const canInfer = role === "DOCTOR" || role === "ADMIN";

  const [caze, setCaze] = useState<CaseView | null>(null);
  const [images, setImages] = useState<CaseImageView[]>([]);
  const [imgUrls, setImgUrls] = useState<Record<number, string>>({});

  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // 推理相关
  const [job, setJob] = useState<InferenceJob | null>(null);
  const [inferErr, setInferErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

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

  // 2) 加载病例图片列表，并拉取 blob 作为预览（保留你原实现）
  useEffect(() => {
    if (!Number.isFinite(caseId)) return;

    (async () => {
      try {
        const imgs = await listCaseImages(caseId);
        setImages(imgs);

        // 拉取每张图片 blob 并生成本地 url
        const urlMap: Record<number, string> = {};
        for (const img of imgs) {
          try {
            const blob = await fetchCaseImageBlob(img.id);
            urlMap[img.id] = URL.createObjectURL(blob);
          } catch {
            // 单张图片失败不影响页面
          }
        }
        setImgUrls((prev) => {
          // 先释放旧的
          Object.values(prev).forEach((u) => URL.revokeObjectURL(u));
          return urlMap;
        });
      } catch {
        // 图片加载失败不打断主流程
      }
    })();
  }, [caseId]);

  // 3) 如果病例已有推理任务/结果：这里按你原逻辑轮询/获取（保留）
  useEffect(() => {
    if (!caze?.latestJobId) return;

    (async () => {
      try {
        const j = await getJob(caze.latestJobId);
        setJob(j);
      } catch {
        // ignore
      }
    })();
  }, [caze?.latestJobId]);

  // 4) 当 job 完成，拉取结果（保留）
  useEffect(() => {
    if (!job?.id) return;
    if (job.status !== "SUCCEEDED") return;

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
      setInferErr("无权限：只有医生/管理员可以推理");
      return;
    }

    setStarting(true);
    try {
      const j = await startInfer(caze.id);
      setJob(j);
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
        <div style={{ opacity: 0.8, marginTop: 8 }}>正在获取病例信息</div>
      </div>
    );
  }

  if (err) {
    return (
      <div style={{ padding: 24 }}>
        <h3>查看病例</h3>
        <div style={{ color: "crimson" }}>{err}</div>
        <div style={{ height: 12 }} />
        {/* ✅ CHANGED：回 edit 列表 */}
        <button onClick={() => nav(backTo)}>返回</button>
      </div>
    );
  }

  if (!caze) return null;

  const reportText = result?.reportText || result?.rawResult || "";

  return (
    <div style={{ maxWidth: 980 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
        <h3 style={{ margin: 0 }}>查看病例 #{caze.id}</h3>

        {/* 右上角：返回列表 + 推理按钮 */}
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {/* ✅ CHANGED：回 edit 列表 */}
          <button onClick={() => nav(backTo)}>返回列表</button>

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

      {/* ✅ NEW：修改成功/图片更新成功提示（只显示一次） */}
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

      <div style={{ height: 14 }} />

      {/* 病例信息（保留你原字段展示写法） */}
      <div className="card" style={{ padding: 14 }}>
        <div style={{ display: "grid", gap: 8 }}>
          <div>
            <b>患者：</b>
            {caze.patientName || "-"}
          </div>
          <div>
            <b>性别：</b>
            {caze.patientSex || "-"}　<b>年龄：</b>
            {caze.patientAge ?? "-"}
          </div>
          <div>
            <b>科室：</b>
            {caze.department || "-"}　<b>状态：</b>
            {caze.status || "-"}
          </div>
          <div>
            <b>基本信息：</b>
            {caze.chiefComplaint || "-"}
          </div>
          <div>
            <b>病史：</b>
            {caze.history || "-"}
          </div>
        </div>
      </div>

      {/* ✅ 病例图片：保留你原逻辑（listCaseImages + blob url） */}
      <div style={{ height: 14 }} />
      <div className="card" style={{ padding: 14 }}>
        <div style={{ fontWeight: 800, marginBottom: 8 }}>病例图片</div>

        {images.length === 0 ? (
          <div style={{ opacity: 0.75 }}>暂无图片</div>
        ) : (
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {images.map((img) => {
              const url = imgUrls[img.id] || img.url; // 有 blob 用 blob；否则退化到后端 url
              return (
                <div key={img.id} style={{ width: 260 }}>
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
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* 推理错误 */}
      {inferErr && (
        <div style={{ marginTop: 12, color: "crimson" }}>
          {inferErr}
        </div>
      )}

      {/* 推理结果展示（保留你原组件） */}
      <div style={{ height: 14 }} />
      <div className="card" style={{ padding: 14 }}>
        <div style={{ fontWeight: 800, marginBottom: 8 }}>推理报告</div>
        {resultLoading ? (
          <div style={{ opacity: 0.8 }}>加载推理结果中...</div>
        ) : reportText ? (
          <ReportViewer text={reportText} />
        ) : (
          <div style={{ opacity: 0.75 }}>暂无报告</div>
        )}
      </div>
    </div>
  );
}
