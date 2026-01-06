// src/api/infer.ts
// ✅ FIX: 对齐后端：POST /cases/{caseId}/infer 只返回 { jobId, status }
// ✅ FIX: 提供 getJob(jobId) 轮询用

import http from "./http";

export type StartInferResp = {
  jobId: number;
  status: "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | string;
};

export type InferenceJob = {
  id: number;
  caseId: number;
  createdBy: number;
  status: "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | string;
  attemptCount?: number;
  lastError?: string | null;
  createdAt?: string;
  startedAt?: string | null;
  finishedAt?: string | null;
};

export type InferenceResultView = {
  id?: number;
  jobId?: number;
  caseId?: number;
  rawResult?: string;     // 有些后端会存原始JSON
  reportText?: string;    // 有些后端会直接存文本
  // 其余字段可选
  [k: string]: any;
};

export async function startInfer(caseId: number): Promise<StartInferResp> {
  const { data } = await http.post(`/cases/${caseId}/infer`);
  return data;
}

export async function getJob(jobId: number): Promise<InferenceJob> {
  const { data } = await http.get(`/jobs/${jobId}`);
  return data;
}

export async function getInferenceByJobId(jobId: number): Promise<InferenceResultView> {
  const { data } = await http.get(`/inferences/${jobId}`);
  return data;
}
