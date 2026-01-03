import { http } from "./http";

export type InferStartResp = { jobId: number; status: string };

export type InferenceJob = {
  id: number;
  caseId: number;
  status: string;
  lastError?: string | null;
};

export type InferenceResultView = {
  id: number;
  jobId: number;
  caseId: number;
  predLabel?: string | null;
  reportText?: string | null;   // 如果后端没这个字段，也会用 rawResult 兜底
  rawResult?: string | null;
  createdAt?: string | null;
};

export async function startInfer(caseId: number) {
  const { data } = await http.post<InferStartResp>(`/cases/${caseId}/infer`);
  return data;
}

export async function getJob(jobId: number) {
  const { data } = await http.get<InferenceJob>(`/jobs/${jobId}`);
  return data;
}

export async function getInferenceByJobId(jobId: number) {
  const { data } = await http.get<InferenceResultView>(`/inferences/${jobId}`);
  return data;
}
