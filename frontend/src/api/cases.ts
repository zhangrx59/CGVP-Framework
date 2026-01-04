import { http } from "./http";

// 病例视图（与后端 CaseDtos.CaseView 对齐）
export type CaseView = {
  id: number;
  patientName?: string;
  patientSex?: string;
  patientAge?: number;
  chiefComplaint: string;
  history?: string;

  status?: string;
  createdBy?: number;
  dept?: string;
};

export type CreateCaseReq = {
  patientName?: string;
  patientSex?: string;
  patientAge?: number;
  chiefComplaint: string;
  history?: string;
};

export async function createCase(req: CreateCaseReq) {
  const { data } = await http.post<CaseView>("/cases", req);
  return data;
}

// 获取所有病例（DOCTOR/ADMIN）
export async function listAllCases() {
  const { data } = await http.get<CaseView[]>("/cases");
  return data;
}

export async function getCase(id: number) {
  const { data } = await http.get<CaseView>(`/cases/${id}`);
  return data;
}

// ⭐ NEW：修改病例（DOCTOR/ADMIN）
export type UpdateCaseReq = {
  patientName?: string;
  patientSex?: string;
  patientAge?: number;
  chiefComplaint: string;
  history?: string;
};

// ⭐ NEW：PUT /cases/{id}
export async function updateCase(id: number, req: UpdateCaseReq) {
  const { data } = await http.put<CaseView>(`/cases/${id}`, req);
  return data;
}

// ⭐ 已实现：删除病例（DOCTOR/ADMIN）
export async function deleteCase(id: number) {
  const { data } = await http.delete(`/cases/${id}`);
  return data as { ok: boolean };
}

// 上传图片：后端字段名必须是 file
export async function uploadCaseImage(caseId: number, file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const { data } = await http.post(`/cases/${caseId}/images`, fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}
