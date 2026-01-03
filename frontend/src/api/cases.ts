import { http } from "./http";

export type CaseView = {
  id: number;
  patientName?: string;
  patientSex?: string;
  patientAge?: number;
  chiefComplaint: string;
  history?: string;
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

export async function getCase(id: number) {
  const { data } = await http.get<CaseView>(`/cases/${id}`);
  return data;
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
