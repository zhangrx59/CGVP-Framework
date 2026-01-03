import { http } from "./http";

export type LoginReq = { username: string; password: string };
export type LoginResp = {
  user: { id: number; username: string; role: string; dept?: string };
  token: string;
};

export async function login(req: LoginReq) {
  const { data } = await http.post<LoginResp>("/auth/login", req);
  return data;
}

export async function me() {
  const { data } = await http.get("/me");
  return data as { userId: number; username: string; role: string };
}
