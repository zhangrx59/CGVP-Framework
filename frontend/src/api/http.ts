import axios from "axios";

export const http = axios.create({
  baseURL: "/api",
  timeout: 120000,
});

export function setAuthToken(token: string | null) {
  if (token) http.defaults.headers.common["Authorization"] = `Bearer ${token}`;
  else delete http.defaults.headers.common["Authorization"];
}

const saved = localStorage.getItem("token");
if (saved) setAuthToken(saved);
