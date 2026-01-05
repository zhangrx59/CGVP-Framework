import axios from "axios";

export const http = axios.create({
  baseURL: "/api", // ⭐ 临时写死
//   withCredentials: false,
});

// 你原来应该就有这个
export function setAuthToken(token: string | null) {
  if (token) {
    http.defaults.headers.common["Authorization"] = `Bearer ${token}`;
  } else {
    delete http.defaults.headers.common["Authorization"];
  }
}

// ⭐ NEW：刷新后也恢复 token（避免刷新丢 header）
setAuthToken(localStorage.getItem("token"));

// ⭐ NEW：统一处理 403（权限不足）
http.interceptors.response.use(
  (res) => res,
  (error) => {
    const status = error?.response?.status;

    if (status === 403) {
      // ✅ 所有人都能看到按钮，但点进去如果权限不足就提示被拒绝
      alert("权限不足：该账号无权执行此操作。");
      // 这里不强制跳转，让页面自己决定是否返回
    }

    // 401 通常是 token 失效（可选）
    if (status === 401) {
      alert("登录已失效，请重新登录。");
      localStorage.removeItem("token");
      localStorage.removeItem("role");
      localStorage.removeItem("username");
      // 可选：跳回登录页
      if (location.pathname !== "/login") {
        location.href = "/login";
      }
    }

    return Promise.reject(error);
  }
);

export default http;
