import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Login from "./pages/Login";
import Cases from "./pages/Cases";
import CaseDetail from "./pages/CaseDetail";
import Landing from "./pages/Landing";
import AllCases from "./pages/AllCases";
import Register from "./pages/Register"; // ⭐ NEW
import CaseUpload from "./pages/CaseUpload"; // ⭐ NEW
import CaseCreate from "./pages/CaseCreate";
import EditCases from "./pages/EditCases";
import EditCaseForm from "./pages/EditCaseForm";
import InferCases from "./pages/InferCases";

function RequireAuth({ children }: { children: JSX.Element }) {
  const token = localStorage.getItem("token");
  if (!token) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  return (
    <Layout>
      <Routes>
        {/* ✅ 新增：开场页 */}

        <Route path="/cases/edit" element={<RequireAuth><EditCases /></RequireAuth>} />
        <Route path="/cases/:id/edit" element={<RequireAuth><EditCaseForm /></RequireAuth>} />
        <Route path="/" element={<Landing />} />
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />
        <Route path="/cases/create" element={<CaseCreate />} />
        <Route path="/cases/:id/upload"element={
            <RequireAuth>
              <CaseUpload />
            </RequireAuth>
          }
        />
           <Route path="/cases/all" element={<RequireAuth><AllCases /></RequireAuth>} />
        <Route path="/cases" element={<RequireAuth><Cases /></RequireAuth>} />
        <Route path="/cases/:id" element={<RequireAuth><CaseDetail /></RequireAuth>} />

        {/* ⭐ NEW：推理病例 */}
        <Route
          path="/cases/infer"
          element={
            <RequireAuth>
              <InferCases />
            </RequireAuth>
          }
        />

        {/* 兜底 */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
