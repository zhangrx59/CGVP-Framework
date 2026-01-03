import { useParams } from "react-router-dom";
import { useState } from "react";
import { uploadCaseImage } from "../api/cases";

export default function CaseUpload() {
  const { id } = useParams();
  const caseId = Number(id);

  const [file, setFile] = useState<File | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  async function upload() {
    if (!file) return;
    setMsg(null);

    try {
      await uploadCaseImage(caseId, file);
      setMsg("上传成功");
    } catch (e: any) {
      setMsg(e?.response?.data?.message || e?.message || "上传失败");
    }
  }

  return (
    <div style={{ maxWidth: 520 }}>
      <h3>上传病例图片（护士）</h3>
      <div className="muted" style={{ marginBottom: 10 }}>
        Case ID: {caseId}
      </div>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />
      <div style={{ height: 10 }} />
      <button onClick={upload} disabled={!file}>
        上传图片
      </button>

      {msg && <div style={{ marginTop: 10 }}>{msg}</div>}
    </div>
  );
}
