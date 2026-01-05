import { useMemo, useState } from "react";

function pickLine(text: string, prefix: string) {
  const m = text.split("\n").find((l) => l.trim().startsWith(prefix));
  return m ? m.trim().slice(prefix.length).trim() : "";
}

export default function ReportViewer({ text }: { text: string }) {
  const [downloading, setDownloading] = useState(false); // ⭐ NEW

  const parsed = useMemo(() => {
    const t = (text || "").trim();
    return {
      raw: t,
      probs: pickLine(t, "预测标签："),
      basis: pickLine(t, "诊断依据："),
      advice: pickLine(t, "就诊建议："),
    };
  }, [text]);

  // ⭐ NEW：下载 txt 文件
  function downloadTxt() {
    if (!parsed.raw) return;

    setDownloading(true);

    try {
      const blob = new Blob([parsed.raw], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = "infer-report.txt"; // ⭐ 可按需要改成带病例 ID
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      URL.revokeObjectURL(url);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="grid">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="muted" style={{ fontSize: 12 }}>
          推理报告（txt）
        </div>

        {/* ✅ MODIFIED：复制全文 → 下载报告 */}
        <button onClick={downloadTxt} disabled={!parsed.raw || downloading}>
          {downloading ? "生成中..." : "下载报告"}
        </button>
      </div>

      <div className="card">
        <div className="cardTitle">预测置信度</div>
        <pre className="report">{parsed.probs || "（暂无）"}</pre>
      </div>

      <div className="card">
        <div className="cardTitle">诊断依据</div>
        <pre className="report">{parsed.basis || "（暂无）"}</pre>
      </div>

      <div className="card">
        <div className="cardTitle">就诊建议</div>
        <pre className="report">{parsed.advice || "（暂无）"}</pre>
      </div>

      <div className="card">
        <div className="cardTitle">原始输出</div>
        <pre className="report">{parsed.raw || "（暂无）"}</pre>
      </div>
    </div>
  );
}
