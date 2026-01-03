import { useMemo, useState } from "react";

function pickLine(text: string, prefix: string) {
  const m = text.split("\n").find((l) => l.trim().startsWith(prefix));
  return m ? m.trim().slice(prefix.length).trim() : "";
}

export default function ReportViewer({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const parsed = useMemo(() => {
    const t = (text || "").trim();
    return {
      raw: t,
      probs: pickLine(t, "预测标签："),
      basis: pickLine(t, "诊断依据："),
      advice: pickLine(t, "就诊建议："),
    };
  }, [text]);

  async function copy() {
    await navigator.clipboard.writeText(parsed.raw);
    setCopied(true);
    setTimeout(() => setCopied(false), 900);
  }

  return (
    <div className="grid">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="muted" style={{ fontSize: 12 }}>报告（txt）</div>
        <button onClick={copy}>{copied ? "已复制" : "复制全文"}</button>
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
