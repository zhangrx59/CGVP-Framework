export default function StatusBadge({ status }: { status: string }) {
  const s = (status || "").toUpperCase();
  let cls = "queued";
  let text = s || "-";

  if (s === "RUNNING") cls = "running";
  else if (s === "SUCCEEDED") cls = "succeeded";
  else if (s === "FAILED") cls = "failed";
  else if (s === "QUEUED") cls = "queued";

  return (
    <span className={`badge ${cls}`}>
      <span className="dot" />
      <span>{text}</span>
    </span>
  );
}
