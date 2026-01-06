import React from "react";

type YesNo = "是" | "否" | "";

export type MetaFormState = {
  fatherOrigin: string; // 父籍贯
  motherOrigin: string; // 母籍贯
  smoke: YesNo; // 是否吸烟
  drink: YesNo; // 是否饮酒
  pesticide: YesNo; // 农药
  tapWater: YesNo; // 生活环境是否有自来水
  sewer: YesNo; // 生活环境是否有下水道
  phototype: string; // 皮肤光型
  region: string; // 区域
  d1: string; // 直径1
  d2: string; // 直径2
  itch: YesNo; // 瘙痒
  grow: YesNo; // 是否长大
  pain: YesNo; // 疼痛
  morph: YesNo; // 形态变化
  bleed: YesNo; // 出血
  elevate: YesNo; // 是否隆起

  skinCancerHx: YesNo; // 皮肤癌病史
  cancerHx: YesNo; // 癌症病史
};

export const defaultMetaState: MetaFormState = {
  fatherOrigin: "",
  motherOrigin: "",
  smoke: "",
  drink: "",
  pesticide: "",
  tapWater: "",
  sewer: "",
  phototype: "",
  region: "",
  d1: "",
  d2: "",
  itch: "",
  grow: "",
  pain: "",
  morph: "",
  bleed: "",
  elevate: "",
  skinCancerHx: "",
  cancerHx: "",
};

function YesNoSelect({
  label,
  value,
  onChange,
}: {
  label: string;
  value: YesNo;
  onChange: (v: YesNo) => void;
}) {
  return (
    <div style={{ display: "grid", gap: 6 }}>
      <div style={{ fontWeight: 700 }}>{label}</div>
      <div style={{ display: "flex", gap: 10 }}>
        <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input
            type="radio"
            name={label}
            checked={value === "是"}
            onChange={() => onChange("是")}
          />
          是
        </label>
        <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input
            type="radio"
            name={label}
            checked={value === "否"}
            onChange={() => onChange("否")}
          />
          否
        </label>
        <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input
            type="radio"
            name={label}
            checked={value === ""}
            onChange={() => onChange("")}
          />
          未填
        </label>
      </div>
    </div>
  );
}

// ✅ 把表单字段拼成后端解析友好的 “key: value；” 文本（基本信息）
export function buildBasicInfoText(m: MetaFormState) {
  const kv: Array<[string, string]> = [
    ["父籍贯", m.fatherOrigin],
    ["母籍贯", m.motherOrigin],
    ["是否吸烟", m.smoke],
    ["是否饮酒", m.drink],
    ["农药", m.pesticide],
    ["生活环境是否有自来水", m.tapWater],
    ["生活环境是否有下水道", m.sewer],
    ["皮肤光型", m.phototype],
    ["区域", m.region],
    ["直径1", m.d1],
    ["直径2", m.d2],
    ["瘙痒", m.itch],
    ["是否长大", m.grow],
    ["疼痛", m.pain],
    ["形态变化", m.morph],
    ["出血", m.bleed],
    ["是否隆起", m.elevate],
  ];

  // 只拼非空，减少噪声；后端 buildStrictMetaJson 会补全缺失字段
  return kv
    .filter(([, v]) => (v ?? "").toString().trim() !== "")
    .map(([k, v]) => `${k}: ${String(v).trim()}；`)
    .join(" ");
}

// ✅ 把表单字段拼成后端解析友好的 “key: value；” 文本（病史）
export function buildHistoryText(m: MetaFormState) {
  const kv: Array<[string, string]> = [
    ["皮肤癌病史", m.skinCancerHx],
    ["癌症病史", m.cancerHx],
  ];

  return kv
    .filter(([, v]) => (v ?? "").toString().trim() !== "")
    .map(([k, v]) => `${k}: ${String(v).trim()}；`)
    .join(" ");
}

// ✅ 从已有 chiefComplaint/history 反解析，给“修改病例”预填
export function parseKvText(text?: string | null): Record<string, string> {
  const out: Record<string, string> = {};
  if (!text) return out;

  const items = text.split(/[；;\r\n]+/).map((s) => s.trim()).filter(Boolean);
  for (const it of items) {
    const idx = it.indexOf("：") >= 0 ? it.indexOf("：") : it.indexOf(":");
    if (idx <= 0) continue;
    const k = it.slice(0, idx).trim();
    const v = it.slice(idx + 1).trim();
    if (k) out[k] = v;
  }
  return out;
}

export function MetaForm({
  value,
  onChange,
}: {
  value: MetaFormState;
  onChange: (next: MetaFormState) => void;
}) {
  const set = (patch: Partial<MetaFormState>) => onChange({ ...value, ...patch });

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div className="card" style={{ padding: 14 }}>
        <div style={{ fontWeight: 900, marginBottom: 10 }}>基本信息（表单）</div>

        <div style={{ display: "grid", gap: 10 }}>
          <input
            className="input"
            placeholder="父籍贯（例如：河北）"
            value={value.fatherOrigin}
            onChange={(e) => set({ fatherOrigin: e.target.value })}
          />
          <input
            className="input"
            placeholder="母籍贯（例如：河北）"
            value={value.motherOrigin}
            onChange={(e) => set({ motherOrigin: e.target.value })}
          />

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <YesNoSelect label="是否吸烟" value={value.smoke} onChange={(v) => set({ smoke: v })} />
            <YesNoSelect label="是否饮酒" value={value.drink} onChange={(v) => set({ drink: v })} />
            <YesNoSelect label="农药" value={value.pesticide} onChange={(v) => set({ pesticide: v })} />
            <YesNoSelect label="生活环境是否有自来水" value={value.tapWater} onChange={(v) => set({ tapWater: v })} />
            <YesNoSelect label="生活环境是否有下水道" value={value.sewer} onChange={(v) => set({ sewer: v })} />
            <YesNoSelect label="瘙痒" value={value.itch} onChange={(v) => set({ itch: v })} />
            <YesNoSelect label="是否长大" value={value.grow} onChange={(v) => set({ grow: v })} />
            <YesNoSelect label="疼痛" value={value.pain} onChange={(v) => set({ pain: v })} />
            <YesNoSelect label="形态变化" value={value.morph} onChange={(v) => set({ morph: v })} />
            <YesNoSelect label="出血" value={value.bleed} onChange={(v) => set({ bleed: v })} />
            <YesNoSelect label="是否隆起" value={value.elevate} onChange={(v) => set({ elevate: v })} />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <input
              className="input"
              placeholder="皮肤光型（例如：III）"
              value={value.phototype}
              onChange={(e) => set({ phototype: e.target.value })}
            />
            <input
              className="input"
              placeholder="区域（例如：face）"
              value={value.region}
              onChange={(e) => set({ region: e.target.value })}
            />
            <input
              className="input"
              placeholder="直径1（mm，例如：8）"
              value={value.d1}
              onChange={(e) => set({ d1: e.target.value })}
              inputMode="numeric"
            />
            <input
              className="input"
              placeholder="直径2（mm，例如：6）"
              value={value.d2}
              onChange={(e) => set({ d2: e.target.value })}
              inputMode="numeric"
            />
          </div>
        </div>
      </div>

      <div className="card" style={{ padding: 14 }}>
        <div style={{ fontWeight: 900, marginBottom: 10 }}>病史（表单）</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <YesNoSelect label="皮肤癌病史" value={value.skinCancerHx} onChange={(v) => set({ skinCancerHx: v })} />
          <YesNoSelect label="癌症病史" value={value.cancerHx} onChange={(v) => set({ cancerHx: v })} />
        </div>
      </div>
    </div>
  );
}
