// Dependency-free SVG charts with labelled axes and hover tooltips (<title>).
export interface Bar { label: string; value: number; tip?: string }

const W = 620, M = { l: 52, r: 12, t: 12, b: 46 };
const niceMax = (v: number) => (v <= 0 ? 1 : Math.pow(10, Math.floor(Math.log10(v))) * Math.ceil(v / Math.pow(10, Math.floor(Math.log10(v)))));

export function BarChart({ data, xLabel, yLabel, ariaLabel, height = 230 }: { data: Bar[]; xLabel: string; yLabel: string; ariaLabel: string; height?: number }) {
  const max = niceMax(Math.max(0, ...data.map((d) => d.value)));
  const iw = W - M.l - M.r, ih = height - M.t - M.b, bw = data.length ? iw / data.length : iw;
  const ticks = [0, max / 2, max];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${height}`} role="img" aria-label={ariaLabel} width="100%">
      {ticks.map((t) => { const y = M.t + ih - (t / max) * ih; return <g key={t}><line className="axis" x1={M.l} x2={W - M.r} y1={y} y2={y} /><text x={M.l - 6} y={y + 4} textAnchor="end">{Number.isInteger(t) ? t : t.toFixed(1)}</text></g>; })}
      {data.map((d, i) => { const h = (d.value / max) * ih; const x = M.l + i * bw + bw * 0.12; return (
        <g key={d.label + i}><rect className="bar" x={x} y={M.t + ih - h} width={bw * 0.76} height={Math.max(h, 0)} rx={3}><title>{d.tip ?? `${d.label}: ${d.value}`}</title></rect>
          <text x={x + bw * 0.38} y={height - M.b + 14} textAnchor="middle">{d.label.length > 9 ? d.label.slice(0, 8) + "…" : d.label}</text></g>); })}
      <text x={M.l + iw / 2} y={height - 6} textAnchor="middle">{xLabel}</text>
      <text transform={`translate(12 ${M.t + ih / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
    </svg>
  );
}

export interface Series { name: string; color: string; values: number[] }
export function LineChart({ xs, series, xLabel, yLabel, ariaLabel, height = 240, yMax = 1 }: { xs: number[]; series: Series[]; xLabel: string; yLabel: string; ariaLabel: string; height?: number; yMax?: number }) {
  const iw = W - M.l - M.r, ih = height - M.t - M.b;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const px = (x: number) => M.l + (x1 === x0 ? iw / 2 : ((x - x0) / (x1 - x0)) * iw);
  const py = (y: number) => M.t + ih - (Math.max(0, Math.min(y, yMax)) / yMax) * ih;
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${W} ${height}`} role="img" aria-label={ariaLabel} width="100%">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => <g key={f}><line className="axis" x1={M.l} x2={W - M.r} y1={py(f * yMax)} y2={py(f * yMax)} /><text x={M.l - 6} y={py(f * yMax) + 4} textAnchor="end">{(f * yMax).toFixed(2)}</text></g>)}
        {xs.map((x) => <text key={x} x={px(x)} y={height - M.b + 14} textAnchor="middle">{x.toFixed(2)}</text>)}
        {series.map((s) => <g key={s.name}><polyline fill="none" stroke={s.color} strokeWidth={2} points={s.values.map((v, i) => `${px(xs[i] ?? 0)},${py(v)}`).join(" ")} />
          {s.values.map((v, i) => <circle key={i} cx={px(xs[i] ?? 0)} cy={py(v)} r={3} fill={s.color}><title>{`${s.name} at threshold ${(xs[i] ?? 0).toFixed(2)}: ${v.toFixed(4)}`}</title></circle>)}</g>)}
        <text x={M.l + iw / 2} y={height - 6} textAnchor="middle">{xLabel}</text>
        <text transform={`translate(12 ${M.t + ih / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
      </svg>
      <div className="legend">{series.map((s) => <span key={s.name}><i className="dot" style={{ background: s.color }} />{s.name}</span>)}</div>
    </div>
  );
}

// x/y curve (precision-recall, ROC). Optional dotted reference line: horizontal (prevalence) or diagonal (chance).
export function CurveChart({ xs, ys, xLabel, yLabel, ariaLabel, reference, height = 240 }: { xs: number[]; ys: number[]; xLabel: string; yLabel: string; ariaLabel: string; reference?: { kind: "horizontal"; y: number } | { kind: "diagonal" }; height?: number }) {
  const iw = W - M.l - M.r, ih = height - M.t - M.b;
  const px = (x: number) => M.l + Math.max(0, Math.min(x, 1)) * iw;
  const py = (y: number) => M.t + ih - Math.max(0, Math.min(y, 1)) * ih;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${height}`} role="img" aria-label={ariaLabel} width="100%">
      {ticks.map((f) => <g key={f}><line className="axis" x1={M.l} x2={W - M.r} y1={py(f)} y2={py(f)} /><text x={M.l - 6} y={py(f) + 4} textAnchor="end">{f.toFixed(2)}</text><text x={px(f)} y={height - M.b + 14} textAnchor="middle">{f.toFixed(2)}</text></g>)}
      {reference?.kind === "horizontal" ? <line x1={px(0)} x2={px(1)} y1={py(reference.y)} y2={py(reference.y)} stroke="currentColor" strokeDasharray="3 3" opacity={0.5}><title>{`Prevalence (no-skill precision): ${reference.y.toFixed(4)}`}</title></line> : null}
      {reference?.kind === "diagonal" ? <line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} stroke="currentColor" strokeDasharray="3 3" opacity={0.5}><title>Chance level</title></line> : null}
      <polyline fill="none" stroke="#3fb7a6" strokeWidth={2} points={xs.map((x, i) => `${px(x)},${py(ys[i] ?? 0)}`).join(" ")} />
      <text x={M.l + iw / 2} y={height - 6} textAnchor="middle">{xLabel}</text>
      <text transform={`translate(12 ${M.t + ih / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
    </svg>
  );
}

// Horizontal bars with full labels (feature importance).
export function HBarChart({ data, xLabel, ariaLabel }: { data: Bar[]; xLabel: string; ariaLabel: string }) {
  const rowH = 22, labelW = 190, h = data.length * rowH + 40, max = Math.max(1e-12, ...data.map((d) => d.value));
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${h}`} role="img" aria-label={ariaLabel} width="100%">
      {data.map((d, i) => (
        <g key={d.label}><text x={labelW - 6} y={i * rowH + 15} textAnchor="end">{d.label}</text>
          <rect className="bar" x={labelW} y={i * rowH + 4} width={Math.max(1, (d.value / max) * (W - labelW - 20))} height={rowH - 8} rx={3}><title>{d.tip ?? `${d.label}: ${d.value.toFixed(4)}`}</title></rect></g>
      ))}
      <text x={labelW + (W - labelW) / 2} y={h - 6} textAnchor="middle">{xLabel}</text>
    </svg>
  );
}
