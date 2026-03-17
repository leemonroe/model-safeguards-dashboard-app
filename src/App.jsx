import { useState, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine, Area,
  AreaChart, ComposedChart
} from "recharts";

// ── Utility ─────────────────────────────────────────────────────────────────

const fmt = (n) => {
  if (n >= 1e12) return `$${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  if (n >= 1) return `$${n.toFixed(0)}`;
  if (n >= 0.01) return `$${n.toFixed(2)}`;
  return `<$0.01`;
};

const fmtX = (n) => {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M×`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K×`;
  if (n >= 100) return `${n.toFixed(0)}×`;
  if (n >= 10) return `${n.toFixed(1)}×`;
  return `${n.toFixed(2)}×`;
};

const fmtParams = (n) => {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return `${n.toFixed(0)}`;
};

const logScale = (min, max, f) => Math.pow(10, Math.log10(min) + f * (Math.log10(max) - Math.log10(min)));
const logFrac = (min, max, v) => (Math.log10(Math.max(v, min)) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));

// ── Model Core ──────────────────────────────────────────────────────────────

// Baseline fine-tuning steps (unprotected model). Fixed internal constant.
const BASELINE_STEPS = 100;

// Reference model size for compute cost slider. computePer1K is "at this size",
// then scales O(N) with nThreshold.
const REFERENCE_PARAMS = 70e9;

// Derive internal parameters from user-facing ones
function deriveInternal(p) {
  // Hardware: user provides rate (×/yr) and years-to-plateau
  // We model as asymptotic: hw_factor(t) = 1 + (headroom-1) * (1 - e^(-t/tau))
  // tau = yearsToPlateauHw / 3 (so ~95% captured by plateau year)
  // headroom = hwRate ^ yearsToPlateauHw (total improvement if rate held constant)
  const tau = p.yearsToPlateauHw / 3;
  const headroom = Math.pow(p.hwRate, p.yearsToPlateauHw);

  // TR alpha: user provides "10× larger model is X× harder to break"
  // R(N) = base * (N/1B)^alpha => R(10N)/R(N) = 10^alpha = trScaling10x
  // => alpha = log10(trScaling10x);
  const alpha = Math.log10(p.trScaling10x);

  // Compute cost scales O(N) with model size relative to 70B reference
  const computeScale = p.nThreshold / REFERENCE_PARAMS;
  const scaledComputePer1K = p.computePer1K * computeScale;

  // Derive costFtBase and trBase from new user-facing params:
  //   setupCost = fixed overhead (data, expertise) — TR doesn't affect
  //   scaledComputePer1K = GPU cost per 1K steps at actual model size
  //   stepsToCircumvent = steps needed to break the safeguard
  const costFtBase = p.setupCost + scaledComputePer1K * BASELINE_STEPS / 1000;
  const trBase = p.stepsToCircumvent / BASELINE_STEPS;

  return { tau, headroom, alpha, costFtBase, trBase, computeScale, scaledComputePer1K };
}

function computeModel(p) {
  const { tau, headroom, alpha, costFtBase, trBase, scaledComputePer1K } = deriveInternal(p);
  const data = [];
  const YEARS = 25;

  for (let t = 0; t <= YEARS; t++) {
    // ── Hardware: asymptotic ──
    const hwFactor = t === 0 ? 1 : 1 + (headroom - 1) * (1 - Math.exp(-t / tau));

    // ── Algorithmic: decelerating exponential (floor = 1.0) ──
    let cumAlgo = 1;
    for (let i = 1; i <= t; i++) {
      const decay = Math.pow(0.5, i / p.algoHalflife);
      const rate = 1 + (p.algoRate - 1) * decay;
      cumAlgo *= rate;
    }

    // ── Combined training cost ──
    const cumTrainReduction = hwFactor * cumAlgo;
    const costTrain = p.costTrainBase / cumTrainReduction;
    const costTrainHwOnly = p.costTrainBase / hwFactor;

    // ── Fine-tuning: benefits from same hw + algo (simplified) ──
    // Only the compute portion declines; setup cost stays fixed
    // Compute cost scales O(N) with model size
    const cumFtReduction = hwFactor * cumAlgo;
    const computeBase = scaledComputePer1K * BASELINE_STEPS / 1000;
    const costFtRaw = p.setupCost + computeBase / cumFtReduction;

    // ── Attack improvement: decelerates on same timeline as algo ──
    let cumAttack = 1;
    for (let i = 1; i <= t; i++) {
      const decay = Math.pow(0.5, i / p.algoHalflife);
      const rate = 1 + (p.attackRate - 1) * decay;
      cumAttack *= rate;
    }

    // ── Tamper resistance ──
    // Effective steps = user steps × model size scaling
    const sizeScaling = Math.pow(p.nThreshold / 1e9, alpha);
    const effectiveSteps = p.stepsToCircumvent * sizeScaling;
    const trMultiplier = effectiveSteps / BASELINE_STEPS;

    // ── Cost to remove safeguard ──
    // Compute scales with steps and O(N) with model size; setup is fixed
    const computeRemove = (scaledComputePer1K * effectiveSteps / 1000) / cumFtReduction;
    const costRemove = (p.setupCost + computeRemove) / cumAttack;

    // ── Derived ──
    const costAttack = Math.min(costRemove, costTrain);
    const maxUsefulTR = costTrain / costFtRaw * cumAttack;

    // Effective annual rates for decomposition
    let effHwRate = 1;
    if (t > 0) {
      const hwPrev = 1 + (headroom - 1) * (1 - Math.exp(-(t - 1) / tau));
      effHwRate = hwFactor / hwPrev;
    }
    const algoDecay = Math.pow(0.5, Math.max(t, 1) / p.algoHalflife);
    const effAlgoRate = 1 + (p.algoRate - 1) * algoDecay;

    data.push({
      year: 2026 + t, t,
      costTrain, costTrainHwOnly, costRemove, costFtRaw, costAttack,
      hwFactor, cumAlgo, cumTrainReduction,
      effHwRate, effAlgoRate, effCombined: effHwRate * effAlgoRate,
      trMultiplier, maxUsefulTR,
    });
  }
  return data;
}

function findRelevanceYear(data, budget) {
  for (let i = 0; i < data.length; i++) {
    if (data[i].costAttack <= budget) return i;
  }
  return data.length;
}

// ── Monte Carlo Global Sensitivity ──────────────────────────────────────────

const PARAM_RANGES = [
  { key: "costTrainBase", label: "Training cost today", lo: 1e7, hi: 5e10, log: true,
    src: "GPT-4 ~$100M; frontier 2025 ~$500M-$2B; smaller capable models much less" },
  { key: "setupCost", label: "Attack setup cost", lo: 2e3, hi: 2e4, log: true,
    src: "Data sourcing, expertise, infrastructure setup. Range $2K-$20K reflects realistic attack preparation costs. Slider allows wider exploration." },
  { key: "computePer1K", label: "Compute per 1K steps (at 70B)", lo: 20, hi: 500, log: true,
    src: "Assumes full fine-tuning ($20-$500/1K at 70B). Conservative: excludes QLoRA ($0.50-$5) because we want safeguard-optimistic estimates. If QLoRA suffices to break TR, real costs are much lower." },
  { key: "hwRate", label: "Hardware improvement rate", lo: 1.1, hi: 1.55, log: false,
    src: "Centered on Epoch AI estimate: GPU FLOP/$ doubles every ~2.5yr (1.32×/yr). Range reflects uncertainty around this estimate, not speculative extremes." },
  { key: "yearsToPlateauHw", label: "Years until hardware plateaus", lo: 5, hi: 15, log: false,
    src: "Centered on mainstream estimates. Below 5yr requires near-immediate physical limits; above 15yr requires novel substrate breakthroughs." },
  { key: "algoRate", label: "Algorithmic efficiency rate", lo: 1.5, hi: 4.5, log: false,
    src: "Centered on Epoch AI controlled estimate (~3×/yr). Lower end allows for deceleration from current pace; upper end for continued rapid discovery." },
  { key: "algoHalflife", label: "Years of fast algo progress", lo: 3, hi: 20, log: false,
    src: "Catch-up from transformers may stall fast (3yr). AI-driven R&D could sustain 15-20yr." },
  { key: "attackRate", label: "Attack method improvement", lo: 1.0, hi: 2.5, log: false,
    src: "Poorly measured — no Epoch-quality estimates exist. Capped at 2.5× for safeguard-optimistic analysis. Real rate is highly uncertain." },
  { key: "stepsToCircumvent", label: "Steps to circumvent safeguard", lo: 100, hi: 1e8, log: true,
    src: "No known upper bound on achievable TR. Upper bound set where removal cost approaches training cost (the natural ceiling). Samples above this enter training-limited regime automatically." },
  { key: "trScaling10x", label: "TR scaling (10× larger model)", lo: 1.0, hi: 5.0, log: false,
    src: "Almost no empirical data. Most uncertain parameter in the entire model." },
  { key: "nThreshold", label: "Capability threshold", lo: 1e9, hi: 5e11, log: true,
    src: "Capability-dependent. Some risks at 7-70B; others may require 100B+." },
];

function sampleUniform(lo, hi, log) {
  if (log) return Math.exp(Math.log(lo) + Math.random() * (Math.log(hi) - Math.log(lo)));
  return lo + Math.random() * (hi - lo);
}

function sampleParams(overrides) {
  const s = {};
  PARAM_RANGES.forEach(r => {
    const o = overrides?.[r.key];
    s[r.key] = o ? sampleUniform(o[0], o[1], r.log) : sampleUniform(r.lo, r.hi, r.log);
  });
  return s;
}

function fastRelevanceYear(p, budget) {
  const tau = p.yearsToPlateauHw / 3;
  const headroom = Math.pow(p.hwRate, p.yearsToPlateauHw);
  const alpha = Math.log10(Math.max(p.trScaling10x, 1.001));
  const sizeScaling = Math.pow(p.nThreshold / 1e9, alpha);
  const effectiveSteps = p.stepsToCircumvent * sizeScaling;
  const computeScale = p.nThreshold / REFERENCE_PARAMS;
  const scaledCPK = p.computePer1K * computeScale;
  let cumAlgo = 1, cumAttack = 1;
  for (let t = 0; t <= 25; t++) {
    const hwF = t === 0 ? 1 : 1 + (headroom - 1) * (1 - Math.exp(-t / tau));
    if (t > 0) {
      const ad = Math.pow(0.5, t / p.algoHalflife);
      cumAlgo *= 1 + (p.algoRate - 1) * ad;
      cumAttack *= 1 + (p.attackRate - 1) * ad;
    }
    const costTrain = p.costTrainBase / (hwF * cumAlgo);
    const computeRemove = (scaledCPK * effectiveSteps / 1000) / (hwF * cumAlgo);
    const costRemove = (p.setupCost + computeRemove) / cumAttack;
    if (Math.min(costRemove, costTrain) <= budget) return t;
  }
  return 26;
}

function runMonteCarlo(budget, n, overrides) {
  const results = [];
  for (let i = 0; i < n; i++) {
    const s = sampleParams(overrides);
    results.push({ p: s, yr: fastRelevanceYear(s, budget) });
  }
  return results;
}

function analyzeResults(results) {
  const yrs = results.map(r => r.yr).sort((a, b) => a - b);
  const n = yrs.length;
  const mean = yrs.reduce((a, b) => a + b, 0) / n;
  const totalVar = yrs.reduce((a, b) => a + (b - mean) ** 2, 0) / n;

  const pctl = (f) => yrs[Math.min(Math.floor(f * n), n - 1)];
  const percentiles = { p5: pctl(0.05), p10: pctl(0.1), p25: pctl(0.25), p50: pctl(0.5), p75: pctl(0.75), p90: pctl(0.9), p95: pctl(0.95) };

  // Histogram
  const hist = [];
  for (let y = 0; y <= 26; y++) {
    hist.push({ year: y >= 26 ? ">25" : `${y}`, count: yrs.filter(v => v === y).length, pct: +(yrs.filter(v => v === y).length / n * 100).toFixed(1) });
  }

  // Variance decomposition (split-halves approximation of first-order Sobol)
  const decomp = PARAM_RANGES.map(r => {
    const sorted = [...results].sort((a, b) => a.p[r.key] - b.p[r.key]);
    const half = Math.floor(n / 2);
    const loMean = sorted.slice(0, half).reduce((a, s) => a + s.yr, 0) / half;
    const hiMean = sorted.slice(half).reduce((a, s) => a + s.yr, 0) / (n - half);
    const varCE = ((loMean - mean) ** 2 + (hiMean - mean) ** 2) / 2;
    return {
      ...r, importance: totalVar > 0 ? varCE / totalVar : 0,
      loMean: +loMean.toFixed(1), hiMean: +hiMean.toFixed(1),
      importancePct: totalVar > 0 ? +(varCE / totalVar * 100).toFixed(1) : 0,
    };
  }).sort((a, b) => b.importance - a.importance);

  return { mean: +mean.toFixed(1), percentiles, hist, decomp, totalVar, n };
}

// ── Theme ───────────────────────────────────────────────────────────────────

const C = {
  bg: "#08090d", bgSide: "#0c0e14", bgCard: "#10131b", bgInner: "#0a0c12",
  border: "#1a1e2e", borderLt: "#252a3a",
  text: "#d8dce8", muted: "#8892a8", dim: "#5a6278",
  train: "#e87f3a", remove: "#3abfe8", attack: "#b07ae8",
  green: "#3ae88a", red: "#e85a5a", yellow: "#e8cf3a",
  hw: "#e87f3a", algo: "#3abfe8", combined: "#b07ae8",
};

const ACTORS = [
  { label: "Individual ($10K)", budget: 1e4 },
  { label: "Small Org ($1M)", budget: 1e6 },
  { label: "Well-funded Org ($100M)", budget: 1e8 },
  { label: "Nation State ($10B)", budget: 1e10 },
  { label: "Custom", budget: null },
];

// ── Components ──────────────────────────────────────────────────────────────

function Slider({ label, value, onChange, min, max, step, format, log, tip, color }) {
  const [showTip, setShowTip] = useState(false);
  const display = format ? format(value) : value;
  const sv = log ? logFrac(min, max, value) * 1000 : ((value - min) / (max - min)) * 1000;
  const handle = (e) => {
    const r = Number(e.target.value) / 1000;
    if (log) onChange(logScale(min, max, r));
    else {
      const v = min + r * (max - min);
      onChange(step ? Math.round(v / step) * step : v);
    }
  };
  const ac = color || C.dim;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
        <span style={{ color: C.muted, fontSize: 11, fontFamily: "var(--f)", position: "relative" }}>
          {label}
          {tip && (
            <span onMouseEnter={() => setShowTip(true)} onMouseLeave={() => setShowTip(false)}
              onClick={() => setShowTip(s => !s)}
              style={{ marginLeft: 4, cursor: "help", color: C.dim, fontSize: 10, userSelect: "none" }}>
              ⓘ
              {showTip && (
                <span style={{
                  position: "absolute", left: 0, top: "calc(100% + 6px)", zIndex: 100,
                  background: "#1a2030", border: `1px solid ${C.borderLt}`, borderRadius: 5,
                  padding: "8px 10px", fontSize: 10, lineHeight: 1.5, color: C.muted,
                  width: 230, boxShadow: "0 6px 24px rgba(0,0,0,0.5)", fontWeight: 400,
                  pointerEvents: "none",
                }}>{tip}</span>
              )}
            </span>
          )}
        </span>
        <span style={{
          color: C.text, fontSize: 12, fontFamily: "var(--f)", fontWeight: 600,
          background: ac + "18", padding: "1px 6px", borderRadius: 3,
        }}>{display}</span>
      </div>
      <input type="range" min={0} max={1000} value={sv} onChange={handle} style={{
        width: "100%", height: 3, appearance: "none",
        background: `linear-gradient(to right, ${ac}44, ${ac})`,
        borderRadius: 2, outline: "none", cursor: "pointer", accentColor: ac,
      }} />
    </div>
  );
}

function GroupLabel({ children, color }) {
  return (
    <div style={{
      fontSize: 10, color: color || C.dim, marginBottom: 6, marginTop: 18,
      fontWeight: 700, fontFamily: "var(--f)", textTransform: "uppercase",
      letterSpacing: "0.1em", display: "flex", alignItems: "center", gap: 6,
    }}>
      <span style={{ width: 8, height: 2, background: color, display: "inline-block", borderRadius: 1 }} />
      {children}
    </div>
  );
}

function Section({ children, sub, mt }) {
  return (
    <div style={{ marginBottom: 14, marginTop: mt ?? 28 }}>
      <h2 style={{
        fontSize: 14, fontWeight: 600, color: C.text, fontFamily: "var(--f)",
        letterSpacing: "0.02em", margin: 0, paddingBottom: 6,
        borderBottom: `1px solid ${C.border}`,
      }}>{children}</h2>
      {sub && <p style={{ fontSize: 11, color: C.dim, margin: "5px 0 0", fontFamily: "var(--f)", lineHeight: 1.5 }}>{sub}</p>}
    </div>
  );
}

function Tip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#161a28ee", border: `1px solid ${C.border}`, borderRadius: 5,
      padding: "8px 12px", fontFamily: "var(--f)", fontSize: 10,
      boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
    }}>
      <div style={{ color: C.muted, marginBottom: 5, fontSize: 11 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 1 }}>
          {p.name}: {typeof p.value === 'number' && p.value > 100 ? fmt(p.value) : typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
        </div>
      ))}
    </div>
  );
}

function Chart({ children, h }) {
  return (
    <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: "16px 12px 8px" }}>
      <ResponsiveContainer width="100%" height={h || 380}>{children}</ResponsiveContainer>
    </div>
  );
}

function Metric({ label, value, sub, color }) {
  return (
    <div style={{
      background: C.bgCard, border: `1px solid ${C.border}`,
      borderLeft: `3px solid ${color || C.dim}`,
      borderRadius: 6, padding: "14px 16px", flex: 1, minWidth: 150,
    }}>
      <div style={{ fontSize: 10, color: C.dim, fontFamily: "var(--f)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 5 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: color || C.text, fontFamily: "var(--f)", lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: C.muted, marginTop: 4, fontFamily: "var(--f)" }}>{sub}</div>}
    </div>
  );
}

const tickStyle = { fontSize: 10, fontFamily: "var(--f)" };

// ── Main ────────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState("model");

  // Core parameters
  const [costTrainBase, setCostTrainBase] = useState(5e8);
  const [setupCost, setSetupCost] = useState(5e3);
  const [computePer1K, setComputePer1K] = useState(5);
  const [hwRate, setHwRate] = useState(1.4);
  const [yearsToPlateauHw, setYearsToPlateauHw] = useState(10);
  const [algoRate, setAlgoRate] = useState(2.5);
  const [algoHalflife, setAlgoHalflife] = useState(8);
  const [attackRate, setAttackRate] = useState(2.0);
  const [stepsToCircumvent, setStepsToCircumvent] = useState(10000);
  const [trScaling10x, setTrScaling10x] = useState(2.0);
  const [nThreshold, setNThreshold] = useState(7e10);

  // Actor selection
  const [actorIdx, setActorIdx] = useState(1);
  const [customBudget, setCustomBudget] = useState(1e6);
  const budget = ACTORS[actorIdx].budget ?? customBudget;

  const allBudgets = useMemo(() => ACTORS.filter(a => a.budget !== null).map(a => a), []);

  const p = useMemo(() => ({
    costTrainBase, setupCost, computePer1K, hwRate, yearsToPlateauHw,
    algoRate, algoHalflife, attackRate,
    stepsToCircumvent, trScaling10x, nThreshold,
  }), [costTrainBase, setupCost, computePer1K, hwRate, yearsToPlateauHw, algoRate, algoHalflife, attackRate, stepsToCircumvent, trScaling10x, nThreshold]);

  const data = useMemo(() => computeModel(p), [p]);
  const derived = useMemo(() => deriveInternal(p), [p]);
  const { headroom, costFtBase, trBase, scaledComputePer1K } = derived;
  const hwFloor = costTrainBase / headroom;
  const costToCircumventToday = setupCost + scaledComputePer1K * stepsToCircumvent / 1000;

  // Relevance windows for all preset actors
  const windows = useMemo(() => allBudgets.map(a => ({
    ...a,
    years: findRelevanceYear(data, a.budget),
  })), [data, allBudgets]);

  const selectedWindow = useMemo(() => findRelevanceYear(data, budget), [data, budget]);

  // Monte Carlo — run on budget change (memoized, seeded fresh each time)
  const [showAdvRanges, setShowAdvRanges] = useState(false);
  const [rangeOverrides, setRangeOverrides] = useState({});
  const [heatMode, setHeatMode] = useState("exact");

  const TR_MULTS = [2, 3, 5, 10, 30, 100, 300, 1000];
  const HEAT_MAX_YR = 10;
  const HEAT_N = 3000;

  // TR Investment heatmap MC
  const trInvestData = useMemo(() => {
    const rng = (lo, hi, log) => log
      ? Math.exp(Math.log(lo) + Math.random() * (Math.log(hi) - Math.log(lo)))
      : lo + Math.random() * (hi - lo);
    const results = {};
    TR_MULTS.forEach(x => { results[x] = []; });
    const baseSamples = [];
    for (let i = 0; i < HEAT_N; i++) {
      const s = {};
      PARAM_RANGES.forEach(r => { s[r.key] = rng(r.lo, r.hi, r.log); });
      baseSamples.push(s);
    }
    for (let i = 0; i < HEAT_N; i++) {
      const s = baseSamples[i];
      const baseYr = fastRelevanceYear(s, budget);
      TR_MULTS.forEach(x => {
        const s2 = { ...s, stepsToCircumvent: s.stepsToCircumvent * x };
        const impYr = fastRelevanceYear(s2, budget);
        results[x].push(Math.max(0, Math.min(impYr - baseYr, HEAT_MAX_YR + 1)));
      });
    }
    return results;
  }, [budget]);

  const heatRows = useMemo(() => TR_MULTS.map(x => {
    const gains = trInvestData[x] || [];
    const n = gains.length || 1;
    const sorted = [...gains].sort((a, b) => a - b);
    const row = { tr: `${x}×`, trNum: x };
    for (let y = 0; y <= HEAT_MAX_YR; y++) {
      const label = y === HEAT_MAX_YR ? `${y}+` : `${y}`;
      row[label] = +((gains.filter(g => y === HEAT_MAX_YR ? g >= y : g === y).length / n) * 100).toFixed(1);
      row[`gte_${y}`] = +((gains.filter(g => g >= y).length / n) * 100).toFixed(1);
    }
    row.mean = +(gains.reduce((a, b) => a + b, 0) / n).toFixed(1);
    row.median = sorted[Math.floor(n / 2)] || 0;
    return row;
  }), [trInvestData]);
  const mcResults = useMemo(() => {
    const samples = runMonteCarlo(budget, 5000, Object.keys(rangeOverrides).length ? rangeOverrides : undefined);
    return analyzeResults(samples);
  }, [budget, rangeOverrides]);

  // Also compute for all preset actors
  const mcByActor = useMemo(() => {
    return ACTORS.filter(a => a.budget !== null).map(a => {
      const samples = runMonteCarlo(a.budget, 2000);
      const analysis = analyzeResults(samples);
      return { ...a, ...analysis };
    });
  }, [rangeOverrides]);

  const tabs = [
    { id: "model", label: "Cost Curves" },
    { id: "decomp", label: "Decomposition" },
    { id: "relevance", label: "Relevance" },
    { id: "requiredTR", label: "Required TR" },
    { id: "trInvest", label: "TR Investment" },
    { id: "sensitivity", label: "Sensitivity" },
    { id: "details", label: "Details" },
    { id: "sources", label: "Sources" },
  ];

  return (
    <div style={{ "--f": "'IBM Plex Mono', monospace", background: C.bg, color: C.text, minHeight: "100vh", fontFamily: "var(--f)" }}>

      {/* Header */}
      <div style={{ borderBottom: `1px solid ${C.border}`, padding: "16px 24px", background: `linear-gradient(180deg, ${C.bgCard} 0%, ${C.bg} 100%)` }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
          <h1 style={{ fontSize: 18, fontWeight: 700, margin: 0, letterSpacing: "-0.02em" }}>
            Safeguard Relevance Model
          </h1>
          <span style={{ fontSize: 10, color: C.dim }}>
            Open-weight · Hardware asymptote + algorithmic trends · 11 parameters
          </span>
        </div>
        <div style={{ display: "flex", gap: 6, marginTop: 12 }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              background: tab === t.id ? C.train + "1a" : "transparent",
              border: `1px solid ${tab === t.id ? C.train + "66" : C.border}`,
              color: tab === t.id ? C.train : C.muted,
              borderRadius: 5, padding: "5px 12px", fontSize: 11,
              fontFamily: "var(--f)", cursor: "pointer",
            }}>{t.label}</button>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", minHeight: "calc(100vh - 90px)" }}>

        {/* Sidebar */}
        <div style={{
          width: 296, minWidth: 296, borderRight: `1px solid ${C.border}`,
          padding: "16px 16px", overflowY: "auto", background: C.bgSide,
          maxHeight: "calc(100vh - 90px)",
        }}>
          <GroupLabel color={C.train}>Cost Baselines</GroupLabel>
          <Slider label="Training cost today" value={costTrainBase} onChange={setCostTrainBase}
            min={1e6} max={1e11} log format={fmt} color={C.train}
            tip="Current dollar cost to train a model at the capability threshold from scratch. GPT-4 scale ≈ $100M; frontier 2025 ≈ $500M-$1B." />
          <Slider label="Attack setup cost (data, expertise)" value={setupCost} onChange={setSetupCost}
            min={100} max={1e6} log format={fmt} color={C.train}
            tip="Fixed overhead for an attack: sourcing training data, infrastructure setup, expertise. Not affected by tamper resistance — the attacker pays this regardless." />
          <Slider label="Compute cost per 1K steps (at 70B)" value={computePer1K} onChange={setComputePer1K}
            min={0.1} max={1e3} log format={fmt} color={C.train}
            tip="GPU cost per 1K fine-tuning steps at 70B params. Scales O(N) with model size. Default $5 reflects QLoRA — the cheapest realistic attack: ~$0.50-$1 for QLoRA, ~$5-$50 for LoRA, ~$50-$500 for full fine-tune. We default to QLoRA-range because a rational attacker uses the cheapest method that works." />

          <GroupLabel color={C.hw}>Hardware (Has a Ceiling)</GroupLabel>
          <Slider label="Improvement rate (×/yr)" value={hwRate} onChange={setHwRate}
            min={1.05} max={2.5} step={0.05} format={v => `${v.toFixed(2)}×`} color={C.hw}
            tip="Current annual hardware price-performance gain. Epoch AI data: GPU FLOP/$ doubles every ~2.5yr ≈ 1.32×/yr. Including specialization (lower precision, tensor cores): ~1.4×/yr." />
          <Slider label="Years until plateau" value={yearsToPlateauHw} onChange={setYearsToPlateauHw}
            min={3} max={25} step={1} format={v => `${v}yr`} color={C.hw}
            tip="When hardware gains become negligible. Driven by transistor scaling limits (~2nm), memory bandwidth walls, and fab economics. Most estimates: practical limits reached in 8-15 years." />

          <GroupLabel color={C.algo}>Algorithms (No Known Ceiling)</GroupLabel>
          <Slider label="Efficiency rate (×/yr)" value={algoRate} onChange={setAlgoRate}
            min={1} max={8} step={0.1} format={v => `${v.toFixed(1)}×`} color={C.algo}
            tip="Current annual algorithmic efficiency gain. Controlled estimates: ~3×/yr (Epoch, 2025). Catch-up estimates range from 2× to 60×/yr with massive uncertainty." />
          <Slider label="Years of fast progress" value={algoHalflife} onChange={setAlgoHalflife}
            min={2} max={25} step={1} format={v => `${v}yr`} color={C.algo}
            tip="How long algorithmic improvement stays near its current pace before noticeably slowing. After this many years, the rate has dropped to roughly half its starting value. Longer = sustained progress." />

          <GroupLabel color={C.remove}>Attacks & Tamper Resistance</GroupLabel>
          <Slider label="Attack improvement (×/yr)" value={attackRate} onChange={setAttackRate}
            min={1} max={5} step={0.1} format={v => `${v.toFixed(1)}×`} color={C.remove}
            tip="Annual improvement in adversarial fine-tuning methods for removing safeguards (LoRA optimization, loss design, etc). Decelerates on the same timeline as algorithmic progress." />
          <Slider label="Steps to circumvent safeguard" value={stepsToCircumvent} onChange={setStepsToCircumvent}
            min={10} max={1e7} log format={v => v >= 1e6 ? `${(v/1e6).toFixed(1)}M` : v >= 1000 ? `${(v/1000).toFixed(0)}K` : `${v.toFixed(0)}`} color={C.green}
            tip="Fine-tuning steps needed to break the safeguard. Reference points: No safeguard ~100 steps. Post-training safety tuning ~50-200. TAR (Tamirisa et al.) ~1K-5K. Deep Ignorance (EleutherAI) ~10K. This is what TR papers actually measure." />
          {/* Live cost readout */}
          <div style={{
            background: C.green + "12", border: `1px solid ${C.green}33`,
            borderRadius: 5, padding: "8px 10px", marginBottom: 8,
          }}>
            <div style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 3 }}>
              Cost to circumvent today
            </div>
            <div style={{ fontSize: 16, fontWeight: 700, color: C.green, fontFamily: "var(--f)" }}>
              {fmt(costToCircumventToday)}
            </div>
            <div style={{ fontSize: 9, color: C.muted, marginTop: 2 }}>
              {fmt(setupCost)} setup + {fmt(scaledComputePer1K * stepsToCircumvent / 1000)} compute ({stepsToCircumvent.toLocaleString()} steps × {fmt(scaledComputePer1K)}/1K at {fmtParams(nThreshold)})
            </div>
          </div>
          <Slider label="10× bigger model → ___× harder" value={trScaling10x} onChange={setTrScaling10x}
            min={1} max={10} step={0.1} format={v => `${v.toFixed(1)}×`} color={C.green}
            tip="If the model is 10× larger, how much harder is the safeguard to break? 1× = no scaling benefit, 2× = twice as hard, 3× = three times. This is the most uncertain parameter in the model." />

          <GroupLabel color={C.yellow}>Capability</GroupLabel>
          <Slider label="Threshold model size" value={nThreshold} onChange={setNThreshold}
            min={1e8} max={1e13} log format={v => `${fmtParams(v)} params`} color={C.yellow}
            tip="Minimum parameter count for the dangerous capability to exist. This is capability-specific — bioweapons may require different scale than cyber offense." />

          <GroupLabel color={C.red}>Threat Actor</GroupLabel>
          <div style={{ marginBottom: 10 }}>
            <select value={actorIdx} onChange={e => setActorIdx(Number(e.target.value))} style={{
              width: "100%", background: C.bgInner, border: `1px solid ${C.border}`,
              color: C.text, borderRadius: 4, padding: "6px 8px", fontSize: 11,
              fontFamily: "var(--f)", cursor: "pointer", outline: "none",
            }}>
              {ACTORS.map((a, i) => <option key={i} value={i}>{a.label}</option>)}
            </select>
          </div>
          {actorIdx === ACTORS.length - 1 && (
            <Slider label="Custom budget" value={customBudget} onChange={setCustomBudget}
              min={100} max={1e11} log format={fmt} color={C.red}
              tip="Custom threat actor compute budget." />
          )}
        </div>

        {/* Main */}
        <div style={{ flex: 1, padding: "20px 24px", overflowY: "auto", maxHeight: "calc(100vh - 90px)" }}>

          {/* Metrics */}
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 4, alignItems: "stretch" }}>
            <Metric label="Safeguard relevant for" value={selectedWindow >= 25 ? ">25yr" : `${selectedWindow}yr`}
              sub={`vs ${ACTORS[actorIdx].label}`} color={C.green} />
            <div style={{
              background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6,
              padding: "14px 16px", flex: 1, minWidth: 150,
              display: "flex", flexDirection: "column", justifyContent: "center",
            }}>
              <div style={{ fontSize: 10, color: C.dim, fontFamily: "var(--f)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 5 }}>
                Cheapest attack path today
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, color: C.muted, fontFamily: "var(--f)", lineHeight: 1.3 }}>
                {data[0]?.costRemove < data[0]?.costTrain
                  ? "Fine-tune to remove safeguard"
                  : "Train a new model from scratch"}
              </div>
              <div style={{ fontSize: 10, color: C.dim, marginTop: 4, fontFamily: "var(--f)" }}>
                {data[0]?.costRemove < data[0]?.costTrain
                  ? `${fmt(data[0]?.costRemove)} vs ${fmt(data[0]?.costTrain)} to train`
                  : `${fmt(data[0]?.costTrain)} vs ${fmt(data[0]?.costRemove)} to remove`}
              </div>
            </div>
          </div>

          {/* ── Cost Curves ── */}
          {tab === "model" && (<>
            <Section sub="Log-scale projection of attacker costs. Safeguard is relevant while costs exceed the actor budget (dashed line).">
              Attacker Cost Curves
            </Section>
            <Chart h={440}>
              <ComposedChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="year" stroke={C.dim} tick={tickStyle} />
                <YAxis scale="log" domain={['auto', 'auto']} tickFormatter={fmt} stroke={C.dim} tick={tickStyle} />
                <Tooltip content={<Tip />} />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--f)" }} />
                <Line type="monotone" dataKey="costTrain" name="Train from scratch" stroke={C.train} strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="costRemove" name="Remove safeguard" stroke={C.remove} strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="costAttack" name="Min attack cost" stroke={C.attack} strokeWidth={2} strokeDasharray="6 3" dot={false} />
                <ReferenceLine y={hwFloor} stroke={C.hw} strokeDasharray="2 4" strokeWidth={1}
                  label={{ value: "Hardware floor", position: "right", style: { fontSize: 9, fill: C.hw } }} />
                <ReferenceLine y={budget} stroke={C.red} strokeDasharray="4 4" strokeWidth={1.5}
                  label={{ value: `Actor budget: ${fmt(budget)}`, position: "insideTopRight", style: { fontSize: 9, fill: C.red } }} />
              </ComposedChart>
            </Chart>

            <Section sub="Above the ceiling, additional tamper resistance is wasted — the attacker trains from scratch.">
              Tamper Resistance Ceiling
            </Section>
            <Chart h={260}>
              <ComposedChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="year" stroke={C.dim} tick={tickStyle} />
                <YAxis scale="log" domain={['auto', 'auto']} tickFormatter={fmtX} stroke={C.dim} tick={tickStyle} />
                <Tooltip content={<Tip />} />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--f)" }} />
                <Area type="monotone" dataKey="maxUsefulTR" name="Max useful TR" fill={C.green + "22"} stroke={C.green} strokeWidth={2} />
                <ReferenceLine y={trBase * Math.pow(nThreshold / 1e9, Math.log10(trScaling10x))}
                  stroke={C.remove} strokeDasharray="6 3"
                  label={{ value: `Current TR (${(stepsToCircumvent).toLocaleString()} steps)`, position: "right", style: { fontSize: 9, fill: C.remove } }} />
              </ComposedChart>
            </Chart>
          </>)}

          {/* ── Decomposition ── */}
          {tab === "decomp" && (<>
            <Section sub="Hardware-only cost (orange, asymptotes) vs. hardware + algorithmic (purple, keeps falling). The gap is the algorithmic contribution.">
              Hardware vs. Algorithmic
            </Section>
            <Chart h={440}>
              <ComposedChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="year" stroke={C.dim} tick={tickStyle} />
                <YAxis scale="log" domain={['auto', 'auto']} tickFormatter={fmt} stroke={C.dim} tick={tickStyle} />
                <Tooltip content={<Tip />} />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--f)" }} />
                <Line type="monotone" dataKey="costTrainHwOnly" name="Hardware only" stroke={C.hw} strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="costTrain" name="Hardware + Algorithmic" stroke={C.combined} strokeWidth={2.5} dot={false} />
                <ReferenceLine y={hwFloor} stroke={C.hw} strokeDasharray="2 4"
                  label={{ value: `Hardware floor: ${fmt(hwFloor)}`, position: "right", style: { fontSize: 9, fill: C.hw } }} />
                <ReferenceLine y={budget} stroke={C.red} strokeDasharray="4 4" strokeWidth={1.5}
                  label={{ value: "Actor budget", position: "insideTopRight", style: { fontSize: 9, fill: C.red } }} />
              </ComposedChart>
            </Chart>

            <Section sub="Hardware saturates; algorithms decelerate but persist.">
              Annual Improvement Rates
            </Section>
            <Chart h={280}>
              <ComposedChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="year" stroke={C.dim} tick={tickStyle} />
                <YAxis domain={[0.9, 'auto']} tickFormatter={v => `${v.toFixed(1)}×`} stroke={C.dim} tick={tickStyle} />
                <Tooltip content={<Tip />} />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--f)" }} />
                <Line type="monotone" dataKey="effHwRate" name="Hardware" stroke={C.hw} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="effAlgoRate" name="Algorithmic" stroke={C.algo} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="effCombined" name="Combined" stroke={C.combined} strokeWidth={2} strokeDasharray="6 3" dot={false} />
                <ReferenceLine y={1} stroke={C.red} strokeDasharray="2 4" label={{ value: "No improvement", position: "right", style: { fontSize: 9, fill: C.red } }} />
              </ComposedChart>
            </Chart>

            <Section sub="Hardware hits its ceiling; algorithms keep compounding.">
              Cumulative Factors
            </Section>
            <Chart h={280}>
              <ComposedChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="year" stroke={C.dim} tick={tickStyle} />
                <YAxis scale="log" domain={['auto', 'auto']} tickFormatter={fmtX} stroke={C.dim} tick={tickStyle} />
                <Tooltip content={<Tip />} />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--f)" }} />
                <Line type="monotone" dataKey="hwFactor" name="Hardware" stroke={C.hw} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="cumAlgo" name="Algorithmic" stroke={C.algo} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="cumTrainReduction" name="Combined" stroke={C.combined} strokeWidth={2} strokeDasharray="6 3" dot={false} />
                <ReferenceLine y={headroom} stroke={C.hw} strokeDasharray="2 4"
                  label={{ value: `Hardware ceiling: ${headroom.toFixed(0)}×`, position: "right", style: { fontSize: 9, fill: C.hw } }} />
              </ComposedChart>
            </Chart>
          </>)}

          {/* ── Relevance ── */}
          {tab === "relevance" && (<>
            <Section sub="Years each actor type is blocked by the safeguard.">
              Relevance by Threat Actor
            </Section>
            <Chart h={260}>
              <BarChart data={windows} layout="vertical" margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis type="number" domain={[0, 25]} tick={tickStyle} stroke={C.dim} />
                <YAxis type="category" dataKey="label" width={160} tick={tickStyle} stroke={C.dim} />
                <Tooltip content={<Tip />} />
                <Bar dataKey="years" name="Years" radius={[0, 4, 4, 0]}>
                  {windows.map((_, i) => <Cell key={i} fill={[C.train, C.remove, C.attack, C.green][i]} />)}
                </Bar>
              </BarChart>
            </Chart>

            <Section sub="Timeline of protection.">Relevance Timeline</Section>
            <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20 }}>
              {windows.map((w, idx) => {
                const pct = Math.min((w.years / 25) * 100, 100);
                const col = [C.train, C.remove, C.attack, C.green][idx];
                return (
                  <div key={idx} style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                      <span style={{ fontSize: 11, color: col, fontWeight: 600 }}>{w.label}</span>
                      <span style={{ fontSize: 10, color: C.muted }}>{w.years >= 25 ? ">25yr" : `${w.years}yr`}</span>
                    </div>
                    <div style={{ background: C.border, borderRadius: 3, height: 20, position: "relative", overflow: "hidden" }}>
                      <div style={{
                        background: `linear-gradient(90deg, ${col}bb, ${col}33)`,
                        height: "100%", width: `${pct}%`, borderRadius: 3,
                      }} />
                      {[5, 10, 15, 20].map(y => (
                        <div key={y} style={{
                          position: "absolute", left: `${(y / 25) * 100}%`,
                          top: 0, bottom: 0, width: 1, background: C.dim + "33",
                        }} />
                      ))}
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 2 }}>
                      <span style={{ fontSize: 9, color: C.dim }}>2026</span>
                      <span style={{ fontSize: 9, color: C.dim }}>2051</span>
                    </div>
                  </div>
                );
              })}
            </div>

            <Section sub="Which attack path dominates over time.">Binding Constraint</Section>
            <Chart h={150}>
              <AreaChart data={data.map(d => ({
                year: d.year,
                "Removal cheaper": d.costRemove < d.costTrain ? 1 : 0,
                "Training cheaper": d.costTrain <= d.costRemove ? 1 : 0,
              }))} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                <XAxis dataKey="year" stroke={C.dim} tick={tickStyle} />
                <YAxis hide domain={[0, 1]} />
                <Area type="step" dataKey="Removal cheaper" fill={C.remove + "44"} stroke={C.remove} strokeWidth={0} />
                <Area type="step" dataKey="Training cheaper" fill={C.train + "44"} stroke={C.train} strokeWidth={0} />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--f)" }} />
              </AreaChart>
            </Chart>
          </>)}

          {/* ── Required TR ── */}
          {tab === "requiredTR" && (() => {
            // Compute required TR for each year and each actor
            const { headroom: hd, alpha: al, scaledComputePer1K: scaledCPK } = deriveInternal(p);
            const reqData = [];
            for (let t = 0; t <= 25; t++) {
              const tau = p.yearsToPlateauHw / 3;
              const hwF = t === 0 ? 1 : 1 + (hd - 1) * (1 - Math.exp(-t / tau));
              let cumAlgo = 1, cumAttack = 1;
              for (let i = 1; i <= t; i++) {
                const ad = Math.pow(0.5, i / p.algoHalflife);
                cumAlgo *= 1 + (p.algoRate - 1) * ad;
                cumAttack *= 1 + (p.attackRate - 1) * ad;
              }
              const costTrainT = p.costTrainBase / (hwF * cumAlgo);
              const computeBaseT = (scaledCPK * BASELINE_STEPS / 1000) / (hwF * cumAlgo);
              const costFtT = p.setupCost + computeBaseT;
              const trCeiling = costTrainT / costFtT * cumAttack;

              const row = { year: 2026 + t, t, trCeiling, costTrainT };
              ACTORS.filter(a => a.budget !== null).forEach(a => {
                if (costTrainT <= a.budget) {
                  row[`tr_${a.label}`] = null; // impossible
                  row[`impossible_${a.label}`] = true;
                } else {
                  row[`tr_${a.label}`] = a.budget * cumAttack / costFtT;
                  row[`impossible_${a.label}`] = false;
                }
              });
              reqData.push(row);
            }

            // Find the year training becomes affordable for each actor
            const trainAffordableYear = {};
            ACTORS.filter(a => a.budget !== null).forEach(a => {
              const yr = reqData.find(d => d.costTrainT <= a.budget);
              trainAffordableYear[a.label] = yr ? yr.year : null;
            });

            // Compute required annual TR growth rate for target windows
            const targetYears = [3, 5, 10];
            const trGrowthTable = targetYears.map(T => {
              const row = { target: `${T} years` };
              ACTORS.filter(a => a.budget !== null).forEach(a => {
                const d = reqData[T];
                if (!d || d[`impossible_${a.label}`]) {
                  row[a.label] = "Impossible";
                } else {
                  const trNeeded = d[`tr_${a.label}`];
                  // Find worst (highest) TR needed across years 0..T
                  let worst = 0;
                  let impossible = false;
                  for (let i = 0; i <= T; i++) {
                    if (reqData[i][`impossible_${a.label}`]) { impossible = true; break; }
                    worst = Math.max(worst, reqData[i][`tr_${a.label}`]);
                  }
                  if (impossible) row[a.label] = "Impossible";
                  else row[a.label] = worst;
                }
              });
              return row;
            });

            const actorColors = [C.train, C.remove, C.attack, C.green];
            const presetActors = ACTORS.filter(a => a.budget !== null);

            return (<>
              <Section mt={12} sub="What level of tamper resistance is needed to keep safeguards relevant for a given number of years? 'Impossible' means training from scratch becomes affordable regardless of TR.">
                Required Tamper Resistance
              </Section>

              {/* Summary table */}
              <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20, marginBottom: 20 }}>
                <div style={{ fontSize: 12, color: C.text, fontWeight: 600, marginBottom: 12 }}>
                  Minimum tamper resistance for target relevance windows
                </div>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "var(--f)" }}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                      <th style={{ textAlign: "left", padding: "6px 8px", color: C.dim }}>Target</th>
                      {presetActors.map((a, i) => (
                        <th key={i} style={{ textAlign: "right", padding: "6px 8px", color: actorColors[i], fontWeight: 600 }}>{a.label.split(" (")[0]}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {trGrowthTable.map((row, ri) => (
                      <tr key={ri} style={{ borderBottom: `1px solid ${C.border}22` }}>
                        <td style={{ padding: "8px 8px", color: C.text, fontWeight: 600 }}>{row.target}</td>
                        {presetActors.map((a, i) => (
                          <td key={i} style={{ padding: "8px 8px", textAlign: "right", fontFamily: "var(--f)" }}>
                            {row[a.label] === "Impossible"
                              ? <span style={{ color: C.red, fontSize: 10 }}>No TR helps — training affordable</span>
                              : <span style={{ color: actorColors[i], fontWeight: 700 }}>{fmtX(row[a.label])}</span>
                            }
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* When training becomes affordable */}
              <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20, marginBottom: 20 }}>
                <div style={{ fontSize: 12, color: C.text, fontWeight: 600, marginBottom: 12 }}>
                  When training from scratch becomes affordable (safeguard ceiling)
                </div>
                <div style={{ fontSize: 11, color: C.muted, marginBottom: 12, lineHeight: 1.6 }}>
                  After this year, no amount of tamper resistance helps — the attacker can simply train a new model without the safeguard.
                </div>
                {presetActors.map((a, i) => {
                  const yr = trainAffordableYear[a.label];
                  return (
                    <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}22` }}>
                      <span style={{ color: actorColors[i], fontWeight: 600 }}>{a.label}</span>
                      <span style={{ color: yr ? C.red : C.green, fontWeight: 600 }}>
                        {yr ? `${yr} (${yr - 2026} years)` : ">2051"}
                      </span>
                    </div>
                  );
                })}
              </div>

              {/* Chart: Required TR over time for each actor */}
              <Section sub="The tamper resistance needed at each year to keep the safeguard relevant. When the line disappears, training from scratch has become affordable and no TR helps.">
                Required TR Over Time
              </Section>
              <Chart h={400}>
                <ComposedChart data={reqData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="year" stroke={C.dim} tick={tickStyle} />
                  <YAxis scale="log" domain={['auto', 'auto']} tickFormatter={fmtX} stroke={C.dim} tick={tickStyle} />
                  <Tooltip content={<Tip />} />
                  <Legend wrapperStyle={{ fontSize: 10, fontFamily: "var(--f)" }} />
                  {presetActors.map((a, i) => (
                    <Line key={i} type="monotone" dataKey={`tr_${a.label}`}
                      name={a.label.split(" (")[0]}
                      stroke={actorColors[i]} strokeWidth={2} dot={false}
                      connectNulls={false} />
                  ))}
                  <Area type="monotone" dataKey="trCeiling" name="TR ceiling (useless above)"
                    fill={C.green + "11"} stroke={C.green} strokeWidth={1} strokeDasharray="4 4" />
                </ComposedChart>
              </Chart>

              {/* Interpretation */}
              <Section sub="What the required TR growth rate implies.">
                Implications
              </Section>
              <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20, fontSize: 11, color: C.muted, lineHeight: 1.8 }}>
                <p style={{ margin: "0 0 10px" }}>
                  <span style={{ color: C.text, fontWeight: 600 }}>Required TR grows exponentially.</span>{" "}
                  Because both base attack costs fall and attack methods improve each year, the tamper resistance needed to maintain relevance compounds. Linear or slowly-growing TR is overwhelmed within 1-2 years. Under the current parameters, TR must grow at roughly{" "}
                  <span style={{ color: C.yellow, fontWeight: 600 }}>
                    {(() => {
                      const d5 = reqData[5];
                      const d0 = reqData[0];
                      const actor = presetActors[0];
                      if (!d5 || d5[`impossible_${actor.label}`] || !d0[`tr_${actor.label}`]) return "N/A";
                      const r = Math.pow(d5[`tr_${actor.label}`] / Math.max(d0[`tr_${actor.label}`], 1), 0.2);
                      return `${r.toFixed(1)}×/yr`;
                    })()}
                  </span>{" "}
                  to keep up, just to protect against individuals.
                </p>
                <p style={{ margin: "0 0 10px" }}>
                  <span style={{ color: C.text, fontWeight: 600 }}>There is a hard time limit.</span>{" "}
                  Once training from scratch becomes affordable to an actor, tamper resistance is completely irrelevant for that actor class. This limit is set entirely by training cost decline and is outside the safeguard developer's control. The safeguard community should think of TR as buying time within a closing window, not as a permanent defense.
                </p>
                <p style={{ margin: "0 0 10px" }}>
                  <span style={{ color: C.text, fontWeight: 600 }}>The asymmetric race.</span>{" "}
                  The entire AI industry is working to reduce training and fine-tuning costs. Tamper resistance must outpace this to remain effective. Currently, perhaps a few dozen researchers work on TR, while the cost-reduction effort is backed by hundreds of billions of dollars. This structural asymmetry means TR improvements must be exceptionally efficient to matter.
                </p>
                <p style={{ margin: 0 }}>
                  <span style={{ color: C.text, fontWeight: 600 }}>Policy implication.</span>{" "}
                  The value of safeguards is in the time they buy. Policy should focus on what defensive infrastructure can be built within the safeguard window: data governance (restricting dangerous training data), compute monitoring (tracking large training runs), biosurveillance, and hardened defenses that work even after safeguards fail.
                </p>
              </div>
            </>);
          })()}

          {/* ── TR Investment ── */}
          {tab === "trInvest" && (<>
            <Section mt={12} sub={`${HEAT_N.toLocaleString()} parameterizations sampled. Shows how much additional protection time a given TR improvement buys across the range of plausible futures.`}>
              What Does a TR Improvement Buy You?
            </Section>

            {/* Mode toggle */}
            <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
              {[
                { id: "exact", label: "Exactly N years" },
                { id: "cumulative", label: "At least N years" },
              ].map(m => (
                <button key={m.id} onClick={() => setHeatMode(m.id)} style={{
                  background: heatMode === m.id ? C.algo + "22" : "transparent",
                  border: `1px solid ${heatMode === m.id ? C.algo + "66" : C.border}`,
                  color: heatMode === m.id ? C.algo : C.muted,
                  borderRadius: 4, padding: "4px 12px", fontSize: 11,
                  fontFamily: "var(--f)", cursor: "pointer",
                }}>{m.label}</button>
              ))}
            </div>

            {/* Heatmap */}
            <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20, overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--f)", fontSize: 11 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "6px 8px", color: C.dim, borderBottom: `1px solid ${C.border}`, minWidth: 80 }}>
                      TR improvement →
                    </th>
                    {Array.from({ length: HEAT_MAX_YR + 1 }, (_, y) => (
                      <th key={y} style={{ textAlign: "center", padding: "6px 6px", color: C.muted, borderBottom: `1px solid ${C.border}`, minWidth: 50 }}>
                        {y === HEAT_MAX_YR ? `${y}+yr` : `${y}yr`}
                      </th>
                    ))}
                    <th style={{ textAlign: "center", padding: "6px 8px", color: C.muted, borderBottom: `1px solid ${C.border}`, minWidth: 60 }}>Mean</th>
                    <th style={{ textAlign: "center", padding: "6px 8px", color: C.muted, borderBottom: `1px solid ${C.border}`, minWidth: 60 }}>Median</th>
                  </tr>
                </thead>
                <tbody>
                  {heatRows.map((row, ri) => (
                    <tr key={ri}>
                      <td style={{ padding: "8px 8px", color: C.algo, fontWeight: 700, borderBottom: `1px solid ${C.border}22` }}>
                        {row.tr}
                      </td>
                      {Array.from({ length: HEAT_MAX_YR + 1 }, (_, y) => {
                        const label = y === HEAT_MAX_YR ? `${y}+` : `${y}`;
                        const val = heatMode === "cumulative" ? row[`gte_${y}`] : row[label];
                        const hc = val === 0 ? C.bgInner : val < 5 ? C.algo + "15" : val < 15 ? C.algo + "30" : val < 30 ? C.algo + "50" : val < 50 ? C.algo + "70" : C.algo + "90";
                        return (
                          <td key={y} style={{
                            padding: "8px 4px", textAlign: "center",
                            background: hc,
                            color: val > 0 ? C.text : C.dim,
                            fontWeight: val >= 30 ? 700 : 400,
                            borderBottom: `1px solid ${C.border}22`,
                            borderRight: `1px solid ${C.border}11`,
                          }}>
                            {val > 0 ? `${val}%` : "·"}
                          </td>
                        );
                      })}
                      <td style={{ padding: "8px 6px", textAlign: "center", color: C.yellow, fontWeight: 600, borderBottom: `1px solid ${C.border}22` }}>
                        {row.mean}yr
                      </td>
                      <td style={{ padding: "8px 6px", textAlign: "center", color: C.green, fontWeight: 600, borderBottom: `1px solid ${C.border}22` }}>
                        {row.median}yr
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Reading guide */}
            <div style={{ marginTop: 12, fontSize: 10, color: C.dim, lineHeight: 1.6, fontFamily: "var(--f)" }}>
              {heatMode === "exact"
                ? `Read as: "A ${TR_MULTS[3]}× TR improvement buys exactly 1 year of additional protection in ${heatRows[3]?.["1"] || 0}% of plausible parameterizations." Darker cells = more likely outcomes.`
                : `Read as: "A ${TR_MULTS[3]}× TR improvement buys at least 1 year of additional protection in ${heatRows[3]?.gte_1 || 0}% of plausible parameterizations." Use this view to answer "what's the probability my investment buys at least N years?"`
              }
            </div>

            {/* Interpretation */}
            <Section sub="Key patterns in the data.">Interpretation</Section>
            <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20, fontSize: 11, color: C.muted, lineHeight: 1.8 }}>
              <p style={{ margin: "0 0 10px" }}>
                <span style={{ color: C.text, fontWeight: 600 }}>The "0 years" column is the training-path regime.</span>{" "}
                When a TR improvement buys zero additional time, it means the attacker is already on the training-from-scratch path where tamper resistance is irrelevant.
                Against {ACTORS[actorIdx].label}, this happens in{" "}
                <span style={{ color: C.red, fontWeight: 600 }}>{heatRows[0]?.["0"] || 0}%</span> of worlds
                even for small improvements.
              </p>
              <p style={{ margin: "0 0 10px" }}>
                <span style={{ color: C.text, fontWeight: 600 }}>Diminishing returns per order of magnitude.</span>{" "}
                Going from 10× to 100× adds roughly as much time as going from 1× to 10×. Time gained scales as log(TR improvement).
              </p>
              <p style={{ margin: "0 0 10px" }}>
                <span style={{ color: C.text, fontWeight: 600 }}>The research investment question.</span>{" "}
                If a team works for 6 months and achieves 3× with 50% probability or 10× with 15% probability,
                the expected gain is roughly{" "}
                {(0.15 * (heatRows[3]?.mean || 0) + 0.5 * (heatRows[1]?.mean || 0) + 0.35 * 0).toFixed(1)} years —
                weighed against 0.5 years of researcher time.
              </p>
              <p style={{ margin: 0 }}>
                <span style={{ color: C.text, fontWeight: 600 }}>Minimum viable improvement.</span>{" "}
                For a {">"} 50% chance of buying at least 1 year of protection, you need roughly{" "}
                <span style={{ color: C.algo, fontWeight: 700 }}>
                  {(() => {
                    for (const row of heatRows) {
                      if (row.gte_1 >= 50) return row.tr;
                    }
                    return `>${TR_MULTS[TR_MULTS.length - 1]}×`;
                  })()}
                </span>{" "}
                against {ACTORS[actorIdx].label}.
              </p>
            </div>
          </>)}

          {/* ── Sensitivity ── */}
          {tab === "sensitivity" && (<>
            <Section mt={12} sub={`${mcResults.n.toLocaleString()} random parameterizations sampled from plausible ranges. Actor: ${ACTORS[actorIdx].label}.`}>
              Global Sensitivity Analysis
            </Section>

            {/* Distribution summary */}
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
              <Metric label="Median outcome" value={`${mcResults.percentiles.p50}yr`} sub="50th percentile" color={C.green} />
              <Metric label="Optimistic" value={`${mcResults.percentiles.p90}yr`} sub="90th percentile" color={C.algo} />
              <Metric label="Pessimistic" value={`${mcResults.percentiles.p10}yr`} sub="10th percentile" color={C.red} />
              <Metric label="Full range" value={`${mcResults.percentiles.p5}–${mcResults.percentiles.p95}yr`} sub="5th–95th percentile" color={C.muted} />
            </div>

            {/* Histogram */}
            <Section sub="Distribution of safeguard relevance windows across all sampled parameterizations.">
              Outcome Distribution
            </Section>
            <Chart h={280}>
              <BarChart data={mcResults.hist.filter(h => h.count > 0)} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="year" stroke={C.dim} tick={tickStyle} label={{ value: "Years of relevance", position: "bottom", style: { fontSize: 10, fill: C.dim } }} />
                <YAxis tickFormatter={v => `${v}%`} stroke={C.dim} tick={tickStyle} />
                <Tooltip content={<Tip />} />
                <Bar dataKey="pct" name="% of runs" radius={[3, 3, 0, 0]}>
                  {mcResults.hist.filter(h => h.count > 0).map((h, i) => (
                    <Cell key={i} fill={h.year === ">25" ? C.green + "cc" : C.algo + "88"} />
                  ))}
                </Bar>
              </BarChart>
            </Chart>

            {/* Variance decomposition */}
            <Section sub="What fraction of the variation in outcomes is explained by each parameter. Higher = more influential = higher-value research target.">
              Variance Attribution
            </Section>
            <Chart h={Math.max(360, mcResults.decomp.length * 38)}>
              <BarChart data={mcResults.decomp} layout="vertical" margin={{ top: 10, right: 40, left: 180, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis type="number" tickFormatter={v => `${v}%`} tick={tickStyle} stroke={C.dim}
                  label={{ value: "% of outcome variance explained", position: "bottom", style: { fontSize: 10, fill: C.dim } }} />
                <YAxis type="category" dataKey="label" width={170} tick={tickStyle} stroke={C.dim} />
                <Tooltip content={<Tip />} />
                <Bar dataKey="importancePct" name="Variance explained (%)" radius={[0, 4, 4, 0]}>
                  {mcResults.decomp.map((d, i) => (
                    <Cell key={i} fill={i === 0 ? C.red + "cc" : i < 3 ? C.yellow + "99" : C.algo + "55"} />
                  ))}
                </Bar>
              </BarChart>
            </Chart>

            {/* Ranked findings */}
            <Section sub="Parameters ranked by how much they affect conclusions.">
              Key Findings
            </Section>
            <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20 }}>
              {mcResults.decomp.slice(0, 5).map((d, i) => (
                <div key={i} style={{ padding: "10px 0", borderBottom: i < 4 ? `1px solid ${C.border}` : "none" }}>
                  <div style={{ fontSize: 12, color: C.text, fontWeight: 600, marginBottom: 3, display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{
                      background: i === 0 ? C.red + "33" : C.border,
                      color: i === 0 ? C.red : C.muted,
                      fontSize: 10, padding: "2px 6px", borderRadius: 3, fontWeight: 700,
                    }}>#{i + 1}</span>
                    {d.label}
                    <span style={{ fontSize: 10, color: C.dim, fontWeight: 400 }}>
                      ({d.importancePct}% of variance)
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.6 }}>
                    Low values → {d.loMean}yr average. High values → {d.hiMean}yr average.
                    {i === 0 && " This is the highest-leverage empirical question — investing in measuring this parameter reduces uncertainty most."}
                  </div>
                </div>
              ))}
            </div>

            {/* Cross-actor comparison */}
            <Section sub="How the distribution shifts across actor types.">
              Outcomes by Actor Type
            </Section>
            <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "var(--f)" }}>
                <thead>
                  <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                    <th style={{ textAlign: "left", padding: "6px 8px", color: C.dim }}>Actor</th>
                    <th style={{ textAlign: "right", padding: "6px 8px", color: C.dim }}>10th %ile</th>
                    <th style={{ textAlign: "right", padding: "6px 8px", color: C.dim }}>Median</th>
                    <th style={{ textAlign: "right", padding: "6px 8px", color: C.dim }}>90th %ile</th>
                    <th style={{ textAlign: "right", padding: "6px 8px", color: C.dim }}>Mean</th>
                  </tr>
                </thead>
                <tbody>
                  {mcByActor.map((a, i) => (
                    <tr key={i} style={{ borderBottom: `1px solid ${C.border}22` }}>
                      <td style={{ padding: "6px 8px", color: [C.train, C.remove, C.attack, C.green][i], fontWeight: 600 }}>{a.label}</td>
                      <td style={{ padding: "6px 8px", color: C.muted, textAlign: "right" }}>{a.percentiles.p10}yr</td>
                      <td style={{ padding: "6px 8px", color: C.text, textAlign: "right", fontWeight: 600 }}>{a.percentiles.p50}yr</td>
                      <td style={{ padding: "6px 8px", color: C.muted, textAlign: "right" }}>{a.percentiles.p90}yr</td>
                      <td style={{ padding: "6px 8px", color: C.muted, textAlign: "right" }}>{a.mean}yr</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Plausible ranges (advanced toggle) */}
            <Section sub="The parameter ranges used for sampling. These represent the boundaries of plausible values based on current literature.">
              Assumed Parameter Ranges
            </Section>
            <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <span style={{ fontSize: 11, color: C.muted }}>
                  {showAdvRanges ? "Edit ranges below. Changes trigger a new Monte Carlo run." : "Using literature-based defaults."}
                </span>
                <button onClick={() => setShowAdvRanges(s => !s)} style={{
                  background: "transparent", border: `1px solid ${C.border}`, color: C.muted,
                  borderRadius: 4, padding: "3px 10px", fontSize: 10, fontFamily: "var(--f)", cursor: "pointer",
                }}>
                  {showAdvRanges ? "Hide" : "Customize ranges"}
                </button>
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, fontFamily: "var(--f)" }}>
                <thead>
                  <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                    <th style={{ textAlign: "left", padding: "5px 8px", color: C.dim }}>Parameter</th>
                    <th style={{ textAlign: "right", padding: "5px 8px", color: C.dim }}>Low</th>
                    <th style={{ textAlign: "right", padding: "5px 8px", color: C.dim }}>High</th>
                    {!showAdvRanges && <th style={{ textAlign: "left", padding: "5px 8px", color: C.dim }}>Justification</th>}
                  </tr>
                </thead>
                <tbody>
                  {PARAM_RANGES.map((r, i) => {
                    const ov = rangeOverrides[r.key];
                    const lo = ov ? ov[0] : r.lo;
                    const hi = ov ? ov[1] : r.hi;
                    const fmtV = (v) => r.log && v >= 1e6 ? `${(v/1e6).toFixed(0)}M` : r.log && v >= 1e3 ? `${(v/1e3).toFixed(0)}K` : r.key.includes("Rate") || r.key.includes("rate") || r.key === "hwRate" || r.key === "trScaling10x" ? `${v.toFixed(2)}×` : r.key === "nThreshold" ? fmtParams(v) : r.log ? fmt(v) : v.toFixed?.(1) ?? v;
                    return (
                      <tr key={i} style={{ borderBottom: `1px solid ${C.border}22` }}>
                        <td style={{ padding: "5px 8px", color: C.muted }}>{r.label}</td>
                        <td style={{ padding: "5px 8px", color: C.text, textAlign: "right" }}>
                          {showAdvRanges ? (
                            <input type="number" value={lo} onChange={e => {
                              const v = Number(e.target.value);
                              if (!isNaN(v)) setRangeOverrides(prev => ({ ...prev, [r.key]: [v, hi] }));
                            }} style={{
                              width: 80, background: C.bgInner, border: `1px solid ${C.border}`,
                              color: C.text, borderRadius: 3, padding: "2px 6px", fontSize: 10,
                              fontFamily: "var(--f)", textAlign: "right",
                            }} />
                          ) : fmtV(lo)}
                        </td>
                        <td style={{ padding: "5px 8px", color: C.text, textAlign: "right" }}>
                          {showAdvRanges ? (
                            <input type="number" value={hi} onChange={e => {
                              const v = Number(e.target.value);
                              if (!isNaN(v)) setRangeOverrides(prev => ({ ...prev, [r.key]: [lo, v] }));
                            }} style={{
                              width: 80, background: C.bgInner, border: `1px solid ${C.border}`,
                              color: C.text, borderRadius: 3, padding: "2px 6px", fontSize: 10,
                              fontFamily: "var(--f)", textAlign: "right",
                            }} />
                          ) : fmtV(hi)}
                        </td>
                        {!showAdvRanges && <td style={{ padding: "5px 8px", color: C.dim, fontSize: 10, lineHeight: 1.4 }}>{r.src}</td>}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {showAdvRanges && Object.keys(rangeOverrides).length > 0 && (
                <button onClick={() => setRangeOverrides({})} style={{
                  marginTop: 10, background: "transparent", border: `1px solid ${C.border}`,
                  color: C.red, borderRadius: 4, padding: "4px 12px", fontSize: 10,
                  fontFamily: "var(--f)", cursor: "pointer",
                }}>Reset to defaults</button>
              )}
            </div>
          </>)}

          {/* ── Details ── */}
          {tab === "details" && (<>
            <Section>Model Equations</Section>
            <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20, lineHeight: 1.9, fontSize: 12 }}>
              {[
                { c: C.train, t: "Core Inequality",
                  eq: "Safeguard relevant ⟺ min(Cost_remove(t), Cost_train(t)) > Budget" },
                { c: C.hw, t: "Hardware (Asymptotic)",
                  eq: `hw_factor(t) = 1 + (headroom - 1) × (1 - e^(-t/τ))\n\nheadroom = hw_rate ^ years_to_plateau = ${headroom.toFixed(0)}×\nτ = years_to_plateau / 3 = ${(yearsToPlateauHw/3).toFixed(1)}yr\nFloor cost = ${fmt(hwFloor)}` },
                { c: C.algo, t: "Algorithmic (Decelerating)",
                  eq: `algo_rate(t) = 1 + (initial_rate - 1) × 0.5^(t / halflife)\nDecays toward 1× (no improvement).\n\nCurrent: ${algoRate.toFixed(1)}×/yr, half-life: ${algoHalflife}yr\nAt year 10: ~${(1 + (algoRate - 1) * Math.pow(0.5, 10/algoHalflife)).toFixed(2)}×/yr` },
                { c: C.combined, t: "Combined Training Cost",
                  eq: "Cost_train(t) = base / (hw_factor(t) × Π algo_rate(i))" },
                { c: C.remove, t: "Cost to Remove Safeguard",
                  eq: `Cost_remove(t) = setup + (compute_per_1K × N/70B × eff_steps / 1000) / decline(t) / attack_improvement(t)\n\nCompute per 1K steps: ${fmt(computePer1K)} at 70B → ${fmt(scaledComputePer1K)} at ${fmtParams(nThreshold)} (O(N) scaling)\nEffective steps = ${stepsToCircumvent.toLocaleString()} × (N / 1B)^${Math.log10(trScaling10x).toFixed(3)}\n               = ${(stepsToCircumvent * Math.pow(nThreshold / 1e9, Math.log10(trScaling10x))).toFixed(0)} at ${fmtParams(nThreshold)} params\nCost today = ${fmt(costToCircumventToday)}\n\nAttack methods decelerate on same halflife as algo.` },
                { c: C.remove, t: "Compute Cost Scaling (O(N))",
                  eq: `Fine-tuning cost per step scales linearly with model parameter count.\nThis follows from the forward + backward pass being O(N) in parameters.\n\nA 7B model costs ~10× less per step than 70B.\nA 700B model costs ~10× more per step than 70B.\n\nWe assume the attacker uses the cheapest effective method (QLoRA),\nwhich at 70B costs ~$0.50-$1/1K steps. LoRA: ~$5-$50/1K. Full FT: ~$50-$500/1K.\nDefault (${fmt(computePer1K)}/1K at 70B) is conservative — assumes cheapest viable attack.` },
                { c: C.green, t: "Tamper Resistance Ceiling",
                  eq: "Max_useful_TR = Cost_train / Cost_ft × attack_improvement\n\nBeyond this, attacker trains from scratch.\nAdditional TR investment is wasted." },
              ].map(({ c, t, eq }, i) => (
                <div key={i} style={{ marginBottom: 14 }}>
                  <div style={{ color: c, fontWeight: 600, marginBottom: 4, fontSize: 12 }}>{t}</div>
                  <div style={{
                    color: C.muted, fontFamily: "var(--f)", fontSize: 11,
                    padding: "10px 14px", background: C.bgInner, borderRadius: 5,
                    whiteSpace: "pre-wrap", lineHeight: 1.7, borderLeft: `2px solid ${c}33`,
                  }}>{eq}</div>
                </div>
              ))}
            </div>

            <Section>Assumptions & Caveats</Section>
            <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20, lineHeight: 1.8, fontSize: 11, color: C.muted }}>
              {[
                { c: C.red, t: "Critical", d: "Assumes any parameter-level safeguard can eventually be removed through fine-tuning. May not hold for all architectures." },
                { c: C.text, t: "Open weights only", d: "Attacker has full parameter access. Closed models have additional protections not modeled." },
                { c: C.hw, t: "Hardware ceiling", d: "Modeled with finite total headroom based on transistor limits (~2nm CMOS), memory bandwidth (~28%/yr), cooling, and fab economics ($20B+/gen). GPUs are ~10⁸× above Landauer, but practical headroom is ~100-1000×." },
                { c: C.algo, t: "Algorithmic uncertainty", d: "No known physical ceiling. Controlled estimates: ~3×/yr. But Epoch research shows most gains came from ~2 scale-dependent innovations plus better data, not steady discovery of new algorithms." },
                { c: C.text, t: "Data costs omitted", d: "Training requires data, not just compute. For dangerous capabilities, data scarcity may be a natural chokepoint not captured here." },
                { c: C.text, t: "TR scaling unknown", d: "Most tamper-resistance research (TAR, Circuit Breakers, Deep Ignorance) covers only 6-8B scale. Scaling behavior is the least grounded parameter." },
                { c: C.text, t: "O(N) compute scaling", d: "Fine-tuning cost per step scales linearly with parameters. Accurate for full fine-tuning and LoRA. QLoRA is sublinear in practice due to quantization, so this is slightly conservative (overestimates cost at very large scales)." },
                { c: C.text, t: "Static budgets", d: "Actor budgets are constant. Cloud democratization may shift effective budgets over time." },
              ].map(({ c, t, d }, i) => (
                <p key={i} style={{ marginTop: i === 0 ? 0 : 6, marginBottom: 6 }}>
                  <span style={{ color: c, fontWeight: 600 }}>{t}: </span>{d}
                </p>
              ))}
            </div>
          </>)}

          {/* ── Sources ── */}
          {tab === "sources" && (<>
            <Section sub="Citable sources and justifications for every parameter range and model equation. Confidence levels reflect how well-grounded each assumption is.">
              Parameter Sources & Justifications
            </Section>

            {/* Confidence legend */}
            <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
              {[
                { label: "High", color: C.green, desc: "Well-sourced empirical data" },
                { label: "Medium", color: C.yellow, desc: "Reasonable estimates, some uncertainty" },
                { label: "Low", color: C.red, desc: "Expert guess or thin evidence" },
              ].map(({ label, color, desc }) => (
                <div key={label} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: C.muted }}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: color, display: "inline-block" }} />
                  <span style={{ color, fontWeight: 600 }}>{label}</span> — {desc}
                </div>
              ))}
            </div>

            {[
              {
                param: "Training cost today",
                range: "$10M – $50B",
                confidence: "medium",
                lo: "MosaicML MPT-7B (~$1-5M compute); 13B models ~$5-10M. $10M captures smallest models with potentially dangerous capabilities.",
                hi: "$50B captures projected near-future frontier runs with massive overtraining. Current frontier (GPT-4, Llama 3 405B) estimated at $100M-$500M all-in.",
                central: "$500M — Meta Llama 3 405B disclosed 30.84M GPU-hours on H100 (~$60-200M compute). Frontier models with failed runs, salaries, infrastructure: $200M-$500M.",
                sources: [
                  "Meta, 'The Llama 3 Herd of Models,' arXiv:2407.21783 (2024) — 30.84M GPU-hours disclosed",
                  "Sevilla et al., 'Compute Trends Across Three Eras of Machine Learning,' arXiv:2202.05924 (2022), Epoch AI dataset",
                  "Semianalysis, 'GPT-4 Architecture, Infrastructure, Training Dataset, Costs...' (2023)",
                ],
              },
              {
                param: "Attack setup cost",
                range: "$2K – $20K",
                confidence: "low",
                lo: "$2K represents a skilled individual spending ~1 week sourcing data + minimal cloud costs for experimentation.",
                hi: "$20K includes comprehensive dataset curation ($1-5K), H100 cloud rental for experimentation ($2-10K), and specialized expertise ($5-10K).",
                central: "$5K — no published work systematically breaks down adversarial fine-tuning preparation costs. This is an author estimate.",
                sources: [
                  "Qi et al., 'Fine-tuning Aligned Language Models Compromises Safety,' arXiv:2310.03693 (2023) — demonstrated trivial compute cost for the attack itself",
                  "Yang et al., 'Shadow Alignment,' arXiv:2310.02949 (2023) — similar results on attack simplicity",
                  "No systematic attack cost decomposition exists in the literature [GAP]",
                ],
              },
              {
                param: "Compute cost per 1K steps (at 70B)",
                range: "$20 – $500",
                confidence: "medium-high",
                lo: "$20 — 8×H100 at $2/hr spot pricing, ~20 steps/min throughput. 1K steps ≈ 50 min ≈ $13-27. Efficient setups with spot pricing.",
                hi: "$500 — 16×H100 at $4/hr on-demand, conservative 5 steps/min with overhead. Includes communication overhead and redundancy at scale.",
                central: "MC excludes QLoRA ($0.50-$5/1K) for safeguard-optimistic analysis. Full fine-tuning is the assumed attack method. This is a conservative modeling choice — if QLoRA suffices, real costs are 10-100× lower.",
                sources: [
                  "Lambda Cloud, RunPod, CoreWeave pricing (2024) — H100 spot: $1.99-$2.49/hr",
                  "Hugging Face, 'Fine-tuning Llama 2 70B using PyTorch FSDP' (2023) — throughput benchmarks",
                  "Hu et al., 'LoRA,' arXiv:2106.09685 (2021)",
                  "Dettmers et al., 'QLoRA,' arXiv:2305.14314 (2023)",
                ],
              },
              {
                param: "Hardware improvement rate",
                range: "1.1× – 1.55×/yr",
                confidence: "high",
                lo: "1.1×/yr — post-Moore's-Law rates if current scaling approaches hit memory bandwidth walls and diminishing lithography returns.",
                hi: "1.55×/yr — optimistic case with chiplet architectures, HBM4 memory, and competitive GPU market dynamics.",
                central: "1.32×/yr — directly from Epoch AI: GPU FLOP/$ doubles every ~2.5yr. 2^(1/2.5) ≈ 1.32×/yr over 2006-2021.",
                sources: [
                  "Hobbhahn, Besiroglu et al., 'Trends in Machine Learning Hardware,' Epoch AI (2023) — primary source",
                  "Epoch AI Compute Trends dataset — GPU FLOP/$ historical time series",
                ],
              },
              {
                param: "Years until hardware plateaus",
                range: "5 – 15yr",
                confidence: "medium",
                lo: "5yr (~2030) — near-term exhaustion of EUV lithographic scaling, with memory bandwidth as binding constraint for AI workloads.",
                hi: "15yr (~2040) — continued innovation via chiplets, 3D stacking, CFET transistors, 2D materials, and photonic interconnects sustaining improvement beyond lithographic limits.",
                central: "~10yr — consensus: conventional scaling exhausts by ~2030, packaging and architecture innovations sustain gains for another ~5 years.",
                sources: [
                  "IEEE IRDS 'More Moore' and 'Beyond CMOS' chapters (2022 edition)",
                  "TSMC technology roadmap presentations (2023-2024 investor calls)",
                  "Hobbhahn et al., Epoch AI (2023) — hardware headroom discussion",
                ],
              },
              {
                param: "Algorithmic efficiency rate",
                range: "1.5× – 4.5×/yr",
                confidence: "medium-high",
                lo: "1.5×/yr — deceleration from current pace if recent gains are largely catch-up effects (adopting known techniques to LLMs).",
                hi: "4.5×/yr — high end of Epoch estimates, includes architecture-level jumps. Epoch headline: effective compute doubles every ~5-8 months for language modeling.",
                central: "~3×/yr — Epoch AI's most-cited figure. Compute required halves every ~8 months. Note: conflates catch-up gains with genuine frontier innovation.",
                sources: [
                  "Ho, Besiroglu et al., 'Algorithmic Progress in Language Models,' arXiv:2403.05812 (2024) — primary source",
                  "Erdil & Besiroglu, 'Algorithmic Progress in Computer Vision,' arXiv:2212.05153 (2022)",
                  "Hernandez & Brown, 'Measuring the Algorithmic Efficiency of Neural Networks,' arXiv:2005.04305 (2020)",
                ],
              },
              {
                param: "Years of fast algorithmic progress",
                range: "3 – 20yr",
                confidence: "low",
                lo: "3yr — 'low-hanging fruit' from transformer scaling is nearly exhausted. Historical precedent: CNN revolution (2012) saw rapid gains for ~5yr before plateauing (VGG → ResNet → EfficientNet).",
                hi: "20yr — AI-assisted ML research sustains rapid progress; enormous search space for architectures, training algorithms, data strategies. No strong empirical basis.",
                central: "8yr — rough midpoint. Fundamentally a forecasting question about the pace of scientific discovery. No reliable methodology exists.",
                sources: [
                  "Bloom et al., 'Are Ideas Getting Harder to Find?' AER (2020) — empirical support for decelerating discovery rates",
                  "Cotra, 'Forecasting Transformative AI,' Open Philanthropy (2020) — discusses efficiency trends in long-run context",
                  "No direct empirical source for this parameter [STRUCTURAL ASSUMPTION]",
                ],
              },
              {
                param: "Attack method improvement",
                range: "1.0× – 2.5×/yr",
                confidence: "low",
                lo: "1.0×/yr — adversarial fine-tuning methods stagnate; current methods (LoRA, full FT) are near-optimal.",
                hi: "2.5×/yr — based on 2021-2024 timeline: LoRA (Jun 2021) → QLoRA (May 2023) → GaLore (Mar 2024), each reducing requirements by 2-10×. Capped at 2.5× for safeguard-optimistic analysis.",
                central: "2.0×/yr — author estimate. No Epoch-quality analysis of adversarial method improvement rates exists. Note: the model couples attack deceleration to algoHalflife, which is a simplifying assumption with no empirical basis.",
                sources: [
                  "Hu et al., 'LoRA,' arXiv:2106.09685 (2021)",
                  "Dettmers et al., 'QLoRA,' arXiv:2305.14314 (2023)",
                  "Zhao et al., 'GaLore,' arXiv:2403.03507 (2024)",
                  "No systematic survey of adversarial fine-tuning efficiency exists [MAJOR GAP]",
                ],
              },
              {
                param: "Steps to circumvent safeguard",
                range: "100 – 100M",
                confidence: "medium (lo) / low (hi)",
                lo: "100 steps — standard RLHF/DPO safety training is trivially removable. Qi et al. (2023): removed safety alignment with <100 examples, <100 steps.",
                hi: "100M steps — no artificial cap on achievable TR. At 100M steps on 70B, compute cost (~$2M-$50M) approaches training cost, so the model enters the training-limited regime. This is the natural ceiling, not an empirical bound.",
                central: "10,000 steps — Deep Ignorance (arXiv:2508.06601, ICLR 2026) demonstrated resistance to 10K steps / 300M tokens on biothreat text at 6.9B scale. Best published result as of 2026.",
                sources: [
                  "Qi et al., 'Fine-tuning Aligned Language Models Compromises Safety,' arXiv:2310.03693 (2023) — baseline removability",
                  "Tamirisa et al., 'Tamper-Resistant Safeguards for Open-Weight LLMs,' arXiv:2408.00761 (ICLR 2025) — TAR, tested at 8B",
                  "Lynch et al., 'Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards,' arXiv:2508.06601 (ICLR 2026) — 10K steps at 6.9B",
                  "TamperBench, arXiv:2602.06911 (2026) — notes real-world resistance may be 'only several hundred steps' for current methods",
                  "Rosati et al., 'Representation Noising,' arXiv:2405.14577 (2024)",
                ],
              },
              {
                param: "TR scaling with model size",
                range: "1.0× – 5.0× per 10× params",
                confidence: "very low",
                lo: "1.0× — no scaling benefit. TR difficulty may depend on the method, not the model size.",
                hi: "5.0× — theoretical argument: larger models have more redundant representations, making selective unlearning harder. Plausible but entirely unproven.",
                central: "2.0× — author estimate. No empirical data exists above 8B scale. TAR tested at 8B only. Deep Ignorance at 6.9B only.",
                sources: [
                  "No empirical source exists [CRITICAL GAP]",
                  "Theoretical argument discussed informally in Tamirisa et al. (2024) but not tested",
                  "This is the #1 priority for empirical research — a study testing TR at 1B/7B/70B/405B would transform model validity",
                ],
              },
              {
                param: "Capability threshold",
                range: "1B – 500B",
                confidence: "medium",
                lo: "1B — some narrow capabilities appear at small scales, but likely too small for meaningful dangerous capability. Included to capture uncertainty.",
                hi: "500B — largest dense models (Llama 3 405B). Truly dangerous capabilities (e.g., novel pathogen design) may require very large models. Reasonable cap.",
                central: "70B — Llama 2/3 70B is the most studied open-weight model with significant capabilities. Comparable to GPT-3.5-era closed models.",
                sources: [
                  "Mouton et al., 'The Operational Risks of AI in Large-Scale Biological Attacks,' RAND (2024)",
                  "Li et al., 'The WMDP Benchmark,' arXiv:2403.03218 (2024) — measures dangerous knowledge",
                  "Wei et al., 'Emergent Abilities of Large Language Models,' TMLR (2022)",
                  "Schaeffer et al., 'Are Emergent Abilities a Mirage?' arXiv:2304.15004 (2023)",
                ],
              },
            ].map(({ param, range, confidence, lo, hi, central, sources }, idx) => {
              const confColor = confidence.startsWith("high") ? C.green
                : confidence.startsWith("medium") ? C.yellow
                : confidence.startsWith("very") ? C.red : C.red;
              return (
                <div key={idx} style={{
                  background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6,
                  padding: "16px 18px", marginBottom: 12,
                  borderLeft: `3px solid ${confColor}33`,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: C.text }}>{param}</span>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 11, color: C.muted, fontFamily: "var(--f)" }}>MC range: {range}</span>
                      <span style={{
                        fontSize: 9, fontWeight: 700, color: confColor, textTransform: "uppercase",
                        background: confColor + "18", padding: "2px 6px", borderRadius: 3,
                      }}>{confidence}</span>
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7, marginBottom: 8 }}>
                    <div><span style={{ color: C.text, fontWeight: 600 }}>Lower bound: </span>{lo}</div>
                    <div><span style={{ color: C.text, fontWeight: 600 }}>Upper bound: </span>{hi}</div>
                    <div><span style={{ color: C.text, fontWeight: 600 }}>Central estimate: </span>{central}</div>
                  </div>
                  <div style={{ fontSize: 10, color: C.dim, lineHeight: 1.6, paddingTop: 6, borderTop: `1px solid ${C.border}22` }}>
                    {sources.map((s, i) => (
                      <div key={i} style={{ marginBottom: 2 }}>
                        {s.includes("[GAP]") || s.includes("[CRITICAL GAP]") || s.includes("[MAJOR GAP]") || s.includes("[STRUCTURAL ASSUMPTION]")
                          ? <span style={{ color: C.red }}>{s}</span>
                          : <span>{s}</span>
                        }
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}

            <Section mt={28}>Model Equation Sources</Section>
            {[
              {
                eq: "Hardware asymptotic model",
                form: "hw_factor(t) = 1 + (headroom - 1) × (1 - e^(-t/τ))",
                confidence: "medium",
                justification: "Standard saturating exponential. Widely used in engineering for approach-to-limit dynamics. τ = plateau/3 means 'years to plateau' corresponds to ~95% of total gains captured. headroom = rate^years extrapolates current annual rate to compute total remaining improvement.",
                sources: [
                  "Standard engineering model — no specific paper needed",
                  "Farmer & Lafond, 'How Predictable is Technological Progress?' Research Policy (2016) — S-curve alternatives",
                ],
              },
              {
                eq: "Algorithmic deceleration",
                form: "algo_rate(t) = 1 + (rate - 1) × 0.5^(t / halflife)",
                confidence: "medium",
                justification: "Exponential decay of the excess rate above 1×. Unlike hardware, algorithmic progress has no known physical ceiling, so a plateau model is inappropriate. Instead, the rate itself decays — progress continues forever but decelerates. Similar to models in endogenous growth theory.",
                sources: [
                  "Bloom et al., 'Are Ideas Getting Harder to Find?' AER (2020) — empirical support for decelerating discovery",
                  "Jones, 'R&D-Based Models of Economic Growth,' JPE (1995) — semi-endogenous growth with decelerating progress",
                ],
              },
              {
                eq: "O(N) compute scaling",
                form: "effective_cost = computePer1K × (N / 70B)",
                confidence: "high",
                justification: "Per-step compute for fine-tuning scales linearly with parameter count: each step is a forward + backward pass over all parameters. Well-established. Caveat: memory requirements scale worse than linearly (optimizer states), so total cost may scale slightly super-linearly due to more GPUs and communication overhead.",
                sources: [
                  "Kaplan et al., 'Scaling Laws for Neural Language Models,' arXiv:2001.08361 (2020)",
                  "Standard computational complexity result for neural network forward/backward pass",
                ],
              },
              {
                eq: "Attack deceleration tied to algo halflife",
                form: "attack_rate(t) uses algoHalflife for its decay",
                confidence: "low",
                justification: "Assumes adversarial fine-tuning improvements decelerate on the same timeline as general algorithmic progress, since they draw from the same research community. This is weakly justified — attack research could decelerate faster (small community) or slower (adversarial competition incentives). A separate attackHalflife parameter would be more honest but adds complexity.",
                sources: [
                  "No empirical source — simplifying assumption [STRUCTURAL ASSUMPTION]",
                  "The shared halflife is a known limitation acknowledged in the model",
                ],
              },
            ].map(({ eq, form, confidence, justification, sources }, idx) => {
              const confColor = confidence === "high" ? C.green : confidence === "medium" ? C.yellow : C.red;
              return (
                <div key={idx} style={{
                  background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6,
                  padding: "16px 18px", marginBottom: 12,
                  borderLeft: `3px solid ${confColor}33`,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: C.text }}>{eq}</span>
                    <span style={{
                      fontSize: 9, fontWeight: 700, color: confColor, textTransform: "uppercase",
                      background: confColor + "18", padding: "2px 6px", borderRadius: 3,
                    }}>{confidence}</span>
                  </div>
                  <div style={{
                    fontSize: 11, color: C.algo, fontFamily: "var(--f)",
                    padding: "6px 10px", background: C.bgInner, borderRadius: 4, marginBottom: 8,
                  }}>{form}</div>
                  <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.7, marginBottom: 8 }}>{justification}</div>
                  <div style={{ fontSize: 10, color: C.dim, lineHeight: 1.6, paddingTop: 6, borderTop: `1px solid ${C.border}22` }}>
                    {sources.map((s, i) => (
                      <div key={i} style={{ marginBottom: 2, color: s.includes("[STRUCTURAL") ? C.red : C.dim }}>{s}</div>
                    ))}
                  </div>
                </div>
              );
            })}

            <Section mt={28}>Key Literature Gaps</Section>
            <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20, lineHeight: 1.8, fontSize: 11, color: C.muted }}>
              {[
                { priority: "1", gap: "TR scaling with model size", d: "No empirical data above 8B. A study testing TAR or Deep Ignorance at 1B / 7B / 70B / 405B would be the single highest-value contribution to reducing model uncertainty." },
                { priority: "2", gap: "Attack method improvement rate", d: "No Epoch-quality retrospective analysis exists. A systematic study of fine-tuning efficiency gains (2019-2026) would fill this gap." },
                { priority: "3", gap: "Attack setup cost decomposition", d: "No published work breaks down the non-compute costs of an adversarial fine-tuning attack. Even a hypothetical/anonymized red-team cost analysis would help." },
                { priority: "4", gap: "Attack deceleration independence", d: "The model ties attack method improvement to the same halflife as general algorithmic progress. No evidence supports this coupling; a separate attackHalflife would be more honest." },
              ].map(({ priority, gap, d }, i) => (
                <p key={i} style={{ marginTop: i === 0 ? 0 : 6, marginBottom: 6 }}>
                  <span style={{ color: C.red, fontWeight: 600 }}>#{priority} — {gap}: </span>{d}
                </p>
              ))}
            </div>

            <Section mt={28}>Full Bibliography</Section>
            <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 6, padding: 20, fontSize: 10, color: C.dim, lineHeight: 1.8 }}>
              {[
                "Bloom, Jones, Van Reenen & Webb (2020). 'Are Ideas Getting Harder to Find?' American Economic Review, 110(4).",
                "Cotra (2020). 'Forecasting Transformative AI with Biological Anchors.' Open Philanthropy.",
                "Dettmers et al. (2023). 'QLoRA: Efficient Finetuning of Quantized Language Models.' arXiv:2305.14314.",
                "Erdil & Besiroglu (2022). 'Algorithmic Progress in Computer Vision.' arXiv:2212.05153.",
                "Farmer & Lafond (2016). 'How Predictable is Technological Progress?' Research Policy, 45(3).",
                "Hernandez & Brown (2020). 'Measuring the Algorithmic Efficiency of Neural Networks.' arXiv:2005.04305.",
                "Ho, Besiroglu et al. (2024). 'Algorithmic Progress in Language Models.' arXiv:2403.05812.",
                "Hobbhahn, Besiroglu et al. (2023). 'Trends in Machine Learning Hardware.' Epoch AI.",
                "Hu et al. (2021). 'LoRA: Low-Rank Adaptation of Large Language Models.' arXiv:2106.09685.",
                "Jones (1995). 'R&D-Based Models of Economic Growth.' Journal of Political Economy.",
                "Kaplan et al. (2020). 'Scaling Laws for Neural Language Models.' arXiv:2001.08361.",
                "Li et al. (2024). 'The WMDP Benchmark.' arXiv:2403.03218.",
                "Lynch et al. (2025). 'Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards.' arXiv:2508.06601.",
                "Meta AI (2024). 'The Llama 3 Herd of Models.' arXiv:2407.21783.",
                "Mouton, Chase & Horton (2024). 'The Operational Risks of AI in Large-Scale Biological Attacks.' RAND.",
                "Qi et al. (2023). 'Fine-tuning Aligned Language Models Compromises Safety.' arXiv:2310.03693.",
                "Rosati et al. (2024). 'Representation Noising Effectively Prevents Harmful Fine-tuning.' arXiv:2405.14577.",
                "Schaeffer et al. (2023). 'Are Emergent Abilities a Mirage?' arXiv:2304.15004.",
                "Sevilla et al. (2022). 'Compute Trends Across Three Eras of Machine Learning.' arXiv:2202.05924.",
                "Tamirisa et al. (2024). 'Tamper-Resistant Safeguards for Open-Weight LLMs.' arXiv:2408.00761.",
                "TamperBench (2026). arXiv:2602.06911.",
                "Wei et al. (2022). 'Emergent Abilities of Large Language Models.' TMLR.",
                "Yang et al. (2023). 'Shadow Alignment.' arXiv:2310.02949.",
                "Zhao et al. (2024). 'GaLore.' arXiv:2403.03507.",
              ].map((ref, i) => (
                <div key={i} style={{ marginBottom: 3 }}>[{i + 1}] {ref}</div>
              ))}
            </div>
          </>)}

        </div>
      </div>
    </div>
  );
}
