import { useState, useMemo } from "react";

// ── Utility helpers ──────────────────────────────────────────────
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmt = (n) => {
  if (n >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  if (n >= 1) return `$${n.toFixed(0)}`;
  return `$${n.toFixed(2)}`;
};
const fmtNum = (n) => {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(0);
};
const fmtSteps = (n) => {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return n.toFixed(0);
};

// ── Core model: decaying exponential cost with floor ────────────
function computeCostAtYear(baseCost, initialDecayRate, year, damping, floor) {
  const k0 = Math.log(initialDecayRate);
  const effectiveDamping = Math.max(damping, 0.001);
  const cumulativeDecline = (k0 / effectiveDamping) * (1 - Math.exp(-effectiveDamping * year));
  const cost = floor + (baseCost - floor) * Math.exp(-cumulativeDecline);
  return Math.max(floor, cost);
}

function instantaneousRate(initialDecayRate, year, damping) {
  const k0 = Math.log(initialDecayRate);
  const effectiveDamping = Math.max(damping, 0.001);
  const kt = k0 * Math.exp(-effectiveDamping * year);
  return Math.exp(kt);
}

function capabilityCurve(params, midpoint, steepness = 1.5) {
  const x = Math.log10(params) - Math.log10(midpoint);
  return 1 / (1 + Math.exp(-steepness * x * 5));
}

function safeguardEffectiveness(safeguardBase, attackerBudgetRatio) {
  return safeguardBase * Math.exp(-0.5 * Math.max(0, attackerBudgetRatio - 1));
}

// ── Fine-tuning steps to GPU-hours/cost mapping ─────────────────
// Theoretical calculation:
// - FLOP per step ≈ 6 × params × batch × seq = 6 × params_B × 1e9 × 8 × 2048
// - H100 throughput: ~3e15 FLOP/s
//
// BUT: Real-world training is 10-30x less efficient than theoretical FLOP calc
// due to: memory bandwidth limits, optimizer state overhead, gradient 
// accumulation, multi-GPU communication, data loading, checkpointing.
//
// Empirical calibration from Deep Ignorance paper (EleutherAI 2025):
// - 6.9B model, 10K steps, 305M tokens = ~17 GPU-hours on 2×H200
// - That's ~8.5 GPU-hours per run, or ~3 seconds/step actual
// - Theoretical would predict ~0.23 sec/step — so ~13x efficiency loss
//
// We use a 15x efficiency factor (conservative) to match empirical data.

function stepsToGpuHours(steps, modelSizeB, efficiencyFactor = 15) {
  const flopPerStep = 6 * modelSizeB * 1e9 * 8 * 2048;
  const h100Throughput = 3e15; // FLOP/s theoretical
  const theoreticalSecondsPerStep = flopPerStep / h100Throughput;
  const actualSecondsPerStep = theoreticalSecondsPerStep * efficiencyFactor;
  return (steps * actualSecondsPerStep) / 3600;
}

function gpuHoursToCost(gpuHours, costPerGpuHour) {
  return gpuHours * costPerGpuHour;
}

function stepsToBreakCost(steps, modelSizeB, costPerGpuHour, efficiencyFactor = 15) {
  return gpuHoursToCost(stepsToGpuHours(steps, modelSizeB, efficiencyFactor), costPerGpuHour);
}

// ── Slider components ────────────────────────────────────────────
function Slider({ label, value, onChange, min, max, step, format, description }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
        <label style={{ fontSize: 13, fontFamily: "'IBM Plex Mono', monospace", color: "#c4c8d4", letterSpacing: "0.02em" }}>
          {label}
        </label>
        <span style={{ fontSize: 14, fontFamily: "'IBM Plex Mono', monospace", color: "#e8e0d4", fontWeight: 600 }}>
          {format ? format(value) : value}
        </span>
      </div>
      {description && (
        <div style={{ fontSize: 11, color: "#7a7e8a", marginBottom: 6, lineHeight: 1.4, fontFamily: "'IBM Plex Sans', sans-serif" }}>
          {description}
        </div>
      )}
      <div style={{ position: "relative", height: 24, display: "flex", alignItems: "center" }}>
        <div style={{ position: "absolute", width: "100%", height: 3, background: "#2a2d38", borderRadius: 2, overflow: "hidden" }}>
          <div style={{ width: `${pct}%`, height: "100%", background: "linear-gradient(90deg, #6ECFB0, #4AA88D)", borderRadius: 2, transition: "width 0.1s ease" }} />
        </div>
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          style={{ position: "absolute", width: "100%", height: 24, opacity: 0, cursor: "pointer", margin: 0 }} />
        <div style={{
          position: "absolute", left: `calc(${pct}% - 7px)`, width: 14, height: 14,
          borderRadius: "50%", background: "#e8e0d4", boxShadow: "0 0 8px rgba(110,207,176,0.4)",
          pointerEvents: "none", transition: "left 0.1s ease"
        }} />
      </div>
    </div>
  );
}

// Log-scale slider for steps (100 to 1M)
function LogSlider({ label, value, onChange, minExp, maxExp, format, description }) {
  const logValue = Math.log10(value);
  const pct = ((logValue - minExp) / (maxExp - minExp)) * 100;
  const handleChange = (e) => {
    const newLogValue = parseFloat(e.target.value);
    onChange(Math.pow(10, newLogValue));
  };
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
        <label style={{ fontSize: 13, fontFamily: "'IBM Plex Mono', monospace", color: "#c4c8d4", letterSpacing: "0.02em" }}>
          {label}
        </label>
        <span style={{ fontSize: 14, fontFamily: "'IBM Plex Mono', monospace", color: "#e8e0d4", fontWeight: 600 }}>
          {format ? format(value) : value}
        </span>
      </div>
      {description && (
        <div style={{ fontSize: 11, color: "#7a7e8a", marginBottom: 6, lineHeight: 1.4, fontFamily: "'IBM Plex Sans', sans-serif" }}>
          {description}
        </div>
      )}
      <div style={{ position: "relative", height: 24, display: "flex", alignItems: "center" }}>
        <div style={{ position: "absolute", width: "100%", height: 3, background: "#2a2d38", borderRadius: 2, overflow: "hidden" }}>
          <div style={{ width: `${pct}%`, height: "100%", background: "linear-gradient(90deg, #9B8FFF, #7B6FDF)", borderRadius: 2, transition: "width 0.1s ease" }} />
        </div>
        <input type="range" min={minExp} max={maxExp} step={0.1} value={logValue}
          onChange={handleChange}
          style={{ position: "absolute", width: "100%", height: 24, opacity: 0, cursor: "pointer", margin: 0 }} />
        <div style={{
          position: "absolute", left: `calc(${pct}% - 7px)`, width: 14, height: 14,
          borderRadius: "50%", background: "#e8e0d4", boxShadow: "0 0 8px rgba(155,143,255,0.4)",
          pointerEvents: "none", transition: "left 0.1s ease"
        }} />
      </div>
    </div>
  );
}

// ── Line chart ───────────────────────────────────────────────────
function MiniChart({ data, width = 400, height = 180, colors, legend, yFormat, logScale = false, dashed = [] }) {
  const pad = { top: 24, right: 20, bottom: 36, left: 56 };
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;
  const allY = data.flatMap((s) => s.values);
  const minY = logScale ? Math.max(1, Math.min(...allY.filter(v => v > 0))) : Math.min(...allY, 0);
  const maxY = Math.max(...allY, 1);
  const toX = (i) => pad.left + (i / (data[0].values.length - 1)) * w;
  const toY = (v) => {
    if (logScale) {
      const logMin = Math.log10(Math.max(1, minY));
      const logMax = Math.log10(Math.max(2, maxY));
      return pad.top + h - ((Math.log10(Math.max(1, v)) - logMin) / (logMax - logMin || 1)) * h;
    }
    return pad.top + h - ((v - minY) / (maxY - minY || 1)) * h;
  };
  const gridLines = 4;
  const yTicks = Array.from({ length: gridLines + 1 }, (_, i) => {
    if (logScale) {
      const logMin = Math.log10(Math.max(1, minY));
      const logMax = Math.log10(Math.max(2, maxY));
      return Math.pow(10, logMin + (i / gridLines) * (logMax - logMin));
    }
    return minY + (i / gridLines) * (maxY - minY);
  });

  return (
    <div style={{ position: "relative" }}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {yTicks.map((v, i) => (
          <g key={i}>
            <line x1={pad.left} x2={width - pad.right} y1={toY(v)} y2={toY(v)} stroke="#2a2d38" strokeWidth={1} />
            <text x={pad.left - 8} y={toY(v) + 4} textAnchor="end" fill="#5a5e6a" fontSize={10} fontFamily="'IBM Plex Mono', monospace">
              {yFormat ? yFormat(v) : (v >= 1000 ? `${(v/1000).toFixed(0)}k` : v.toFixed(v < 10 ? 1 : 0))}
            </text>
          </g>
        ))}
        {data.map((series, si) => {
          const path = series.values.map((v, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(" ");
          return (
            <g key={si}>
              <path d={path} fill="none" stroke={colors[si]} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" opacity={0.9} strokeDasharray={dashed[si] ? "6,4" : "none"} />
              <circle cx={toX(series.values.length - 1)} cy={toY(series.values[series.values.length - 1])} r={3} fill={colors[si]} />
            </g>
          );
        })}
        <line x1={pad.left} x2={pad.left} y1={pad.top} y2={pad.top + h} stroke="#3a3d48" strokeWidth={1} />
        <line x1={pad.left} x2={width - pad.right} y1={pad.top + h} y2={pad.top + h} stroke="#3a3d48" strokeWidth={1} />
        {data[0].values.map((_, i) => {
          if (i % 3 === 0 || i === data[0].values.length - 1) {
            return (
              <text key={i} x={toX(i)} y={pad.top + h + 18} textAnchor="middle" fill="#5a5e6a" fontSize={10} fontFamily="'IBM Plex Mono', monospace">
                {i === 0 ? "Now" : `+${i}y`}
              </text>
            );
          }
          return null;
        })}
      </svg>
      {legend && (
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", padding: "4px 0 0 56px" }}>
          {legend.map((l, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <svg width={14} height={4}><line x1={0} y1={2} x2={14} y2={2} stroke={colors[i]} strokeWidth={2} strokeDasharray={dashed[i] ? "4,3" : "none"} /></svg>
              <span style={{ fontSize: 10, color: "#7a7e8a", fontFamily: "'IBM Plex Mono', monospace" }}>{l}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Intervention bars ────────────────────────────────────────────
function InterventionBars({ interventions, width = 400 }) {
  const pad = { top: 8, right: 20, bottom: 4, left: 120 };
  const barH = 22, gap = 10;
  const totalH = interventions.length * (barH + gap) + pad.top + pad.bottom;
  const w = width - pad.left - pad.right;
  return (
    <svg width={width} height={totalH} viewBox={`0 0 ${width} ${totalH}`}>
      {interventions.map((item, i) => {
        const y = pad.top + i * (barH + gap);
        const barW = (item.value / 100) * w;
        return (
          <g key={i} opacity={item.active ? 1 : 0.3}>
            <text x={pad.left - 8} y={y + barH / 2 + 4} textAnchor="end" fill={item.active ? "#c4c8d4" : "#5a5e6a"} fontSize={11} fontFamily="'IBM Plex Mono', monospace">{item.name}</text>
            <rect x={pad.left} y={y} width={w} height={barH} rx={3} fill="#1e2028" />
            <rect x={pad.left} y={y} width={Math.max(0, barW)} height={barH} rx={3} fill={item.color} style={{ transition: "width 0.3s ease" }} />
            <text x={pad.left + Math.max(0, barW) + 6} y={y + barH / 2 + 4} fill={item.active ? "#e8e0d4" : "#5a5e6a"} fontSize={11} fontFamily="'IBM Plex Mono', monospace" fontWeight={600}>{item.value.toFixed(0)}%</text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Threat heatmap ───────────────────────────────────────────────
function ThreatHeatmap({ data, years, profiles, width = 400 }) {
  const maxCols = Math.min(years, 16);
  const step = years > maxCols ? Math.ceil(years / maxCols) : 1;
  const cols = [];
  for (let i = 0; i < years; i += step) cols.push(i);
  const cellW = Math.floor((width - 140) / cols.length);
  const cellH = 36;
  const pad = { left: 140, top: 28 };
  const getColor = (v) => {
    if (v < 0.25) return `rgba(110, 207, 176, ${0.2 + v * 2})`;
    if (v < 0.5) return `rgba(242, 196, 109, ${0.3 + (v - 0.25) * 2})`;
    if (v < 0.75) return `rgba(232, 139, 110, ${0.4 + (v - 0.5) * 1.5})`;
    return `rgba(212, 93, 121, ${0.5 + (v - 0.75) * 2})`;
  };
  return (
    <svg width={width} height={pad.top + profiles.length * cellH + 20} viewBox={`0 0 ${width} ${pad.top + profiles.length * cellH + 20}`}>
      {cols.map((yi, ci) => (
        <text key={ci} x={pad.left + ci * cellW + cellW / 2} y={pad.top - 8} textAnchor="middle" fill="#5a5e6a" fontSize={10} fontFamily="'IBM Plex Mono', monospace">
          {yi === 0 ? "Now" : `+${yi}y`}
        </text>
      ))}
      {profiles.map((profile, pi) => (
        <g key={pi}>
          <text x={pad.left - 8} y={pad.top + pi * cellH + cellH / 2 + 4} textAnchor="end" fill={profile.color} fontSize={11} fontFamily="'IBM Plex Mono', monospace">{profile.name} ({fmtNum(profile.count)})</text>
          {cols.map((yi, ci) => {
            const v = data[pi][yi];
            return (
              <g key={ci}>
                <rect x={pad.left + ci * cellW + 1} y={pad.top + pi * cellH + 1} width={cellW - 2} height={cellH - 2} rx={4} fill={getColor(v)} style={{ transition: "fill 0.3s ease" }} />
                <text x={pad.left + ci * cellW + cellW / 2} y={pad.top + pi * cellH + cellH / 2 + 4} textAnchor="middle" fill="#e8e0d4" fontSize={10} fontFamily="'IBM Plex Mono', monospace" fontWeight={500}>{(v * 100).toFixed(0)}%</text>
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

// ── Main ─────────────────────────────────────────────────────────
export default function ModelSafeguardsDashboard() {
  // Model & compute params
  const [modelSize, setModelSize] = useState(40);
  const [trainingCostBase, setTrainingCostBase] = useState(20);
  const [fineTuneCostBase, setFineTuneCostBase] = useState(0.5);
  const [trainingDecayRate, setTrainingDecayRate] = useState(2.5);
  const [fineTuneDecayRate, setFineTuneDecayRate] = useState(4);
  const [trainingDamping, setTrainingDamping] = useState(0.15);
  const [fineTuneDamping, setFineTuneDamping] = useState(0.12);
  const [trainingFloor, setTrainingFloor] = useState(50);
  const [fineTuneFloor, setFineTuneFloor] = useState(0.01);
  
  // Capability params
  const [dangerousCapThreshold, setDangerousCapThreshold] = useState(10);
  const [noveltyRequiresScale, setNoveltyRequiresScale] = useState(true);
  
  // Attacker populations (tunable)
  const [loneActorCount, setLoneActorCount] = useState(10000);
  const [smallGroupCount, setSmallGroupCount] = useState(100);
  const [fundedOrgCount, setFundedOrgCount] = useState(20);
  const [stateActorCount, setStateActorCount] = useState(15);
  
  // Attacker budgets (tunable)
  const [loneActorBudget, setLoneActorBudget] = useState(1000);
  const [smallGroupBudget, setSmallGroupBudget] = useState(50000);
  const [fundedOrgBudget, setFundedOrgBudget] = useState(2000000);
  const [stateActorBudget, setStateActorBudget] = useState(500000000);
  
  // Safeguard robustness crux
  const [stepsToBreak, setStepsToBreak] = useState(10000);
  const [gpuHourCost, setGpuHourCost] = useState(2);
  const [safeguardBudgetThreshold, setSafeguardBudgetThreshold] = useState(1000);
  
  // Intervention params
  const [safeguardStrength, setSafeguardStrength] = useState(70);
  const [screeningCoverage, setScreeningCoverage] = useState(40);
  const [screeningNovelDetect, setScreeningNovelDetect] = useState(30);
  const [computeGovThreshold, setComputeGovThreshold] = useState(1);
  const [surveillanceEff, setSurveillanceEff] = useState(50);

  const YEARS = 15;

  // Build attacker profiles from tunable params
  const attackerProfiles = useMemo(() => [
    { name: "Lone actor", budget: loneActorBudget, count: loneActorCount, color: "#6ECFB0" },
    { name: "Small group", budget: smallGroupBudget, count: smallGroupCount, color: "#F2C46D" },
    { name: "Well-funded org", budget: fundedOrgBudget, count: fundedOrgCount, color: "#E88B6E" },
    { name: "State actor", budget: stateActorBudget, count: stateActorCount, color: "#D45D79" },
  ], [loneActorBudget, loneActorCount, smallGroupBudget, smallGroupCount, fundedOrgBudget, fundedOrgCount, stateActorBudget, stateActorCount]);

  const results = useMemo(() => {
    const years = Array.from({ length: YEARS + 1 }, (_, i) => i);
    const trainingCosts = years.map(y => computeCostAtYear(trainingCostBase * 1e6, trainingDecayRate, y, trainingDamping, trainingFloor * 1e3));
    const fineTuneCosts = years.map(y => computeCostAtYear(fineTuneCostBase * 1e3, fineTuneDecayRate, y, fineTuneDamping, fineTuneFloor * 1e3));
    const trainingCostsNaive = years.map(y => trainingCostBase * 1e6 * Math.pow(1 / trainingDecayRate, y));
    const fineTuneCostsNaive = years.map(y => fineTuneCostBase * 1e3 * Math.pow(1 / fineTuneDecayRate, y));
    const trainingRates = years.map(y => instantaneousRate(trainingDecayRate, y, trainingDamping));
    const fineTuneRates = years.map(y => instantaneousRate(fineTuneDecayRate, y, fineTuneDamping));
    const modelParams = modelSize * 1e9;
    const dangerousCap = capabilityCurve(modelParams, dangerousCapThreshold * 1e9);
    const novelCap = noveltyRequiresScale ? capabilityCurve(modelParams, dangerousCapThreshold * 1e9 * 2.5) : dangerousCap * 0.7;

    // Safeguard breaking cost
    const breakGpuHours = stepsToGpuHours(stepsToBreak, modelSize);
    const breakCost = gpuHoursToCost(breakGpuHours, gpuHourCost);
    
    // Break cost over time (GPU-hour costs also decline, roughly tracking fine-tune decay)
    const breakCostsOverTime = years.map(y => {
      const futureGpuCost = gpuHourCost * Math.pow(1 / fineTuneDecayRate, y * 0.5); // GPU costs decline slower than fine-tuning
      return gpuHoursToCost(breakGpuHours, Math.max(0.1, futureGpuCost));
    });

    const threatMatrix = attackerProfiles.map(attacker => years.map(y => {
      const trainCost = trainingCosts[y];
      const ftCost = fineTuneCosts[y];
      const canTrain = attacker.budget >= trainCost;
      const canFineTune = attacker.budget >= ftCost;
      if (!canTrain && !canFineTune) return 0;
      let threat = 0;
      if (canTrain) { threat = dangerousCap; }
      else if (canFineTune) {
        const safeguardBlock = safeguardEffectiveness(safeguardStrength / 100, attacker.budget / ftCost);
        threat = dangerousCap * (1 - safeguardBlock);
      }
      const threatNovelty = canTrain ? novelCap : threat * 0.4;
      const effectiveThreat = Math.max(threat * 0.6 + threatNovelty * 0.4, 0);
      let residual = effectiveThreat;
      const knownFrac = 1 - (threatNovelty / Math.max(effectiveThreat, 0.01));
      const screeningBlock = (screeningCoverage / 100) * knownFrac + (screeningCoverage / 100) * (screeningNovelDetect / 100) * (1 - knownFrac);
      residual *= (1 - screeningBlock);
      if (trainCost >= computeGovThreshold * 1e6) { residual *= (1 - (canTrain ? 0.6 : 0.2)); }
      residual *= (1 - surveillanceEff / 100 * 0.5);
      return clamp(residual, 0, 1);
    }));

    // Attacker population analysis
    const totalAttackers = attackerProfiles.reduce((sum, p) => sum + p.count, 0);
    
    // How many attackers can afford to train from scratch at year Y?
    const canTrainByYear = years.map(y => {
      return attackerProfiles.reduce((sum, p) => sum + (p.budget >= trainingCosts[y] ? p.count : 0), 0);
    });
    
    // How many attackers can afford to break safeguards at year Y?
    const canBreakSafeguardsByYear = years.map(y => {
      const breakCostY = breakCostsOverTime[y];
      return attackerProfiles.reduce((sum, p) => sum + (p.budget >= breakCostY ? p.count : 0), 0);
    });
    
    // Safeguards block what % of attackers who could fine-tune but not train?
    const safeguardsBlockByYear = years.map(y => {
      const breakCostY = breakCostsOverTime[y];
      const canFineTuneNotTrain = attackerProfiles.filter(p => p.budget >= fineTuneCosts[y] && p.budget < trainingCosts[y]);
      const blockedCount = canFineTuneNotTrain.filter(p => p.budget < breakCostY).reduce((sum, p) => sum + p.count, 0);
      const totalInRange = canFineTuneNotTrain.reduce((sum, p) => sum + p.count, 0);
      return totalInRange > 0 ? blockedCount / totalInRange : 1;
    });

    const yearIdx = 5;
    const interventionValues = [
      { name: "Model safeguards", value: clamp((safeguardStrength / 100) * dangerousCap * 30 * safeguardsBlockByYear[yearIdx], 0, 100), color: "#6ECFB0", active: safeguardStrength > 10 },
      { name: "Synthesis screening", value: clamp(screeningCoverage * 0.7 + screeningNovelDetect * 0.15, 0, 100), color: "#4AA88D", active: screeningCoverage > 10 },
      { name: "Compute governance", value: clamp(trainingCosts[yearIdx] >= computeGovThreshold * 1e6 ? 45 * (1 - Math.min(1, computeGovThreshold * 1e6 / trainingCosts[yearIdx])) : 5, 0, 100), color: "#F2C46D", active: trainingCosts[yearIdx] >= computeGovThreshold * 1e6 * 0.5 },
      { name: "Surveillance", value: clamp(surveillanceEff * 0.6, 0, 100), color: "#E88B6E", active: surveillanceEff > 10 },
    ];

    const naiveTrainYear = trainingCostsNaive.findIndex(c => c <= smallGroupBudget);
    const realTrainYear = trainingCosts.findIndex(c => c <= smallGroupBudget);
    const floorBuysYears = (realTrainYear < 0 ? YEARS : realTrainYear) - (naiveTrainYear < 0 ? YEARS : naiveTrainYear);

    // Safeguard relevance decision framework
    // Evaluates whether model safeguards are worth investing in based on:
    // 1. What % of fine-tune-capable attackers are blocked at year 5?
    // 2. How many years until training-from-scratch is accessible to same population?
    // 3. Is the breaking cost meaningful relative to attacker budgets?
    
    const blockedPct5 = safeguardsBlockByYear[5] * 100;
    
    // Years until small groups can train from scratch (the "window")
    const windowYears = realTrainYear < 0 ? YEARS : realTrainYear;
    
    // Breaking cost as % of median relevant attacker budget (small groups)
    const breakCostPct = (breakCost / smallGroupBudget) * 100;
    
    // Compute relevance score (0-100)
    let relevanceScore = 0;
    let relevanceReason = "";
    
    if (blockedPct5 < 20) {
      relevanceScore = Math.max(0, blockedPct5);
      relevanceReason = `Only ${blockedPct5.toFixed(0)}% of attackers blocked — safeguards ineffective at current robustness`;
    } else if (windowYears < 3) {
      relevanceScore = Math.min(30, blockedPct5 * 0.5);
      relevanceReason = `Training accessible to small groups in ${windowYears}y — safeguards obsolete before they matter`;
    } else if (breakCostPct < 1) {
      relevanceScore = Math.min(40, blockedPct5 * 0.6);
      relevanceReason = `Breaking cost is ${breakCostPct.toFixed(1)}% of attacker budget — trivial barrier`;
    } else if (blockedPct5 >= 50 && windowYears >= 5 && breakCostPct >= 10) {
      relevanceScore = Math.min(100, blockedPct5 + windowYears * 2 + breakCostPct * 0.5);
      relevanceReason = `${blockedPct5.toFixed(0)}% blocked, ${windowYears}y window, ${breakCostPct.toFixed(0)}% of budget — safeguards have value`;
    } else {
      relevanceScore = Math.min(70, blockedPct5 * 0.7 + windowYears * 2);
      relevanceReason = `Marginal value: ${blockedPct5.toFixed(0)}% blocked, ${windowYears}y window`;
    }
    
    const relevanceLevel = relevanceScore >= 60 ? "high" : relevanceScore >= 30 ? "marginal" : "low";

    return { 
      years, trainingCosts, fineTuneCosts, trainingCostsNaive, fineTuneCostsNaive, 
      trainingRates, fineTuneRates, threatMatrix, interventionValues, 
      dangerousCap, novelCap, floorBuysYears,
      breakGpuHours, breakCost, breakCostsOverTime,
      totalAttackers, canTrainByYear, canBreakSafeguardsByYear, safeguardsBlockByYear,
      relevanceScore, relevanceReason, relevanceLevel, blockedPct5, windowYears, breakCostPct
    };
  }, [modelSize, trainingCostBase, fineTuneCostBase, trainingDecayRate, fineTuneDecayRate, 
      trainingDamping, fineTuneDamping, trainingFloor, fineTuneFloor, 
      dangerousCapThreshold, noveltyRequiresScale, 
      safeguardStrength, screeningCoverage, screeningNovelDetect, computeGovThreshold, surveillanceEff,
      attackerProfiles, stepsToBreak, gpuHourCost]);

  const [activeTab, setActiveTab] = useState("cruxes");
  const tabs = [
    { id: "cruxes", label: "Key cruxes" },
    { id: "costs", label: "Cost curves" },
    { id: "limits", label: "Physical limits" },
    { id: "threat", label: "Threat matrix" },
    { id: "value", label: "Intervention value" },
    { id: "model", label: "Model details" },
  ];

  const panelStyle = { background: "#1a1c24", borderRadius: 10, padding: 20, border: "1px solid #2a2d38" };
  const sectionLabelStyle = (color) => ({ fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 16 });
  const insightStyle = (color) => ({ marginTop: 16, padding: "10px 14px", borderLeft: `2px solid ${color}`, background: `${color}10`, borderRadius: "0 6px 6px 0" });
  const insightText = { fontSize: 12, color: "#c4c8d4", lineHeight: 1.5 };
  const statBox = { padding: "10px 14px", background: "#14161c", borderRadius: 6 };
  const statLabel = { fontSize: 10, color: "#7a7e8a", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 4 };
  const statValue = (color = "#e8e0d4") => ({ fontSize: 18, fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace", color });

  return (
    <div style={{ minHeight: "100vh", background: "#14161c", color: "#e8e0d4", fontFamily: "'IBM Plex Sans', sans-serif", padding: "24px 16px" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; }
        input[type="range"] { -webkit-appearance: none; appearance: none; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 14px; height: 14px; }
      `}</style>

      <div style={{ maxWidth: 980, margin: "0 auto" }}>
        <div style={{ marginBottom: 8 }}>
          <span style={{ fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", color: "#6ECFB0", letterSpacing: "0.15em", textTransform: "uppercase" }}>
            Biosecurity × AI Risk Explorer v3
          </span>
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 300, margin: "0 0 6px 0", fontFamily: "'Space Mono', monospace", color: "#e8e0d4", lineHeight: 1.2, letterSpacing: "-0.02em" }}>
          When do model safeguards stop mattering?
        </h1>
        <p style={{ fontSize: 14, color: "#7a7e8a", margin: "0 0 32px 0", lineHeight: 1.5, maxWidth: 700 }}>
          Track two key cruxes: (1) How many actors can train without safeguards? (2) How robust must safeguards be to matter?
          Now with tunable attacker populations and fine-tuning cost breakdowns.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: 32, alignItems: "start" }}>
          {/* ── Controls ── */}
          <div>
            <div style={{ ...panelStyle, marginBottom: 16 }}>
              <div style={sectionLabelStyle("#6ECFB0")}>Model & Compute</div>
              <Slider label="Target model size" value={modelSize} onChange={setModelSize} min={1} max={200} step={1} format={v => `${v}B params`} description="Biological foundation model size (Evo2 ≈ 40B)" />
              <Slider label="Training cost (today)" value={trainingCostBase} onChange={setTrainingCostBase} min={0.1} max={200} step={0.1} format={v => fmt(v * 1e6)} description="Current cost to train from scratch" />
              <Slider label="Fine-tune cost (today)" value={fineTuneCostBase} onChange={setFineTuneCostBase} min={0.01} max={50} step={0.01} format={v => fmt(v * 1e3)} description="Current LoRA fine-tuning cost" />
              <Slider label="Initial training decay" value={trainingDecayRate} onChange={setTrainingDecayRate} min={1.2} max={10} step={0.1} format={v => `${v.toFixed(1)}×/yr`} description="Starting annual cost reduction" />
              <Slider label="Initial fine-tune decay" value={fineTuneDecayRate} onChange={setFineTuneDecayRate} min={1.5} max={15} step={0.1} format={v => `${v.toFixed(1)}×/yr`} description="Starting annual fine-tuning cost reduction" />
            </div>

            <div style={{ ...panelStyle, marginBottom: 16 }}>
              <div style={sectionLabelStyle("#9B8FFF")}>Safeguard Robustness Crux</div>
              <LogSlider label="Steps to break safeguards" value={stepsToBreak} onChange={setStepsToBreak} minExp={2} maxExp={6} format={v => `${fmtSteps(v)} steps`} description="Fine-tuning steps needed to undo safeguards. Basic RLHF: ~1K. Deep Ignorance (SOTA): ~10K." />
              <Slider label="GPU-hour cost" value={gpuHourCost} onChange={setGpuHourCost} min={0.1} max={10} step={0.1} format={v => `$${v.toFixed(2)}/hr`} description="Current H100 spot pricing" />
              <div style={{ marginTop: 8, padding: "10px 12px", background: "#14161c", borderRadius: 6 }}>
                <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Breaking safeguards costs:</div>
                <div style={{ display: "flex", gap: 16, alignItems: "baseline" }}>
                  <span style={{ fontSize: 16, fontWeight: 600, color: "#9B8FFF", fontFamily: "'IBM Plex Mono', monospace" }}>{fmt(results.breakCost)}</span>
                  <span style={{ fontSize: 11, color: "#7a7e8a" }}>({results.breakGpuHours.toFixed(1)} GPU-hrs)</span>
                </div>
              </div>
            </div>

            <div style={{ ...panelStyle, marginBottom: 16 }}>
              <div style={sectionLabelStyle("#D45D79")}>Attacker Populations</div>
              <Slider label="Lone actors" value={loneActorCount} onChange={setLoneActorCount} min={100} max={100000} step={100} format={v => fmtNum(v)} description={`Budget: ${fmt(loneActorBudget)} each`} />
              <Slider label="Small groups" value={smallGroupCount} onChange={setSmallGroupCount} min={1} max={1000} step={1} format={v => fmtNum(v)} description={`Budget: ${fmt(smallGroupBudget)} each`} />
              <Slider label="Well-funded orgs" value={fundedOrgCount} onChange={setFundedOrgCount} min={1} max={200} step={1} format={v => fmtNum(v)} description={`Budget: ${fmt(fundedOrgBudget)} each`} />
              <Slider label="State actors" value={stateActorCount} onChange={setStateActorCount} min={1} max={50} step={1} format={v => fmtNum(v)} description={`Budget: ${fmt(stateActorBudget)} each`} />
              <details style={{ marginTop: 8 }}>
                <summary style={{ fontSize: 11, color: "#7a7e8a", cursor: "pointer" }}>Adjust budgets</summary>
                <div style={{ marginTop: 12 }}>
                  <LogSlider label="Lone actor budget" value={loneActorBudget} onChange={setLoneActorBudget} minExp={2} maxExp={5} format={v => fmt(v)} />
                  <LogSlider label="Small group budget" value={smallGroupBudget} onChange={setSmallGroupBudget} minExp={3} maxExp={6} format={v => fmt(v)} />
                  <LogSlider label="Well-funded org budget" value={fundedOrgBudget} onChange={setFundedOrgBudget} minExp={5} maxExp={8} format={v => fmt(v)} />
                  <LogSlider label="State actor budget" value={stateActorBudget} onChange={setStateActorBudget} minExp={7} maxExp={10} format={v => fmt(v)} />
                </div>
              </details>
            </div>

            <div style={{ ...panelStyle, marginBottom: 16 }}>
              <div style={sectionLabelStyle("#F2C46D")}>Physical Limits</div>
              <Slider label="Training damping" value={trainingDamping} onChange={setTrainingDamping} min={0} max={0.5} step={0.01} format={v => v < 0.01 ? "None" : v.toFixed(2)} description="How fast training decay rate slows" />
              <Slider label="Training floor" value={trainingFloor} onChange={setTrainingFloor} min={1} max={5000} step={1} format={v => fmt(v * 1e3)} description="Irreducible minimum cost" />
            </div>

            <div style={panelStyle}>
              <div style={sectionLabelStyle("#E88B6E")}>Interventions</div>
              <Slider label="Safeguard strength" value={safeguardStrength} onChange={setSafeguardStrength} min={0} max={100} step={1} format={v => `${v}%`} description="Resistance to fine-tuning attacks" />
              <Slider label="Synthesis screening" value={screeningCoverage} onChange={setScreeningCoverage} min={0} max={100} step={1} format={v => `${v}%`} description="Provider coverage" />
              <Slider label="Novel detection" value={screeningNovelDetect} onChange={setScreeningNovelDetect} min={0} max={100} step={1} format={v => `${v}%`} description="Catch rate for AI-designed sequences" />
              <Slider label="Surveillance" value={surveillanceEff} onChange={setSurveillanceEff} min={0} max={100} step={1} format={v => `${v}%`} description="Metagenomic surveillance effectiveness" />
            </div>
          </div>

          {/* ── Visualizations ── */}
          <div>
            <div style={{ display: "flex", gap: 2, marginBottom: 20, flexWrap: "wrap" }}>
              {tabs.map(tab => (
                <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
                  padding: "8px 14px", border: "none", cursor: "pointer", borderRadius: 6,
                  background: activeTab === tab.id ? "#2a2d38" : "transparent",
                  color: activeTab === tab.id ? "#e8e0d4" : "#5a5e6a",
                  fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", letterSpacing: "0.03em", transition: "all 0.2s ease"
                }}>{tab.label}</button>
              ))}
            </div>

            {/* ── Cruxes tab ── */}
            {activeTab === "cruxes" && (
              <div>
                {/* Crux 1: Training from scratch */}
                <div style={{ ...panelStyle, marginBottom: 16 }}>
                  <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 12, fontFamily: "'Space Mono', monospace", color: "#D45D79" }}>
                    Crux 1: Who can train a {modelSize}B model without safeguards?
                  </div>
                  
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 16 }}>
                    <div style={statBox}>
                      <div style={statLabel}>Today</div>
                      <div style={statValue()}>{fmt(results.trainingCosts[0])}</div>
                      <div style={{ fontSize: 10, color: "#5a5e6a" }}>{fmtNum(results.canTrainByYear[0])} can afford</div>
                    </div>
                    <div style={statBox}>
                      <div style={statLabel}>Year 5</div>
                      <div style={statValue(results.canTrainByYear[5] > 100 ? "#D45D79" : "#F2C46D")}>{fmt(results.trainingCosts[5])}</div>
                      <div style={{ fontSize: 10, color: "#5a5e6a" }}>{fmtNum(results.canTrainByYear[5])} can afford</div>
                    </div>
                    <div style={statBox}>
                      <div style={statLabel}>Year 10</div>
                      <div style={statValue(results.canTrainByYear[10] > 100 ? "#D45D79" : "#F2C46D")}>{fmt(results.trainingCosts[10])}</div>
                      <div style={{ fontSize: 10, color: "#5a5e6a" }}>{fmtNum(results.canTrainByYear[10])} can afford</div>
                    </div>
                    <div style={statBox}>
                      <div style={statLabel}>Year 15</div>
                      <div style={statValue(results.canTrainByYear[15] > 1000 ? "#D45D79" : "#F2C46D")}>{fmt(results.trainingCosts[Math.min(15, YEARS)])}</div>
                      <div style={{ fontSize: 10, color: "#5a5e6a" }}>{fmtNum(results.canTrainByYear[Math.min(15, YEARS)])} can afford</div>
                    </div>
                  </div>

                  <div style={{ marginBottom: 12 }}>
                    {attackerProfiles.map(p => {
                      const yearAfford = results.trainingCosts.findIndex(c => p.budget >= c);
                      return (
                        <div key={p.name} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                          <div style={{ width: 8, height: 8, borderRadius: "50%", background: p.color }} />
                          <span style={{ fontSize: 12, color: p.color, fontFamily: "'IBM Plex Mono', monospace", width: 140 }}>{p.name} ({fmtNum(p.count)})</span>
                          <span style={{ fontSize: 12, color: "#e8e0d4" }}>
                            {yearAfford < 0 ? `Can't afford in ${YEARS}y` : yearAfford === 0 ? "Can afford now" : `Can afford in +${yearAfford}y`}
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  <div style={insightStyle("#D45D79")}>
                    <div style={insightText}>
                      {results.canTrainByYear[5] > 100
                        ? `⚠ By year 5, ${fmtNum(results.canTrainByYear[5])} actors can train from scratch at ${fmt(results.trainingCosts[5])}, completely bypassing model safeguards.`
                        : results.canTrainByYear[10] > 100
                          ? `Training remains expensive through year 5 (${fmt(results.trainingCosts[5])}, only ${fmtNum(results.canTrainByYear[5])} can afford), but ${fmtNum(results.canTrainByYear[10])} can by year 10 at ${fmt(results.trainingCosts[10])}.`
                          : `Training remains prohibitive through year 10 at ${fmt(results.trainingCosts[10])}. Model safeguards have extended relevance.`}
                    </div>
                  </div>
                </div>

                {/* Crux 2: Safeguard robustness */}
                <div style={panelStyle}>
                  <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 12, fontFamily: "'Space Mono', monospace", color: "#9B8FFF" }}>
                    Crux 2: How many fine-tuning steps to break safeguards?
                  </div>
                  
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 16 }}>
                    <div style={statBox}>
                      <div style={statLabel}>Steps to break</div>
                      <div style={statValue("#9B8FFF")}>{fmtSteps(stepsToBreak)}</div>
                    </div>
                    <div style={statBox}>
                      <div style={statLabel}>GPU-hours</div>
                      <div style={statValue()}>{results.breakGpuHours.toFixed(1)}</div>
                    </div>
                    <div style={statBox}>
                      <div style={statLabel}>Cost today</div>
                      <div style={statValue()}>{fmt(results.breakCost)}</div>
                    </div>
                  </div>

                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 11, color: "#7a7e8a", marginBottom: 8 }}>SAFEGUARD BENCHMARKS (cost for {modelSize}B model)</div>
                    {[
                      { name: "Basic RLHF", steps: 1000, color: "#D45D79", note: "broken in minutes" },
                      { name: "Constitutional AI", steps: 5000, color: "#E88B6E", note: "" },
                      { name: "Deep Ignorance (SOTA)", steps: 10000, color: "#F2C46D", note: "2025" },
                    ].map(bench => (
                      <div key={bench.name} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                        <div style={{ width: 130 }}>
                          <span style={{ fontSize: 11, color: bench.color }}>{bench.name}</span>
                          {bench.note && <span style={{ fontSize: 9, color: "#5a5e6a", marginLeft: 4 }}>{bench.note}</span>}
                        </div>
                        <div style={{ flex: 1, height: 6, background: "#2a2d38", borderRadius: 3, position: "relative" }}>
                          <div style={{
                            position: "absolute", height: "100%", borderRadius: 3,
                            width: `${Math.min(100, Math.log10(bench.steps) / 6 * 100)}%`,
                            background: bench.color, opacity: 0.6
                          }} />
                        </div>
                        <span style={{ fontSize: 10, color: "#7a7e8a", width: 40, textAlign: "right" }}>{fmtSteps(bench.steps)}</span>
                        <span style={{ fontSize: 10, color: bench.color, width: 55, textAlign: "right", fontWeight: 600 }}>{fmt(stepsToBreakCost(bench.steps, modelSize, gpuHourCost))}</span>
                      </div>
                    ))}
                    {/* Your method slider visualization */}
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8, marginBottom: 4, padding: "6px 0", borderTop: "1px solid #2a2d38" }}>
                      <div style={{ width: 130 }}>
                        <span style={{ fontSize: 11, color: "#6ECFB0", fontWeight: 600 }}>Your method</span>
                        <span style={{ fontSize: 9, color: "#5a5e6a", marginLeft: 4 }}>↑ adjust above</span>
                      </div>
                      <div style={{ flex: 1, height: 6, background: "#2a2d38", borderRadius: 3, position: "relative" }}>
                        <div style={{
                          position: "absolute", height: "100%", borderRadius: 3,
                          width: `${Math.min(100, Math.log10(stepsToBreak) / 6 * 100)}%`,
                          background: "#6ECFB0", opacity: 0.8
                        }} />
                      </div>
                      <span style={{ fontSize: 10, color: "#e8e0d4", width: 40, textAlign: "right", fontWeight: 600 }}>{fmtSteps(stepsToBreak)}</span>
                      <span style={{ fontSize: 10, color: "#6ECFB0", width: 55, textAlign: "right", fontWeight: 600 }}>{fmt(results.breakCost)}</span>
                    </div>
                    <div style={{ marginTop: 10, padding: "8px 10px", background: "#D45D7915", borderRadius: 4, fontSize: 11, color: "#c4c8d4", lineHeight: 1.4 }}>
                      <strong style={{ color: "#D45D79" }}>Context:</strong> Deep Ignorance (SOTA, 2025) resists 10K steps — costs <strong>{fmt(stepsToBreakCost(10000, modelSize, gpuHourCost))}</strong> to break.
                      Prior methods fell to "a few dozen steps." Adjust the slider above to model hypothetical improved methods.
                    </div>
                  </div>

                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 11, color: "#7a7e8a", marginBottom: 8 }}>WHO CAN AFFORD TO BREAK SAFEGUARDS?</div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                      {["Today", "Year 5", "Year 10", "Year 15"].map((label, i) => {
                        const yr = [0, 5, 10, 15][i];
                        const pct = results.safeguardsBlockByYear[yr] * 100;
                        return (
                          <div key={label} style={statBox}>
                            <div style={statLabel}>{label}</div>
                            <div style={statValue(pct > 50 ? "#6ECFB0" : pct > 20 ? "#F2C46D" : "#D45D79")}>{pct.toFixed(0)}%</div>
                            <div style={{ fontSize: 10, color: "#5a5e6a" }}>blocked</div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div style={insightStyle("#9B8FFF")}>
                    <div style={insightText}>
                      {stepsToBreak < 5000
                        ? `⚠ At ${fmtSteps(stepsToBreak)} steps, safeguards cost only ${fmt(results.breakCost)} to break — ineffective against all but lone actors.`
                        : stepsToBreak < 50000
                          ? `Current robustness (${fmtSteps(stepsToBreak)} steps, ${fmt(results.breakCost)}) blocks small groups but not funded orgs. Need 10× more steps to matter.`
                          : `Strong robustness: ${fmtSteps(stepsToBreak)} steps = ${fmt(results.breakCost)}. This blocks ${results.safeguardsBlockByYear[5].toFixed(0) * 100}% of fine-tune-capable attackers at year 5.`}
                    </div>
                  </div>

                  <div style={{ marginTop: 16, padding: "12px 14px", background: "#14161c", borderRadius: 8, borderLeft: "3px solid #6ECFB0" }}>
                    <div style={{ fontSize: 11, color: "#6ECFB0", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 6 }}>REQUIRED SAFEGUARD TARGET</div>
                    <div style={{ fontSize: 12, color: "#c4c8d4", lineHeight: 1.5 }}>
                      {(() => {
                        // Reverse the cost calculation: given budget, how many steps?
                        // cost = gpuHours * gpuHourCost
                        // gpuHours = steps * (6 * modelSize * 1e9 * 8 * 2048 / 3e15) * efficiencyFactor / 3600
                        const efficiencyFactor = 15;
                        const theoreticalSecsPerStep = (6 * modelSize * 1e9 * 8 * 2048) / 3e15;
                        const actualSecsPerStep = theoreticalSecsPerStep * efficiencyFactor;
                        const gpuHoursForBudget = safeguardBudgetThreshold / gpuHourCost;
                        const stepsForBudget = (gpuHoursForBudget * 3600) / actualSecsPerStep;
                        return <>
                          To block attackers with budget &lt; <strong style={{ color: "#e8e0d4" }}>{fmt(safeguardBudgetThreshold)}</strong>, 
                          safeguards need ≥ <strong style={{ color: "#e8e0d4" }}>{fmtSteps(stepsForBudget)} steps</strong> ({gpuHoursForBudget.toFixed(1)} GPU-hours).
                        </>;
                      })()}
                      <br />
                      <span style={{ fontSize: 11, color: "#7a7e8a" }}>Adjust threshold slider to explore different budget targets.</span>
                    </div>
                  </div>

                  {/* Safeguard Relevance Factors */}
                  <div style={{ 
                    marginTop: 16, 
                    padding: "14px 16px", 
                    background: "#1a1c24",
                    borderRadius: 8, 
                    border: "1px solid #2a2d38"
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                      <div style={{ fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color: "#7a7e8a", letterSpacing: "0.05em" }}>
                        FACTORS FOR SAFEGUARD INVESTMENT DECISIONS
                      </div>
                    </div>
                    
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 14 }}>
                      <div style={{ padding: "10px 12px", background: "#14161c", borderRadius: 6 }}>
                        <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Attackers blocked (yr 5)</div>
                        <div style={{ fontSize: 18, fontWeight: 600, color: results.blockedPct5 >= 50 ? "#6ECFB0" : results.blockedPct5 >= 20 ? "#F2C46D" : "#D45D79", fontFamily: "'IBM Plex Mono', monospace" }}>
                          {results.blockedPct5.toFixed(0)}%
                        </div>
                        <div style={{ fontSize: 9, color: "#5a5e6a", marginTop: 2 }}>of fine-tune-capable actors</div>
                      </div>
                      <div style={{ padding: "10px 12px", background: "#14161c", borderRadius: 6 }}>
                        <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Relevance window</div>
                        <div style={{ fontSize: 18, fontWeight: 600, color: results.windowYears >= 5 ? "#6ECFB0" : results.windowYears >= 3 ? "#F2C46D" : "#D45D79", fontFamily: "'IBM Plex Mono', monospace" }}>
                          {results.windowYears}y
                        </div>
                        <div style={{ fontSize: 9, color: "#5a5e6a", marginTop: 2 }}>until training-from-scratch accessible</div>
                      </div>
                      <div style={{ padding: "10px 12px", background: "#14161c", borderRadius: 6 }}>
                        <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Break cost burden</div>
                        <div style={{ fontSize: 18, fontWeight: 600, color: results.breakCostPct >= 10 ? "#6ECFB0" : results.breakCostPct >= 1 ? "#F2C46D" : "#D45D79", fontFamily: "'IBM Plex Mono', monospace" }}>
                          {results.breakCostPct.toFixed(1)}%
                        </div>
                        <div style={{ fontSize: 9, color: "#5a5e6a", marginTop: 2 }}>of small group budget ({fmt(smallGroupBudget)})</div>
                      </div>
                    </div>
                    
                    <div style={{ fontSize: 12, color: "#c4c8d4", lineHeight: 1.5, marginBottom: 12 }}>
                      <strong style={{ color: "#e8e0d4" }}>Current assessment:</strong> {results.relevanceReason}
                    </div>

                    {/* Key assumptions callout */}
                    <div style={{ padding: "10px 12px", background: "#D45D7910", borderRadius: 6, borderLeft: "2px solid #D45D79" }}>
                      <div style={{ fontSize: 10, color: "#D45D79", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 6 }}>⚠ KEY ASSUMPTIONS IN THIS ANALYSIS</div>
                      <div style={{ fontSize: 11, color: "#9a9eb0", lineHeight: 1.5 }}>
                        <strong style={{ color: "#c4c8d4" }}>1. Safeguards can be undone by fine-tuning.</strong> If a technique can be broken with fine-tuning, 
                        the cost to do so is determined by steps required × compute cost per step. At current settings, that's {fmt(results.breakCost)} today, 
                        declining to {fmt(results.breakCostsOverTime[5])} by year 5.
                        <br /><br />
                        <strong style={{ color: "#c4c8d4" }}>2. Opportunity cost is unspecified.</strong> "Worth investing" compared to what? 
                        The same R&D effort on synthesis screening or surveillance might have higher expected value. 
                        This tool doesn't model portfolio allocation across interventions.
                        <br /><br />
                        <strong style={{ color: "#c4c8d4" }}>3. Attacker populations are illustrative.</strong> Real threat modeling requires 
                        better data on who actually has intent + resources. See GovAI threat modeling work for more rigorous estimates.
                      </div>
                    </div>
                    
                    <div style={{ marginTop: 12, fontSize: 10, color: "#5a5e6a", lineHeight: 1.4, fontStyle: "italic" }}>
                      These factors inform but don't determine investment decisions. Consider also: portfolio effects with other interventions, 
                      R&D costs, probability of achieving target robustness, and whether safeguards provide value beyond direct blocking (e.g., signaling, norm-setting).
                    </div>
                  </div>
                </div>

                {/* Chokepoint Comparison */}
                <div style={{ ...panelStyle, marginTop: 16 }}>
                  <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 12, fontFamily: "'Space Mono', monospace", color: "#E88B6E" }}>
                    Chokepoint Comparison: Where's the weakest link?
                  </div>
                  
                  <div style={{ fontSize: 12, color: "#9a9eb0", marginBottom: 16, lineHeight: 1.5 }}>
                    Compare the cost to bypass model safeguards against other chokepoints in the bioweapon development chain. 
                    If one barrier is orders of magnitude cheaper to bypass, it's the weak link.
                  </div>

                  {/* Chokepoint visualization */}
                  <div style={{ marginBottom: 16 }}>
                    {[
                      { name: "Break model safeguards", cost: results.breakCost, color: "#9B8FFF", note: `${fmtSteps(stepsToBreak)} steps, your settings`, dynamic: true },
                      { name: "Train model from scratch", cost: results.trainingCosts[0], color: "#D45D79", note: `${modelSize}B model today`, dynamic: true },
                      { name: "Unscreened DNA synthesis", cost: 100000, color: "#F2C46D", note: "~$0.20/bp × 500K bp genome" },
                      { name: "Benchtop DNA printer", cost: 100000, color: "#F2C46D", note: "Equipment + consumables" },
                      { name: "Basic BSL-2 lab setup", cost: 100000, color: "#6ECFB0", note: "Space, equipment, supplies" },
                      { name: "BSL-3 lab setup", cost: 1000000, color: "#6ECFB0", note: "Containment infrastructure" },
                      { name: "Hire domain expert (1 yr)", cost: 250000, color: "#4AA88D", note: "Salary + recruiting" },
                    ].map((item, i) => {
                      const maxCost = 2000000; // $2M max for visualization
                      const logPct = Math.max(5, (Math.log10(Math.max(1, item.cost)) / Math.log10(maxCost)) * 100);
                      const isWeakest = item.cost <= results.breakCost && item.dynamic;
                      return (
                        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                          <div style={{ width: 160, flexShrink: 0 }}>
                            <div style={{ fontSize: 11, color: item.color, fontWeight: item.dynamic ? 600 : 400 }}>{item.name}</div>
                            <div style={{ fontSize: 9, color: "#5a5e6a" }}>{item.note}</div>
                          </div>
                          <div style={{ flex: 1, height: 8, background: "#2a2d38", borderRadius: 4, position: "relative", overflow: "hidden" }}>
                            <div style={{
                              position: "absolute", height: "100%", borderRadius: 4,
                              width: `${Math.min(100, logPct)}%`,
                              background: item.color,
                              opacity: item.dynamic ? 0.9 : 0.5
                            }} />
                          </div>
                          <div style={{ width: 70, textAlign: "right" }}>
                            <span style={{ fontSize: 12, color: item.dynamic ? "#e8e0d4" : "#7a7e8a", fontFamily: "'IBM Plex Mono', monospace", fontWeight: item.dynamic ? 600 : 400 }}>
                              {fmt(item.cost)}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Ratio analysis */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
                    <div style={{ padding: "10px 12px", background: "#14161c", borderRadius: 6 }}>
                      <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Safeguard bypass vs. lab setup</div>
                      <div style={{ fontSize: 20, fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace", color: results.breakCost < 10000 ? "#D45D79" : results.breakCost < 50000 ? "#F2C46D" : "#6ECFB0" }}>
                        {(100000 / results.breakCost).toFixed(0)}×
                      </div>
                      <div style={{ fontSize: 10, color: "#5a5e6a" }}>cheaper to break safeguards</div>
                    </div>
                    <div style={{ padding: "10px 12px", background: "#14161c", borderRadius: 6 }}>
                      <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Safeguard bypass vs. train from scratch</div>
                      <div style={{ fontSize: 20, fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace", color: results.breakCost < results.trainingCosts[0] * 0.01 ? "#D45D79" : results.breakCost < results.trainingCosts[0] * 0.1 ? "#F2C46D" : "#6ECFB0" }}>
                        {(results.trainingCosts[0] / results.breakCost).toFixed(0)}×
                      </div>
                      <div style={{ fontSize: 10, color: "#5a5e6a" }}>cheaper to break safeguards</div>
                    </div>
                  </div>

                  <div style={{ padding: "10px 14px", borderLeft: "2px solid #E88B6E", background: "#E88B6E10", borderRadius: "0 6px 6px 0" }}>
                    <div style={{ fontSize: 12, color: "#c4c8d4", lineHeight: 1.5 }}>
                      {results.breakCost < 1000 
                        ? `⚠ At ${fmt(results.breakCost)}, breaking safeguards is ~${(100000 / results.breakCost).toFixed(0)}× cheaper than physical infrastructure. Safeguards are the weakest link by far.`
                        : results.breakCost < 10000
                          ? `Breaking safeguards (${fmt(results.breakCost)}) is still ~${(100000 / results.breakCost).toFixed(0)}× cheaper than lab setup. Not the binding constraint on serious attackers.`
                          : results.breakCost < 100000
                            ? `At ${fmt(results.breakCost)}, safeguard costs approach lab setup costs. Starting to be a meaningful barrier.`
                            : `At ${fmt(results.breakCost)}, breaking safeguards rivals physical infrastructure costs. Safeguards are a real chokepoint.`}
                    </div>
                  </div>
                  
                  <div style={{ marginTop: 12, fontSize: 10, color: "#5a5e6a", lineHeight: 1.4 }}>
                    Physical infrastructure costs are rough estimates and vary by region/context. DNA synthesis costs declining ~15-20%/year.
                    The key insight: if one chokepoint is 100-1000× cheaper to bypass, it won't be the binding constraint.
                  </div>

                  {/* Novel vs known pathogens distinction */}
                  <div style={{ marginTop: 16, padding: "12px 14px", background: "#6ECFB010", borderRadius: 8, border: "1px solid #6ECFB030" }}>
                    <div style={{ fontSize: 11, color: "#6ECFB0", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 8 }}>
                      IMPORTANT CAVEAT: NOVEL VS. KNOWN PATHOGENS
                    </div>
                    <div style={{ fontSize: 12, color: "#c4c8d4", lineHeight: 1.6 }}>
                      <p style={{ margin: "0 0 10px 0" }}>
                        The chokepoint comparison above assumes all paths lead to equivalent harm. But they may not:
                      </p>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                        <div style={{ padding: "8px 10px", background: "#14161c", borderRadius: 6 }}>
                          <div style={{ fontSize: 10, color: "#F2C46D", marginBottom: 4, fontFamily: "'IBM Plex Mono', monospace" }}>WITHOUT AI UPLIFT</div>
                          <div style={{ fontSize: 11, color: "#9a9eb0" }}>
                            Limited to known pathogens and published modifications. Dangerous, but the design space is bounded by existing literature.
                          </div>
                        </div>
                        <div style={{ padding: "8px 10px", background: "#14161c", borderRadius: 6 }}>
                          <div style={{ fontSize: 10, color: "#D45D79", marginBottom: 4, fontFamily: "'IBM Plex Mono', monospace" }}>WITH AI UPLIFT</div>
                          <div style={{ fontSize: 11, color: "#9a9eb0" }}>
                            Potential access to novel designs — optimizations for transmissibility, lethality, or immune evasion that don't exist in literature. This is where catastrophic tail risk may live.
                          </div>
                        </div>
                      </div>
                      <p style={{ margin: "0 0 10px 0" }}>
                        If frontier bio models uniquely enable <em>novel</em> catastrophic agents (not just faster access to known ones), 
                        then safeguards have value even if they're "cheap" to bypass — they're guarding a qualitatively different capability.
                      </p>
                      <p style={{ margin: 0, fontSize: 11, color: "#7a7e8a" }}>
                        <strong style={{ color: "#9a9eb0" }}>Implication for safeguard design:</strong> Safeguards that are non-obvious (model appears to work but subtly avoids dangerous optimizations) 
                        may be more valuable than refusal-based safeguards, since attackers may not realize they need to break them.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ── Cost curves ── */}
            {activeTab === "costs" && (
              <div style={panelStyle}>
                <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 4, fontFamily: "'Space Mono', monospace" }}>Compute cost trajectories</div>
                <div style={{ fontSize: 12, color: "#7a7e8a", marginBottom: 16 }}>Solid = with physical limits · Dashed = naive exponential · Log scale</div>
                <MiniChart
                  data={[{ values: results.trainingCosts }, { values: results.fineTuneCosts }, { values: results.trainingCostsNaive }, { values: results.fineTuneCostsNaive }]}
                  width={560} height={240}
                  colors={["#D45D79", "#6ECFB0", "#D45D79", "#6ECFB0"]}
                  dashed={[false, false, true, true]}
                  legend={["Train (w/ limits)", "Fine-tune (w/ limits)", "Train (naive)", "Fine-tune (naive)"]}
                  yFormat={fmt} logScale={true}
                />
                <div style={{ marginTop: 20, padding: "12px 16px", background: "#14161c", borderRadius: 8 }}>
                  <div style={{ fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color: "#7a7e8a", marginBottom: 10 }}>ATTACKER AFFORDABILITY TIMELINE</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                    {attackerProfiles.map(profile => {
                      const yr = results.trainingCosts.findIndex(c => profile.budget >= c);
                      return (
                        <div key={profile.name} style={{ display: "flex", alignItems: "center", gap: 6, padding: "4px 10px", borderRadius: 4, background: yr >= 0 ? `${profile.color}18` : "#1e2028" }}>
                          <div style={{ width: 6, height: 6, borderRadius: "50%", background: profile.color }} />
                          <span style={{ fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color: profile.color }}>{profile.name}:</span>
                          <span style={{ fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", color: "#e8e0d4", fontWeight: 600 }}>
                            {yr < 0 ? `>${YEARS}y` : yr === 0 ? "Now" : `+${yr}y`}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* ── Physical limits ── */}
            {activeTab === "limits" && (
              <div style={panelStyle}>
                <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 4, fontFamily: "'Space Mono', monospace" }}>Decay rate slowdown</div>
                <div style={{ fontSize: 12, color: "#7a7e8a", marginBottom: 16 }}>The annual cost reduction multiplier declines as physical limits approach</div>
                <MiniChart
                  data={[{ values: results.trainingRates }, { values: results.fineTuneRates }]}
                  width={560} height={200}
                  colors={["#D45D79", "#6ECFB0"]}
                  legend={["Training ×/yr", "Fine-tuning ×/yr"]}
                  yFormat={v => `${v.toFixed(1)}×`}
                />
                <div style={{ marginTop: 20, fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color: "#9B8FFF", marginBottom: 12 }}>PHYSICAL LIMIT REGIMES</div>
                {[
                  { period: "2025–2030", rate: `${results.trainingRates[0].toFixed(1)}× → ${results.trainingRates[5].toFixed(1)}×`, label: "Golden era", color: "#6ECFB0" },
                  { period: "2030–2035", rate: `${results.trainingRates[5].toFixed(1)}× → ${results.trainingRates[10].toFixed(1)}×`, label: "Diminishing returns", color: "#F2C46D" },
                  { period: "2035+", rate: `${results.trainingRates[10].toFixed(1)}× → ${results.trainingRates[Math.min(14, YEARS)].toFixed(1)}×`, label: "Near-plateau", color: "#D45D79" },
                ].map((r, i) => (
                  <div key={i} style={{ display: "flex", gap: 12, marginBottom: 8, padding: "8px 12px", background: "#14161c", borderRadius: 6, borderLeft: `3px solid ${r.color}` }}>
                    <div style={{ fontSize: 12, color: r.color, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, width: 80 }}>{r.period}</div>
                    <div style={{ fontSize: 11, color: "#7a7e8a", width: 100 }}>{r.rate}/yr</div>
                    <div style={{ fontSize: 12, color: "#c4c8d4" }}>{r.label}</div>
                  </div>
                ))}
              </div>
            )}

            {/* ── Threat matrix ── */}
            {activeTab === "threat" && (
              <div style={panelStyle}>
                <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 4, fontFamily: "'Space Mono', monospace" }}>Residual threat by attacker profile</div>
                <div style={{ fontSize: 12, color: "#7a7e8a", marginBottom: 16 }}>Probability of successful biological threat creation after all interventions</div>
                <ThreatHeatmap data={results.threatMatrix} years={YEARS + 1} profiles={attackerProfiles} width={560} />
                <div style={insightStyle("#F2C46D")}>
                  <div style={insightText}>
                    {(() => {
                      const yr5 = results.threatMatrix.reduce((s, r) => s + r[5], 0) / 4;
                      if (yr5 > 0.5) return "⚠ Average residual threat at yr 5 exceeds 50%. Intervention portfolio insufficient.";
                      if (yr5 > 0.25) return "Moderate residual risk. Synthesis screening has highest marginal impact.";
                      return "Interventions holding through yr 5+. Physical limits extend window.";
                    })()}
                  </div>
                </div>
              </div>
            )}

            {/* ── Intervention value ── */}
            {activeTab === "value" && (
              <div style={panelStyle}>
                <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 4, fontFamily: "'Space Mono', monospace" }}>Marginal intervention value at year 5</div>
                <div style={{ fontSize: 12, color: "#7a7e8a", marginBottom: 16 }}>How much does each intervention reduce expected risk?</div>
                <InterventionBars interventions={results.interventionValues} width={560} />
                <div style={insightStyle("#E88B6E")}>
                  <div style={insightText}>
                    {(() => {
                      const sorted = [...results.interventionValues].sort((a, b) => b.value - a.value);
                      const top = sorted[0];
                      return `Highest value: ${top.name} (${top.value.toFixed(0)}%). ${
                        top.name === "Model safeguards" ? "Depends on safeguard robustness crux — check the Cruxes tab." :
                        top.name === "Synthesis screening" ? "Robust to cost curves AND physical limit assumptions." :
                        "Effectiveness degrades as costs fall."}`;
                    })()}
                  </div>
                </div>
              </div>
            )}

            {/* ── Model details ── */}
            {activeTab === "model" && (
              <div style={panelStyle}>
                <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 16, fontFamily: "'Space Mono', monospace" }}>Model & Methodology Details</div>
                
                {/* Fine-tuning cost model */}
                <div style={{ marginBottom: 24 }}>
                  <div style={sectionLabelStyle("#9B8FFF")}>Fine-tuning Cost Model</div>
                  <div style={{ fontSize: 12, color: "#c4c8d4", lineHeight: 1.7 }}>
                    <p style={{ margin: "0 0 12px 0" }}>
                      Converting fine-tuning steps to dollar costs requires modeling the compute required per step.
                    </p>
                    <div style={{ background: "#14161c", padding: "12px 14px", borderRadius: 6, fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, marginBottom: 12 }}>
                      <div style={{ color: "#7a7e8a", marginBottom: 8 }}>// Theoretical FLOP per step (LoRA fine-tuning)</div>
                      <div>FLOP_per_step ≈ 6 × params × batch_size × seq_len</div>
                      <div style={{ marginTop: 4 }}>= 6 × {modelSize}B × 8 × 2048 = <span style={{ color: "#6ECFB0" }}>{((6 * modelSize * 8 * 2048) / 1e6).toFixed(1)}T FLOP/step</span></div>
                      <div style={{ marginTop: 12, color: "#7a7e8a" }}>// H100 throughput (theoretical)</div>
                      <div>~3×10¹⁵ FLOP/s → {(3e15 / (6 * modelSize * 1e9 * 8 * 2048)).toFixed(1)} steps/sec theoretical</div>
                    </div>
                    <p style={{ margin: "0 0 12px 0" }}>
                      <strong style={{ color: "#D45D79" }}>However</strong>, real-world training is 10-30× less efficient than theoretical FLOP calculations due to:
                    </p>
                    <ul style={{ margin: "0 0 12px 0", paddingLeft: 20, color: "#9a9eb0" }}>
                      <li>Memory bandwidth limits (not compute-bound)</li>
                      <li>Optimizer state overhead (Adam momentum, variance)</li>
                      <li>Gradient accumulation & synchronization</li>
                      <li>Multi-GPU communication overhead</li>
                      <li>Data loading & checkpointing</li>
                    </ul>
                  </div>
                </div>

                {/* Empirical calibration */}
                <div style={{ marginBottom: 24 }}>
                  <div style={sectionLabelStyle("#6ECFB0")}>Empirical Calibration</div>
                  <div style={{ fontSize: 12, color: "#c4c8d4", lineHeight: 1.7 }}>
                    <p style={{ margin: "0 0 12px 0" }}>
                      We calibrate against empirical data from the <strong>Deep Ignorance</strong> paper (EleutherAI, August 2025):
                    </p>
                    <div style={{ background: "#14161c", padding: "12px 14px", borderRadius: 6, marginBottom: 12 }}>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                        <div>
                          <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Paper's setup</div>
                          <div style={{ fontSize: 12 }}>6.9B parameter model</div>
                          <div style={{ fontSize: 12 }}>10,000 fine-tuning steps</div>
                          <div style={{ fontSize: 12 }}>305M tokens processed</div>
                          <div style={{ fontSize: 12 }}>2× H200 GPUs</div>
                        </div>
                        <div>
                          <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Measured result</div>
                          <div style={{ fontSize: 12 }}>~17 GPU-hours wall clock</div>
                          <div style={{ fontSize: 12 }}>= ~3 seconds/step actual</div>
                          <div style={{ fontSize: 12 }}>vs ~0.2 sec/step theoretical</div>
                          <div style={{ fontSize: 12, color: "#F2C46D", fontWeight: 600 }}>≈ 15× efficiency loss</div>
                        </div>
                      </div>
                    </div>
                    <p style={{ margin: "0 0 12px 0" }}>
                      We apply this <strong style={{ color: "#6ECFB0" }}>15× efficiency factor</strong> to scale theoretical calculations to realistic costs.
                      This is conservative — some setups may be even less efficient.
                    </p>
                  </div>
                </div>

                {/* Current settings calculation */}
                <div style={{ marginBottom: 24 }}>
                  <div style={sectionLabelStyle("#F2C46D")}>Your Current Settings</div>
                  <div style={{ background: "#14161c", padding: "14px 16px", borderRadius: 6 }}>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                      <div>
                        <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Model size</div>
                        <div style={{ fontSize: 18, fontWeight: 600, color: "#e8e0d4", fontFamily: "'IBM Plex Mono', monospace" }}>{modelSize}B</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Steps to break</div>
                        <div style={{ fontSize: 18, fontWeight: 600, color: "#9B8FFF", fontFamily: "'IBM Plex Mono', monospace" }}>{fmtSteps(stepsToBreak)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>GPU-hour rate</div>
                        <div style={{ fontSize: 18, fontWeight: 600, color: "#e8e0d4", fontFamily: "'IBM Plex Mono', monospace" }}>${gpuHourCost}/hr</div>
                      </div>
                    </div>
                    <div style={{ borderTop: "1px solid #2a2d38", marginTop: 14, paddingTop: 14, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                      <div>
                        <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>GPU-hours needed</div>
                        <div style={{ fontSize: 18, fontWeight: 600, color: "#6ECFB0", fontFamily: "'IBM Plex Mono', monospace" }}>{results.breakGpuHours.toFixed(1)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Wall clock (1 GPU)</div>
                        <div style={{ fontSize: 18, fontWeight: 600, color: "#e8e0d4", fontFamily: "'IBM Plex Mono', monospace" }}>{(results.breakGpuHours).toFixed(1)}h</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 10, color: "#7a7e8a", marginBottom: 4 }}>Total cost</div>
                        <div style={{ fontSize: 18, fontWeight: 600, color: "#D45D79", fontFamily: "'IBM Plex Mono', monospace" }}>{fmt(results.breakCost)}</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Training cost model */}
                <div style={{ marginBottom: 24 }}>
                  <div style={sectionLabelStyle("#D45D79")}>Training-from-Scratch Cost Model</div>
                  <div style={{ fontSize: 12, color: "#c4c8d4", lineHeight: 1.7 }}>
                    <p style={{ margin: "0 0 12px 0" }}>
                      Training costs follow a <strong>decaying exponential</strong> model that captures physical limits:
                    </p>
                    <div style={{ background: "#14161c", padding: "12px 14px", borderRadius: 6, fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, marginBottom: 12 }}>
                      <div style={{ color: "#7a7e8a", marginBottom: 8 }}>// Cost model with physical floor</div>
                      <div>cost(t) = floor + (base − floor) × exp(−∫k(s)ds)</div>
                      <div style={{ marginTop: 8, color: "#7a7e8a" }}>// Where decay rate k(t) itself decays:</div>
                      <div>k(t) = k₀ × exp(−damping × t)</div>
                    </div>
                    <p style={{ margin: 0 }}>
                      This captures three regimes: (1) <strong style={{ color: "#6ECFB0" }}>Golden era</strong> (2025-2030) with rapid cost declines,
                      (2) <strong style={{ color: "#F2C46D" }}>Diminishing returns</strong> (2030-2035) as easy gains exhaust,
                      (3) <strong style={{ color: "#D45D79" }}>Near-plateau</strong> (2035+) approaching physical limits.
                    </p>
                  </div>
                </div>

                {/* Key assumption */}
                <div style={{ marginBottom: 24 }}>
                  <div style={sectionLabelStyle("#D45D79")}>Critical Assumption: Fine-tuning Can Break Safeguards</div>
                  <div style={{ fontSize: 12, color: "#c4c8d4", lineHeight: 1.7 }}>
                    <div style={{ background: "#D45D7915", padding: "12px 14px", borderRadius: 6, borderLeft: "3px solid #D45D79", marginBottom: 12 }}>
                      <strong style={{ color: "#D45D79" }}>This tool assumes any safeguard can be undone with enough fine-tuning steps.</strong>
                    </div>
                    <p style={{ margin: "0 0 12px 0" }}>
                      This is the dominant empirical finding for current safeguard methods (RLHF, Constitutional AI, Circuit Breaking, etc.) — 
                      they can typically be broken in dozens to thousands of steps.
                    </p>
                    <p style={{ margin: "0 0 12px 0" }}>
                      <strong style={{ color: "#6ECFB0" }}>However, this assumption may not hold for all safeguard architectures:</strong>
                    </p>
                    <ul style={{ margin: "0 0 12px 0", paddingLeft: 20, color: "#9a9eb0" }}>
                     <li><strong style={{ color: "#c4c8d4" }}>Distributed representations</strong>: If safety-relevant features are deeply entangled with capability features, 
                      low-rank fine-tuning (LoRA) may be unable to target them without destroying capabilities.</li>
                      <li><strong style={{ color: "#c4c8d4" }}>Architectural safeguards</strong>: Hardware-level interventions, inference-time monitors, or 
                      cryptographic commitments can't be fine-tuned away at all.</li>
                    </ul>
                    <p style={{ margin: 0 }}>
                      If such "qualitatively harder" safeguards prove achievable, the calculus changes significantly — 
                      the relevant cost comparison becomes training-from-scratch, not fine-tuning.
                      This remains an open research question.
                    </p>
                  </div>
                </div>

                {/* Key references */}
                <div>
                  <div style={sectionLabelStyle("#7a7e8a")}>Key References</div>
                  <div style={{ fontSize: 11, color: "#9a9eb0", lineHeight: 1.6 }}>
                    <div style={{ marginBottom: 8 }}>
                      <strong style={{ color: "#c4c8d4" }}>Deep Ignorance</strong> (O'Brien, Casper, et al., 2025) — 
                      "Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs" — 
                      State-of-the-art tamper resistance, resisting 10K steps of adversarial fine-tuning.
                      <span style={{ color: "#6ECFB0", marginLeft: 8 }}>arxiv:2508.06601</span>
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      <strong style={{ color: "#c4c8d4" }}>Epoch AI</strong> — Hardware efficiency analysis suggesting ~200× headroom to CMOS ceiling 
                      (Ho & Erdil, 2023). Process node roadmaps from IRDS, TSMC, Intel.
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      <strong style={{ color: "#c4c8d4" }}>Cost calibration</strong> — Training cost estimates from Epoch AI compute trends, 
                      a16z infrastructure reports, and public cloud GPU pricing (H100 spot ~$2/hr as of 2025).
                    </div>
                    <div>
                      <strong style={{ color: "#c4c8d4" }}>Attacker populations</strong> — Illustrative estimates; 
                      see Luca Righetti's threat modeling work at GovAI for more rigorous analysis.
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div style={{ marginTop: 32, padding: "16px 0", borderTop: "1px solid #2a2d38", fontSize: 11, color: "#5a5e6a", fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1.5 }}>
          v3 — Steps-to-cost calibrated from Deep Ignorance (EleutherAI 2025): 6.9B model, 10K steps = ~17 GPU-hrs empirical.
          We use 15× efficiency factor over theoretical FLOP calc to match real-world training costs.
          Cost floor model: floor + (base − floor) × exp(−∫k(t)dt). Attacker populations illustrative.
        </div>
      </div>
    </div>
  );
}
