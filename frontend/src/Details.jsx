import { AnimatePresence, motion } from "framer-motion";
import { createContext, useContext, useMemo, useState } from "react";

/* ---------- hover explanations ---------- */
const TipCtx = createContext(() => {});
export function TipLayer({ children }) {
  const [t, setT] = useState(null);
  return (
    <TipCtx.Provider value={setT}>
      {children}
      <AnimatePresence>
        {t && (
          <motion.div className="tip" key="tip" style={{ "--c": t.color, left: Math.min(t.x + 18, innerWidth - 316), top: Math.max(t.y - 20, 12) }}
            initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.96 }} transition={{ duration: 0.18 }}>
            <b>{t.title}</b>
            {t.lines.filter(Boolean).map((l) => <p key={l}>{l}</p>)}
          </motion.div>
        )}
      </AnimatePresence>
    </TipCtx.Provider>
  );
}

const W = 560, H = 250;
const fmt = (v, d = 1) => (v > 0 ? "+" : "") + v.toFixed(d);
const pct = (v) => Math.round(v * 100) + "%";
const secs = (T) => (v) => v.toFixed(T < 8 ? 1 : 0) + " s";   // short files need a decimal or ticks repeat
const lean = (s) => (s >= 0 ? "human-leaning" : "AI-leaning");

/** Shared plot frame: scales, grid, axes, pointer → nearest item → tooltip + highlight. */
function Plot({ x: [x0, x1], y: [y0, y1], pad = {}, xTicks = [], yTicks = [], xFmt = String, yFmt = String, xLabel, yLabel, color, pick, children }) {
  const setTip = useContext(TipCtx);
  const [hl, setHl] = useState(null);
  const P = { l: 48, r: 14, t: 14, b: 40, ...pad };
  const X = (v) => P.l + ((v - x0) / (x1 - x0)) * (W - P.l - P.r);
  const Y = (v) => P.t + (1 - (v - y0) / (y1 - y0)) * (H - P.t - P.b);
  const s = { X, Y, P };
  const move = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) * W) / r.width, py = ((e.clientY - r.top) * H) / r.height;
    const v = { x: x0 + ((px - P.l) / (W - P.l - P.r)) * (x1 - x0), y: y0 + (1 - (py - P.t) / (H - P.t - P.b)) * (y1 - y0), px, py };
    const hit = px >= P.l - 6 && px <= W - P.r + 6 && pick(v, s);
    setHl(hit ? hit.mark : null);
    setTip(hit ? { ...hit.tip, color: hit.color || color, x: e.clientX, y: e.clientY } : null);
  };
  const leave = () => { setHl(null); setTip(null); };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} onPointerMove={move} onPointerLeave={leave} className="plot">
      {yTicks.map((t) => (
        <g key={"y" + t}><line x1={P.l} x2={W - P.r} y1={Y(t)} y2={Y(t)} className="grid" /><text x={P.l - 8} y={Y(t) + 4} textAnchor="end">{yFmt(t)}</text></g>
      ))}
      {xTicks.map((t) => <text key={"x" + t} x={X(t)} y={H - P.b + 16} textAnchor="middle">{xFmt(t)}</text>)}
      {xLabel && <text className="ax" x={(P.l + W - P.r) / 2} y={H - 6} textAnchor="middle">{xLabel}</text>}
      {yLabel && <text className="ax" transform={`translate(12 ${(P.t + H - P.b) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>}
      {children(s)}
      {hl?.dot && <circle cx={hl.dot[0]} cy={hl.dot[1]} r="7" className="hl" style={{ stroke: hl.c || color }} />}
      {hl?.rect && <rect {...hl.rect} rx="3" className="hl" style={{ stroke: hl.c || color }} />}
      {hl?.vline && <line x1={hl.vline} x2={hl.vline} y1={P.t} y2={H - P.b} className="cross" />}
    </svg>
  );
}

const ticks = (a, b, n = 4) => Array.from({ length: n + 1 }, (_, i) => a + ((b - a) * i) / n);
const nearest = (arr, v) => arr.reduce((bi, a, i) => (Math.abs(a - v) < Math.abs(arr[bi] - v) ? i : bi), 0);
const line = (pts) => pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join("");
const HUMAN = "#2f6db0", AI = "#c2412d";
const div = (v, m) => { // diverging fill: blue = evidence for human, red = evidence for AI, cream in the middle
  const k = Math.min(Math.abs(v) / (m || 1), 1), [r, g, b] = v >= 0 ? [47, 109, 176] : [194, 65, 45];
  return `rgb(${Math.round(250 + (r - 250) * k)},${Math.round(244 + (g - 244) * k)},${Math.round(232 + (b - 232) * k)})`;
};

/* ---------- 1. confidence over time ---------- */
function Timeline({ r }) {
  const ws = r.windows, T = Math.max(...ws.map((w) => w.t1)), mid = (w) => (w.t0 + w.t1) / 2, c = "#1f8a80";
  return (
    <Plot x={[0, T]} y={[0, 1]} xTicks={ticks(0, T)} yTicks={[0, 0.25, 0.5, 0.75, 1]} xFmt={secs(T)} yFmt={pct}
      xLabel="time in the recording" yLabel="chance of a human voice" color={c}
      pick={(v, { X, Y }) => {
        const i = nearest(ws.map(mid), v.x), w = ws[i], prev = ws[i - 1], d = prev ? Math.round((w.p_real - prev.p_real) * 100) : null;
        return {
          mark: { dot: [X(mid(w)), Y(w.p_real)] },
          tip: { title: `${w.t0}–${w.t1} s: ${pct(w.p_real)} human`, lines: [
            w.p_real >= 0.5 ? "This stretch sounds more like a real person than an AI voice." : "This stretch sounds more like a cloned or synthetic voice.",
            d === null ? "This is the first window of the recording." : d === 0 ? "Same as the previous window: the voice stayed consistent." :
              `${d > 0 ? "Up" : "Down"} ${Math.abs(d)} points from the previous window, so this part sounded ${d > 0 ? "more natural" : "more synthetic"}.`,
            "Points above the dashed 50% line count as human, points below count as AI.",
            !w.speech && "Mostly silence here, so it was left out of the final score."] },
        };
      }}>
      {({ X, Y, P }) => {
        const pts = ws.map((w) => [X(mid(w)), Y(w.p_real)]);
        const L = pts.length > 1 ? pts : [[X(0), pts[0][1]], [X(T), pts[0][1]]];   // a single window spans the whole file
        return (
          <>
            {ws.filter((w) => !w.speech).map((w) => <rect key={w.t0} x={X(w.t0)} y={P.t} width={X(w.t1) - X(w.t0)} height={H - P.t - P.b} fill="#e9dcc0" opacity=".5" />)}
            {r.suspicious_span && <rect x={X(r.suspicious_span[0])} y={P.t} width={X(r.suspicious_span[1]) - X(r.suspicious_span[0])} height={H - P.t - P.b} fill="none" stroke={AI} strokeDasharray="4 4" rx="4" />}
            <line x1={P.l} x2={W - P.r} y1={Y(0.5)} y2={Y(0.5)} className="mid" />
            <path d={line(L) + `L${L.at(-1)[0]},${Y(0)}L${L[0][0]},${Y(0)}Z`} fill={c} opacity=".1" />
            <motion.path d={line(L)} fill="none" stroke={c} strokeWidth="2.4" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.4 }} />
            {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="4.2" fill="#fffaf0" stroke={c} strokeWidth="2" />)}
          </>
        );
      }}
    </Plot>
  );
}

/* ---------- 2. spectrogram with per-window score strip ---------- */
const HEAT = [[251, 245, 232], [233, 196, 122], [196, 106, 48], [110, 38, 62], [40, 18, 40]];
const heat = (k) => { const f = k * (HEAT.length - 1), i = Math.min(Math.floor(f), HEAT.length - 2), u = f - i; return HEAT[i].map((a, j) => Math.round(a + (HEAT[i + 1][j] - a) * u)); };
function Spectrogram({ d }) {
  const { rows, cols, v, dur } = d.spec;
  const url = useMemo(() => {
    const cv = document.createElement("canvas"); cv.width = cols; cv.height = rows;
    const g = cv.getContext("2d"), img = g.createImageData(cols, rows);
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      const [r, gg, b] = heat(v[y * cols + x] / 99), o = ((rows - 1 - y) * cols + x) * 4;
      img.data.set([r, gg, b, 255], o);
    }
    g.putImageData(img, 0, 0); return cv.toDataURL();
  }, [v, rows, cols]);
  const sMax = Math.max(1, ...d.scores.map(Math.abs));
  return (
    <Plot x={[0, dur]} y={[0, 8000]} pad={{ t: 34 }} xTicks={ticks(0, dur)} yTicks={[0, 2000, 4000, 6000, 8000]} xFmt={secs(dur)} yFmt={(f) => f / 1000 + "k"}
      xLabel="time" yLabel="pitch (Hz)" color="#6e263e"
      pick={(p, { X, Y, P }) => {
        if (p.py < P.t - 2) {
          const i = nearest(d.centres, p.x), s = d.scores[i];
          return { color: s >= 0 ? HUMAN : AI, mark: { rect: { x: X(d.centres[i] - 1), y: 10, width: X(2) - X(0), height: 16 } },
            tip: { title: `Window at ${d.centres[i]} s: score ${fmt(s)}`, lines: [
              `The strip above the spectrogram shows the model's raw score for each 4-second window; this one is ${lean(s)}.`,
              "Blue means the sound looked natural to the model, red means it looked synthetic. Stronger colour, stronger opinion.",
              i > 0 && `Change from the previous window: ${fmt(s - d.scores[i - 1])}.`] } };
        }
        const cx = Math.max(0, Math.min(cols - 1, Math.floor((p.x / dur) * cols))), cy = Math.max(0, Math.min(rows - 1, Math.floor((p.y / 8000) * rows)));
        const e = v[cy * cols + cx], f = Math.round(((cy + 0.5) / rows) * 8000);
        return { mark: { vline: X((cx + 0.5) / cols * dur) },
          tip: { title: `${((cx + 0.5) / cols * dur).toFixed(1)} s at about ${f} Hz`, lines: [
            `Energy here is ${e}% of the loudest point in the file. Brighter colours mean more sound at this pitch and moment.`,
            f < 4000 ? "Speech energy normally sits below 4 kHz, so this is where the voice itself lives." : "Above 4 kHz you hear breath and fine texture; cloned voices often look too smooth or empty here.",
            "Sharp horizontal cut-offs usually come from compression, not from the voice."] } };
      }}>
      {({ X, Y, P }) => (
        <>
          <image href={url} x={X(0)} y={Y(8000)} width={X(dur) - X(0)} height={Y(0) - Y(8000)} preserveAspectRatio="none" style={{ imageRendering: "pixelated" }} />
          {d.centres.map((c, i) => <rect key={i} x={X(c - 1)} y={10} width={X(2) - X(0)} height={16} fill={div(d.scores[i], sMax)} />)}
          <text x={P.l - 8} y={22} textAnchor="end">score</text>
        </>
      )}
    </Plot>
  );
}

/* ---------- 3. time x frequency-band influence ---------- */
function BandMap({ d }) {
  const n = d.centres.length, m = Math.max(1e-3, ...d.band_map.flat().map(Math.abs));
  return (
    <Plot x={[0, n]} y={[0, 8]} pad={{ l: 70 }} xTicks={ticks(0, n - 1, Math.min(n - 1, 5)).map(Math.round).map((i) => i + 0.5)} yTicks={[]} xFmt={(i) => d.centres[Math.floor(i)] + " s"}
      xLabel="window centre" color={HUMAN}
      pick={(p, { X, Y }) => {
        const i = Math.floor(p.x), b = Math.floor(p.y); if (i < 0 || i >= n || b < 0 || b > 7) return null;
        const v = d.band_map[i][b];
        return { color: v >= 0 ? HUMAN : AI, mark: { rect: { x: X(i), y: Y(b + 1), width: X(1) - X(0), height: Y(0) - Y(1) } },
          tip: { title: `${d.bands[b]} at ${d.centres[i]} s: ${fmt(v, 2)}`, lines: [
            `Silencing only this band in this window changes the score by ${fmt(-v, 2)}.`,
            v >= 0 ? "So here this band was evidence for a real voice: without it the file sounds less human." : "So here this band pushed the verdict towards AI: without it the file sounds more human.",
            "Darker cells matter more. A column that is red from top to bottom marks a moment that sounded synthetic across the whole range."] } };
      }}>
      {({ X, Y }) => (
        <>
          {d.band_map.map((row, i) => row.map((v, b) => <rect key={i + "-" + b} x={X(i) + 0.5} y={Y(b + 1) + 0.5} width={X(1) - X(0) - 1} height={Y(0) - Y(1) - 1} fill={div(v, m)} />))}
          {d.bands.map((l, b) => <text key={l} x={X(0) - 8} y={(Y(b) + Y(b + 1)) / 2 + 4} textAnchor="end">{l}</text>)}
        </>
      )}
    </Plot>
  );
}

/* ---------- 4/5. bar charts of influence ---------- */
function Bars({ vals, labels, xLabel, explain }) {
  const m = Math.max(0.05, ...vals.map(Math.abs)) * 1.15, n = vals.length;
  return (
    <Plot x={[0, n]} y={[-m, m]} xTicks={vals.map((_, i) => i + 0.5)} xFmt={(i) => labels[Math.floor(i)]} yTicks={[-m, -m / 2, 0, m / 2, m].map((t) => +t.toFixed(2))} yFmt={(t) => t.toFixed(2)}
      xLabel={xLabel} yLabel="effect on score" color={HUMAN}
      pick={(p, { X, Y }) => {
        const i = Math.floor(p.x); if (i < 0 || i >= n) return null;
        const v = vals[i], rank = [...vals].map(Math.abs).sort((a, b) => b - a).indexOf(Math.abs(v)) + 1;
        return { color: v >= 0 ? HUMAN : AI, mark: { rect: { x: X(i + 0.15), y: Y(Math.max(v, 0)), width: X(0.7) - X(0), height: Math.abs(Y(v) - Y(0)) } },
          tip: explain(i, v, rank) };
      }}>
      {({ X, Y }) => (
        <>
          <line x1={X(0)} x2={X(n)} y1={Y(0)} y2={Y(0)} className="mid solid" />
          {vals.map((v, i) => (
            <motion.rect key={i} x={X(i + 0.15)} width={X(0.7) - X(0)} rx="3" fill={v >= 0 ? HUMAN : AI} opacity=".85"
              initial={{ y: Y(0), height: 0 }} animate={{ y: Y(Math.max(v, 0)), height: Math.abs(Y(v) - Y(0)) }} transition={{ type: "spring", stiffness: 90, damping: 16, delay: i * 0.04 }} />
          ))}
        </>
      )}
    </Plot>
  );
}

/* ---------- 6. attention ---------- */
function Attention({ d }) {
  const { t, att, env } = d.attention, T = t.at(-1) || 1, c = "#c07c12";
  return (
    <Plot x={[0, T]} y={[0, 1]} xTicks={ticks(0, T)} yTicks={[0, 0.5, 1]} xFmt={secs(T)} yFmt={pct} xLabel="time" yLabel="relative level" color={c}
      pick={(p, { X, Y }) => {
        const i = nearest(t, p.x), a = att[i], l = env[i], prev = att[Math.max(0, i - 3)];
        return { mark: { dot: [X(t[i]), Y(a)], vline: X(t[i]) },
          tip: { title: `${t[i].toFixed(1)} s: attention ${pct(a)}, loudness ${pct(l)}`, lines: [
            a > 0.6 ? "A peak: the model weighted this moment heavily when forming its verdict." : a < 0.25 ? "Low attention: this moment barely affected the verdict." : "Moderate attention: this moment contributed, but was not decisive.",
            a - prev > 0.15 ? "Attention is rising here, so the model found something telling just now." : prev - a > 0.15 ? "Attention is falling, so the model is moving its focus away." : "Attention is steady around here.",
            l < 0.1 && a > 0.4 ? "The audio is nearly silent, so breaths or background noise are shaping the decision." : "The shaded area is loudness, for comparison with where the model listens."] } };
      }}>
      {({ X, Y }) => {
        const L = t.map((v, i) => [X(v), Y(env[i])]), A = t.map((v, i) => [X(v), Y(att[i])]);
        return (
          <>
            <path d={line(L) + `L${L.at(-1)[0]},${Y(0)}L${L[0][0]},${Y(0)}Z`} fill="#b9ad92" opacity=".35" />
            <motion.path d={line(A)} fill="none" stroke={c} strokeWidth="1.8" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.6 }} />
          </>
        );
      }}
    </Plot>
  );
}

/* ---------- 7. long-term spectrum ---------- */
function Spectrum({ d }) {
  const { f, db } = d.spectrum, lo = Math.floor(Math.min(...db) / 10) * 10, hi = Math.ceil(Math.max(...db) / 10) * 10, c = "#4a52b0";
  return (
    <Plot x={[0, 8000]} y={[lo, hi]} xTicks={[0, 2000, 4000, 6000, 8000]} yTicks={ticks(lo, hi, 4)} xFmt={(v) => v / 1000 + " kHz"} yFmt={(v) => v.toFixed(0)}
      xLabel="frequency" yLabel="average level (dB)" color={c}
      pick={(p, { X, Y }) => {
        const i = nearest(f, p.x), step = i > 2 ? db[i] - db[i - 3] : 0;
        return { mark: { dot: [X(f[i]), Y(db[i])] },
          tip: { title: `${Math.round(f[i])} Hz: ${db[i].toFixed(1)} dB`, lines: [
            "The average loudness of this pitch across the whole file.",
            step < -12 ? "A steep drop just before this point: the audio was probably compressed or filtered here." : step > 6 ? "Rising energy here, often a vowel resonance or added noise." : "A gentle slope: nothing unusual at this pitch.",
            f[i] > 4000 ? "Real recordings keep some breath and room noise up here; very clean or empty highs can hint at synthesis." : "Most of a voice's character lives in this lower range."] } };
      }}>
      {({ X, Y }) => {
        const pts = f.map((v, i) => [X(v), Y(db[i])]);
        return (
          <>
            <path d={line(pts) + `L${pts.at(-1)[0]},${Y(lo)}L${pts[0][0]},${Y(lo)}Z`} fill={c} opacity=".09" />
            <motion.path d={line(pts)} fill="none" stroke={c} strokeWidth="2" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.4 }} />
          </>
        );
      }}
    </Plot>
  );
}

/* ---------- 8. reference distribution (only when the model ships reference scores) ---------- */
function Reference({ R }) {
  const g = R.grid, mx = Math.max(...R.real, ...R.fake);
  return (
    <Plot x={[g[0], g.at(-1)]} y={[0, mx]} xTicks={ticks(g[0], g.at(-1))} xFmt={(v) => v.toFixed(0)} yTicks={[]} xLabel="detector score (AI-like to human-like)" color="#8a3f86"
      pick={(p, { X, Y }) => {
        const i = nearest(g, p.x), better = R.real[i] > R.fake[i];
        return { mark: { vline: X(g[i]) }, tip: { title: `Score ${g[i].toFixed(1)}`, lines: [
          `At this score, verified ${better ? "real voices" : "AI voices"} are more common.`,
          `This file scored ${R.file_score}; the dashed line at ${R.threshold} is where the verdict flips.`] } };
      }}>
      {({ X, Y }) => {
        const P = (a) => line(g.map((v, i) => [X(v), Y(a[i])]));
        return (
          <>
            <path d={P(R.fake)} fill="none" stroke={AI} strokeWidth="2" strokeDasharray="6 5" />
            <path d={P(R.real)} fill="none" stroke={HUMAN} strokeWidth="2.2" />
            <line x1={X(R.threshold)} x2={X(R.threshold)} y1={Y(mx)} y2={Y(0)} className="mid" />
            <line x1={X(R.file_score)} x2={X(R.file_score)} y1={Y(mx)} y2={Y(0)} stroke="#3a2c17" strokeWidth="2.5" />
          </>
        );
      }}
    </Plot>
  );
}

/* ---------- window replay: threads branch from the file to every 4-second window ---------- */
function Replay({ r }) {
  const setTip = useContext(TipCtx);
  const narrow = innerWidth < 700;                                               // phones: one narrower column, so text stays readable
  const ws = r.windows.slice(0, 24), RW = narrow ? 560 : 1100, row = 34, top = 14, rx = narrow ? 40 : 70;
  const per = ws.length > 12 && !narrow ? Math.ceil(ws.length / 2) : ws.length;  // one column, or two when long
  const H2 = top * 2 + per * row, cx0 = narrow ? 180 : ws.length > 12 ? 300 : 420, cw = narrow ? 370 : ws.length > 12 ? 330 : 560;
  const pos = (i) => ({ x: cx0 + Math.floor(i / per) * 380, y: top + (i % per) * row });
  return (
    <svg viewBox={`0 0 ${RW} ${H2}`} className="plot replay" onPointerLeave={() => setTip(null)}>
      <circle cx={rx} cy={H2 / 2} r="16" className="root" />
      <text x={rx} y={H2 / 2 + 34} textAnchor="middle" className="ax">this file</text>
      {ws.map((w, i) => {
        const { x, y } = pos(i), yc = y + row / 2 - 3, human = w.p_real >= 0.5, c = !w.speech ? "#b9ad92" : human ? HUMAN : AI;
        const d = `M${rx + 16} ${H2 / 2} C ${rx + (narrow ? 70 : 140)} ${H2 / 2}, ${x - (narrow ? 70 : 150)} ${yc}, ${x} ${yc}`;
        const tip = { title: `${w.t0}–${w.t1} s: ${pct(w.p_real)} human`, color: c, lines: [
          !w.speech ? "Mostly silence: replayed for completeness but left out of the final score."
            : human ? "This window sounded like a real person." : "This window sounded like a cloned or synthetic voice.",
          i > 0 ? `Compared with the window before: ${fmt((w.p_real - ws[i - 1].p_real) * 100, 0)} points.` : "The first window of the recording.",
          "The final verdict is the middle value of all speech windows, so one odd window cannot flip it alone."] };
        return (
          <g key={i} onPointerMove={(e) => setTip({ ...tip, x: e.clientX, y: e.clientY })}>
            <motion.path d={d} fill="none" stroke={c} strokeWidth="1.6" opacity=".7" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
              transition={{ duration: 0.9, delay: 0.3 + i * 0.12, ease: [0.3, 0.7, 0.2, 1] }} />
            <motion.g initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} transition={{ type: "spring", stiffness: 90, damping: 15, delay: 0.9 + i * 0.12 }}>
              <rect x={x} y={y + 2} width={cw} height={row - 10} rx="8" fill="#fffaf0" stroke={c} strokeWidth="1.4" />
              <rect x={x} y={y + 2} width={cw * w.p_real} height={row - 10} rx="8" fill={c} opacity=".12" />
              <text x={x + 12} y={yc + 4} className="rt">{w.t0}–{w.t1} s</text>
              <text x={x + cw - 12} y={yc + 4} textAnchor="end" className="rt" style={{ fill: c }}>{w.speech ? `${pct(w.p_real)} human` : "quiet"}</text>
            </motion.g>
          </g>
        );
      })}
    </svg>
  );
}

/* ---------- acoustic measurements ---------- */
const ACOUSTIC = [
  ["pause_ratio", "Pauses", (v) => pct(v), "Share of the file that is silence. Cloned voices often have unnaturally even, short pauses."],
  ["loudness_range_db", "Loudness range", (v) => v.toFixed(1) + " dB", "Difference between quiet and loud speech. Real speech usually varies more; AI speech can sound flat."],
  ["energy_above_4k_pct", "Energy above 4 kHz", (v) => v.toFixed(2) + "%", "Breath and fine texture. Very low values suggest heavy compression or a synthetic voice with smoothed highs."],
  ["energy_below_300hz_pct", "Energy below 300 Hz", (v) => v.toFixed(1) + "%", "Low rumble and the base of the voice. Phone calls cut this off; studio-clean AI audio keeps it unusually tidy."],
  ["spectral_flatness", "Spectral flatness", (v) => v.toFixed(3), "Near 0 means tonal (clear voice), near 1 means noise-like. Unusual values can come from vocoders."],
  ["spectral_centroid_hz", "Brightness", (v) => Math.round(v) + " Hz", "The average pitch centre of the sound. Higher values sound brighter or thinner."],
  ["centroid_variability_hz", "Brightness variation", (v) => Math.round(v) + " Hz", "How much brightness moves while speaking. Natural speech varies; synthetic speech can be oddly steady."],
];
function Acoustic({ a }) {
  const setTip = useContext(TipCtx);
  return (
    <div className="metrics">
      {ACOUSTIC.map(([k, name, f, why]) => (
        <div key={k} onPointerMove={(e) => setTip({ title: `${name}: ${f(a[k])}`, lines: [why, "Compare these numbers between a known real recording and this file."], color: "#7a6a3a", x: e.clientX, y: e.clientY })}
          onPointerLeave={() => setTip(null)}>
          <span>{name}</span><b>{f(a[k])}</b>
        </div>
      ))}
    </div>
  );
}

/* ---------- page ---------- */
function Card({ title, note, children, i, wide }) {
  return (
    <motion.article className={"chart glass pop" + (wide ? " wide" : "")} initial={{ y: 40, opacity: 0 }} whileInView={{ y: 0, opacity: 1 }} viewport={{ once: true, margin: "-40px" }}
      transition={{ type: "spring", stiffness: 80, damping: 18, delay: (i % 2) * 0.08 }}>
      <h3>{title}</h3><p className="note">{note}</p>{children}
    </motion.article>
  );
}

export default function Details({ r }) {
  const d = r.diagnostics, n = d?.layer_drop?.length || 0;
  const cards = [
    ["Window replay", "The file branches into its 4-second windows, replayed in order. Hover a window for what it means.", <Replay r={r} />, true],
    ["Confidence over time", "Each point is a 4-second window. Hover a point to see what it means.", <Timeline r={r} />],
    d && ["Spectrogram", "What the recording contains, pitch by moment, with the model's opinion of each window on top.", <Spectrogram d={d} />],
    d && ["Which frequencies, when", "Each cell: how much one 1 kHz band at one moment pushed the verdict. Blue is evidence for human, red for AI.", <BandMap d={d} />],
    d && ["Frequency bands overall", "The same influence, averaged over the speech in the file.", <Bars vals={d.band_mean} labels={d.bands.map((b) => b.replace(" kHz", ""))} xLabel="frequency band (kHz)"
      explain={(i, v, rank) => ({ title: `${d.bands[i]}: ${fmt(v, 2)}`, lines: [
        v >= 0 ? "Removing this band lowers the score, so it is evidence of a real voice." : "Removing this band raises the score, so it is evidence of an AI voice.",
        `Rank ${rank} of 8 by influence${rank <= 2 ? ": one of the parts the verdict depends on most." : "."}`,
        i < 4 ? "Lower bands carry the voice itself: pitch, vowels and tone." : "Higher bands carry breath, hiss and texture, where cloning artefacts often hide."] })} />],
    d && ["Model layers", "VoxTrace listens through 13 layers, from raw sound texture (left) to speech content (right).", <Bars vals={d.layer_drop} labels={Array.from({ length: n }, (_, i) => (i ? String(i) : "CNN"))} xLabel="WavLM layer"
      explain={(i, v, rank) => ({ title: `${i ? "Layer " + i : "CNN front end"}: ${fmt(v, 2)}`, lines: [
        v >= 0 ? "This layer's view of the audio made it sound human." : "This layer's view of the audio pointed towards AI.",
        i <= 4 ? "Low layers describe sound texture: noise, timbre, codec traces." : i <= 8 ? "Middle layers describe phonetics: how sounds are formed." : "High layers describe speech content and rhythm.",
        `The model normally gives this layer ${pct(d.layer_weight[i])} of its attention. Rank ${rank} of ${n} for this file.`] })} />],
    d && ["Where the model listens", "The gold line is the model's attention, the shaded area is loudness.", <Attention d={d} />],
    d && ["Tone colour of the file", "The average level of every pitch across the recording.", <Spectrum d={d} />],
    d && ["Acoustic measurements", "Hover a measurement for what it means.", <Acoustic a={d.acoustic} />],
    r.reference && ["Among known voices", "Where this file's score sits between verified real (blue) and AI (red) voices.", <Reference R={r.reference} />],
  ].filter(Boolean);

  return (
    <section className="details">
      <div className="dhead">
        <div><h2>How VoxTrace reached this verdict</h2><p className="muted">{r.verdict} with {r.model}. {r.notes?.[0] || ""}</p></div>
      </div>
      <div className="charts">
        {cards.map(([t, nt, c, wide], i) => <Card key={t} title={t} note={nt} i={i} wide={wide}>{c}</Card>)}
      </div>
      {!d && <p className="muted center">Detailed diagnostics are available for the VoxTrace models only.</p>}
    </section>
  );
}
