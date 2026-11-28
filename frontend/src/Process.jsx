import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import Logo from "./Logo.jsx";

const flow = { type: "spring", stiffness: 90, damping: 19, mass: 1.1 };

/** A constellation of audio windows that forms and links up around the logo while the model listens. */
function Constellation({ done }) {
  const ref = useRef(), doneRef = useRef(done); doneRef.current = done;
  useEffect(() => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const c = ref.current, g = c.getContext("2d"), k = devicePixelRatio || 1;
    const W = (c.width = innerWidth * k), H = (c.height = innerHeight * k), cx = W / 2, cy = H * 0.42;
    const nodes = [], sparks = []; let raf, last = 0, pull = 0;
    const add = (now) => {
      const a = Math.random() * Math.PI * 2, r = (150 + Math.random() * 330) * k * (0.6 + 0.4 * Math.abs(Math.cos(a)));
      const n = { x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r * 0.8, vx: (Math.random() - 0.5) * 0.15 * k, vy: (Math.random() - 0.5) * 0.15 * k,
        s: (2.2 + Math.random() * 3) * k, born: now, links: [] };
      n.links = nodes.map((m, i) => [Math.hypot(m.x - n.x, m.y - n.y), i]).sort((p, q) => p[0] - q[0]).slice(0, 2).map((p) => p[1]);
      nodes.push(n);
    };
    const frame = (ms) => {
      if (!doneRef.current && nodes.length < 110 && ms - last > 70) { add(ms); last = ms; }
      if (doneRef.current) pull = Math.min(pull + 0.025, 1);
      g.clearRect(0, 0, W, H);
      for (const n of nodes) {
        n.x += n.vx; n.y += n.vy;                                    // slow drift, like particles in water
        if (pull) { n.x += (cx - n.x) * pull * 0.12; n.y += (cy - n.y) * pull * 0.12; }
      }
      g.lineWidth = 1 * k;
      nodes.forEach((n, i) => n.links.forEach((j) => {
        const m = nodes[j], a = Math.min((ms - n.born) / 600, 1) * 0.28 * (1 - pull);
        g.strokeStyle = `rgba(166, 122, 44, ${a})`; g.beginPath(); g.moveTo(n.x, n.y); g.lineTo(m.x, m.y); g.stroke();
        if (Math.random() < 0.004) sparks.push({ a: n, b: m, t: 0 });
      }));
      for (const n of nodes) {
        const age = Math.max(0, Math.min((ms - n.born) / 500, 1)), pulse = 1 + 0.25 * Math.sin(ms / 400 + n.born);
        g.fillStyle = `rgba(190, 140, 52, ${0.75 * age * (1 - pull * 0.6)})`;
        g.shadowColor = "rgba(226, 186, 104, .9)"; g.shadowBlur = 10 * k;
        g.beginPath(); g.arc(n.x, n.y, n.s * age * pulse, 0, 7); g.fill();
      }
      g.shadowBlur = 0;
      for (let i = sparks.length - 1; i >= 0; i--) {                // light travelling along a thread
        const s = sparks[i]; s.t += 0.03; if (s.t >= 1) { sparks.splice(i, 1); continue; }
        g.fillStyle = "rgba(255, 246, 220, .95)"; g.beginPath();
        g.arc(s.a.x + (s.b.x - s.a.x) * s.t, s.a.y + (s.b.y - s.a.y) * s.t, 2.2 * k, 0, 7); g.fill();
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, []);
  return <canvas ref={ref} className="constellation" aria-hidden />;
}

/** The authenticating stage: the page blurs away, the logo works, and a status card ticks through the steps. */
export default function Busy({ model, source, done }) {
  const steps = ["Decoding the audio", "Cutting it into 4-second windows", `Listening with ${model}`, `Adjusting for ${source}`, "Forming the verdict"];
  const [n, setN] = useState(1), [t, setT] = useState(0);
  useEffect(() => { const i = setInterval(() => setN((k) => (k % 4) + 1), 380); return () => clearInterval(i); }, []);
  useEffect(() => { const t0 = performance.now(), i = setInterval(() => setT((performance.now() - t0) / 1000), 100); return () => clearInterval(i); }, []);
  // The server does not report progress, so the bar eases towards 92% and fills when the answer arrives.
  const p = done ? 1 : 0.92 * (1 - Math.exp(-t / 2.6));
  const at = done ? steps.length : Math.min(Math.floor(p / 0.92 * steps.length), steps.length - 1);

  return (
    <motion.div className="busy" role="status" aria-live="polite"
      initial={{ backdropFilter: "blur(0px)", backgroundColor: "rgba(250,244,231,0)" }}
      animate={{ backdropFilter: "blur(14px)", backgroundColor: "rgba(250,244,231,.55)" }}
      exit={{ backdropFilter: "blur(0px)", backgroundColor: "rgba(250,244,231,0)" }} transition={{ duration: 0.7 }}>
      <Constellation done={done} />
      <motion.div className="busy-in" initial={{ scale: 0.7, y: 20, opacity: 0 }} animate={{ scale: 1, y: 0, opacity: 1 }} exit={{ scale: 0.85, opacity: 0 }} transition={flow}>
        <Logo size={128} mode="busy" />
        <h2>{done ? "Done" : "Authenticating"}<span className="dots">{done ? "" : ".".repeat(n)}</span></h2>
        <motion.div className="steps glass pop" initial={{ y: 24, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ ...flow, delay: 0.25 }}>
          <ol>
            {steps.map((s, i) => (
              <li key={s} className={i < at ? "ok" : i === at ? "now" : ""}>
                <span className="tick">{i < at ? "✓" : ""}</span>{s}
              </li>
            ))}
          </ol>
          <div className="bar"><motion.i animate={{ width: `${p * 100}%` }} transition={{ duration: 0.35, ease: "easeOut" }} /></div>
        </motion.div>
      </motion.div>
    </motion.div>
  );
}
