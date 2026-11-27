import { motion } from "framer-motion";
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";

// VT mark: a sound-wave crossbar (the T) that keeps slithering, two converging trace arms (the V),
// a centre stem, a scanning ring and the point where the traces meet.
const still = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
export const wave = (t = 0, amp = 6.5) => {
  let d = "M28 34";
  for (let x = 30; x <= 92; x += 2) {
    const u = (x - 28) / 64;
    d += ` L${x} ${(34 - amp * Math.sin(Math.PI * u) ** 0.6 * Math.sin(u * Math.PI * 4 - t)).toFixed(2)}`;
  }
  return d;
};
const ARMS = ["M28 34 L60 91", "M92 34 L60 91"];
const STEM = "M60 46 L60 70", RING = "M112 60 A52 52 0 1 1 8 60 A52 52 0 1 1 112 60";

/** The wave crossbar, moving like a snake (speed in radians per second). */
function Wiggle({ speed = 4, ...p }) {
  const ref = useRef();
  useEffect(() => {
    if (still()) return;
    let raf; const f = (ms) => { ref.current?.setAttribute("d", wave((ms / 1000) * speed)); raf = requestAnimationFrame(f); };
    raf = requestAnimationFrame(f); return () => cancelAnimationFrame(raf);
  }, [speed]);
  return <path ref={ref} d={wave()} {...p} />;
}

function Defs({ id }) {
  return (
    <defs>
      <linearGradient id={id + "a"} x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#ecc874" /><stop offset=".55" stopColor="#b5852f" /><stop offset="1" stopColor="#6e4a17" />
      </linearGradient>
      <linearGradient id={id + "b"} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stopColor="#e3b75e" /><stop offset="1" stopColor="#62410f" />
      </linearGradient>
      <filter id={id + "s"} x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="1.5" stdDeviation="1.2" floodColor="#6e4a17" floodOpacity=".35" /></filter>
    </defs>
  );
}
const P = { fill: "none", strokeLinecap: "round", strokeLinejoin: "round" };
const loop = (duration, extra = {}) => ({ duration, repeat: Infinity, ease: "easeInOut", ...extra });

/** mode: "static" (wave keeps wiggling) | "busy" (while authenticating) */
export default function Logo({ size = 44, mode = "static", layoutId }) {
  const id = useId().replace(/:/g, ""), busy = mode === "busy";
  const a = `url(#${id}a)`, b = `url(#${id}b)`;
  return (
    <motion.svg layoutId={layoutId} width={size} height={size} viewBox="0 0 120 120" style={{ overflow: "visible" }} aria-hidden>
      <Defs id={id} />
      <g filter={`url(#${id}s)`}>
        {busy ? (
          <>
            <circle cx="60" cy="60" r="52" stroke="#dccaa2" strokeWidth="3" {...P} opacity=".6" />
            <motion.circle cx="60" cy="60" r="52" stroke={a} strokeWidth="3.6" {...P} strokeDasharray="70 257" animate={{ rotate: 360 }} transition={loop(1.6, { ease: "linear" })} />
            <Wiggle speed={9} stroke={a} strokeWidth="5.2" {...P} />
            {ARMS.map((d, i) => <motion.path key={d} d={d} stroke={b} strokeWidth="6" {...P} initial={{ pathLength: 1 }} animate={{ pathLength: [1, 0.35, 1] }} transition={loop(1.8, { delay: i * 0.25 })} />)}
            <motion.path d={STEM} stroke={a} strokeWidth="4.4" {...P} style={{ originY: "46px" }} animate={{ scaleY: [1, 0.45, 1] }} transition={loop(0.9)} />
            <motion.circle cx="60" cy="91" r="4.6" fill="#6e4a17" initial={{ cy: 91 }} animate={{ cy: [91, 84, 91] }} transition={loop(0.9)} />
          </>
        ) : (
          <>
            <path d={RING} stroke={a} strokeWidth="3.2" opacity=".9" {...P} />
            <Wiggle stroke={a} strokeWidth="5.2" {...P} />
            {ARMS.map((d) => <path key={d} d={d} stroke={b} strokeWidth="6" {...P} />)}
            <path d={STEM} stroke={a} strokeWidth="4.4" {...P} />
            <circle cx="60" cy="91" r="4.6" fill="#6e4a17" />
          </>
        )}
      </g>
    </motion.svg>
  );
}

/* ---------- opening sequence: wires slither in from the edges and braid into the mark ---------- */
// [approach path that ends where the logo stroke starts, logo stroke, width, gradient]
const WIRES = [
  ["M-420 -260 C -230 40, -120 -140, -40 60 S 10 10, 28 34", wave(), 5.2, "a"],
  ["M-320 420 C -120 300, -160 40, 6 22 S 20 30, 28 34", ARMS[0], 6, "b"],
  ["M440 -300 C 260 -60, 280 120, 130 40 S 100 30, 92 34", ARMS[1], 6, "b"],
  ["M60 520 C 220 360, -90 230, 70 130 S 60 70, 60 46", STEM, 4.4, "a"],
  ["M560 120 C 360 380, 220 -140, 150 40 S 112 60, 112 60", RING, 3.2, "a"],
];
const tail = (d) => d.replace(/^M[-\d.\s]+/, " ");

export function IntroMark({ size }) {
  const id = useId().replace(/:/g, "");
  const svg = useRef(), [done, setDone] = useState(false);
  useLayoutEffect(() => {
    const probe = document.createElementNS("http://www.w3.org/2000/svg", "path");
    svg.current.appendChild(probe);
    const paths = [...svg.current.querySelectorAll(".wire")];
    paths.forEach((p, i) => {
      probe.setAttribute("d", WIRES[i][1]);
      const V = probe.getTotalLength(), L = p.getTotalLength();
      p.style.strokeDasharray = `${V} ${L + V}`;            // one dash, the length of the logo stroke
      p.style.strokeDashoffset = V;                          // parked before the start: invisible
      p.getBoundingClientRect();
      p.style.transition = `stroke-dashoffset ${2.3 + i * 0.14}s cubic-bezier(.45,.05,.2,1) ${0.2 + i * 0.16}s`;
      p.style.strokeDashoffset = -(L - V);                   // slithers along the wire and settles on the logo
    });
    probe.remove();
    const t = setTimeout(() => setDone(true), 3200);
    return () => clearTimeout(t);
  }, []);
  return (
    <motion.svg layoutId="logo" ref={svg} width={size} height={size} viewBox="0 0 120 120" style={{ overflow: "visible" }}
      animate={done ? { filter: ["drop-shadow(0 0 0 rgba(214,168,79,0))", "drop-shadow(0 0 18px rgba(214,168,79,.8))", "drop-shadow(0 0 4px rgba(214,168,79,.3))"] } : {}}
      transition={{ duration: 1.4 }} aria-hidden>
      <Defs id={id} />
      {WIRES.map(([a, l, w, g]) => <path key={a} className="wire" d={a + tail(l)} stroke={`url(#${id}${g})`} strokeWidth={w} {...P} />)}
      <motion.circle cx="60" cy="91" r="4.6" fill="#6e4a17" initial={{ scale: 0 }} animate={{ scale: done ? 1 : 0 }} transition={{ type: "spring", stiffness: 260, damping: 12 }} />
    </motion.svg>
  );
}
