import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import Logo from "./Logo.jsx";

const flow = { type: "spring", stiffness: 90, damping: 19, mass: 1.1 };

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
