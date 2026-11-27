import { useEffect, useRef } from "react";

// Slowly drifting layered waveforms behind everything, like a live spectrogram trace.
export default function Waves() {
  const ref = useRef();
  useEffect(() => {
    const c = ref.current, g = c.getContext("2d");
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf, w, h;
    const size = () => {
      const r = devicePixelRatio || 1;
      w = c.width = innerWidth * r; h = c.height = innerHeight * r;
    };
    const lines = Array.from({ length: 9 }, (_, i) => ({
      y: 0.18 + i * 0.085, amp: 0.025 + (i % 3) * 0.014, k: 1.4 + (i % 4) * 0.55,
      speed: 0.06 + (i % 5) * 0.025, phase: i * 1.7, alpha: 0.16 + (i % 3) * 0.07,
    }));
    const frame = (ms) => {
      const t = ms / 1000;
      g.clearRect(0, 0, w, h);
      for (const L of lines) {
        g.beginPath();
        for (let x = 0; x <= w; x += 6) {
          const u = x / w;
          const env = 0.55 + 0.45 * Math.sin(u * Math.PI * 2 + t * L.speed * 2 + L.phase); // breathing loudness
          const y = L.y * h + h * L.amp * env * (
            Math.sin(u * L.k * 6.283 + t * L.speed * 6 + L.phase) +
            0.35 * Math.sin(u * L.k * 17 - t * L.speed * 9));
          x ? g.lineTo(x, y) : g.moveTo(x, y);
        }
        g.strokeStyle = `rgba(176, 132, 52, ${L.alpha})`;
        g.lineWidth = (1.3 + (L.alpha > 0.25) * 0.8) * (devicePixelRatio || 1);
        g.stroke();
      }
      if (!still) raf = requestAnimationFrame(frame);
    };
    const resize = () => { size(); if (still) frame(0); };
    size(); addEventListener("resize", resize);
    raf = requestAnimationFrame(frame);
    return () => { cancelAnimationFrame(raf); removeEventListener("resize", resize); };
  }, []);
  return <canvas ref={ref} className="waves" aria-hidden />;
}
