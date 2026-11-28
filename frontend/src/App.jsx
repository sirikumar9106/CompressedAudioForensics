import { AnimatePresence, motion, useMotionValue, useSpring } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import Logo, { IntroMark } from "./Logo.jsx";
import Waves from "./Waves.jsx";
import Details, { TipLayer } from "./Details.jsx";

export const flow = { type: "spring", stiffness: 90, damping: 19, mass: 1.1 };   // water-like settle
const ORDER = { home: 0, result: 1, details: 2 };
const fromHash = () => ({ "#/result": "result", "#/details": "details" })[location.hash] || "home";
const go = (p) => { location.hash = p === "home" ? "" : "/" + p; };

// Pages sit on one horizontal track; forward slides left, back slides right.
const slide = {
  enter: (d) => ({ x: `${d * 105}vw`, scale: 0.9, rotateY: d * -10 }),
  center: { x: 0, scale: 1, rotateY: 0 },
  exit: (d) => ({ x: `${d * -105}vw`, scale: 0.9, rotateY: d * 10 }),
};

const ICON = {
  phone: <svg viewBox="0 0 24 24" fill="#fff"><path d="M6.6 10.8a15.2 15.2 0 0 0 6.6 6.6l2.2-2.2c.3-.3.7-.4 1-.2 1.1.4 2.3.6 3.6.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1A17 17 0 0 1 3 4c0-.6.4-1 1-1h3.5c.6 0 1 .4 1 1 0 1.3.2 2.5.6 3.6.1.3 0 .7-.2 1z" /></svg>,
  mail: <svg viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinejoin="round"><rect x="3" y="5" width="18" height="14" rx="2" /><path d="m3.5 6 8.5 7 8.5-7" /></svg>,
};
// Brand-coloured tiles with initials: generic marks, not the companies' logos.
const SOURCES = [
  ["unknown", "I don't know", "?", ""], ["whatsapp", "WhatsApp", "WA", "#25D366"], ["telegram", "Telegram", "TG", "#229ED9"],
  ["signal", "Signal", "SG", "#3A76F0"], ["messenger", "Facebook Messenger", "FB", "linear-gradient(135deg,#00B2FF,#006AFF 60%,#A033FF)"],
  ["instagram", "Instagram", "IG", "linear-gradient(45deg,#F9B649,#F2603A 35%,#D62976 65%,#4F5BD5)"], ["viber", "Viber", "VB", "#7360F2"],
  ["wechat", "WeChat", "WC", "#07C160"], ["discord", "Discord", "DC", "#5865F2"], ["snapchat", "Snapchat", "SC", "#F2D600"],
  ["reddit", "Reddit", "RD", "#FF4500"], ["gmail", "E-mail attachment", ICON.mail, "#a88550"], ["phone", "Phone call or voicemail", ICON.phone, "#34C759"],
  ["other", "Other messenger", "··", "#8a7550"]];
const Chip = ({ s }) => s[0] === "unknown"
  ? <span className="chip q">?</span>
  : <span className="chip" style={{ background: s[3] }}>{s[2]}</span>;

export default function App() {
  const [page, setPage] = useState("home");
  const [dir, setDir] = useState(1);
  const [intro, setIntro] = useState(!matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [form, setForm] = useState({ file: null, model: "voxtrace-wild", src: SOURCES[0] });   // kept when going Back
  const newCheck = () => { setForm((f) => ({ ...f, file: null })); go("home"); };              // New check clears the audio
  const resRef = useRef(null); resRef.current = res;
  const pageRef = useRef(page);

  useEffect(() => {
    if (location.hash) history.replaceState(null, "", location.pathname);   // results live in memory only
    const sync = () => {
      let p = fromHash();
      if (p !== "home" && !resRef.current) { p = "home"; history.replaceState(null, "", location.pathname); }
      setDir(Math.sign(ORDER[p] - ORDER[pageRef.current]) || 1);
      pageRef.current = p; setPage(p); scrollTo(0, 0);
    };
    addEventListener("hashchange", sync);
    const t = setTimeout(() => setIntro(false), 4700);
    return () => { removeEventListener("hashchange", sync); clearTimeout(t); };
  }, []);

  const check = async (form) => {
    setBusy(true);
    try {
      const [r] = await Promise.all([fetch("/api/analyze", { method: "POST", body: form }), new Promise((ok) => setTimeout(ok, 2400))]);
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "VoxTrace could not read this file. Try another audio format.");
      setRes(j); resRef.current = j; go("result");
      return null;
    } catch (e) {
      return e.message === "Failed to fetch" ? "The detector is not running. Start run_app.bat and try again." : e.message;
    } finally { setBusy(false); }
  };

  return (
    <TipLayer>
      <Waves />
      <AnimatePresence>
        {intro && (
          <motion.div className="intro" key="intro" onClick={() => setIntro(false)} exit={{ backdropFilter: "blur(0px)", backgroundColor: "rgba(250,244,231,0)" }} transition={{ duration: 0.9 }}>
            <IntroMark size={190} />
            <motion.h1 layoutId="word" className="word big">
              {[..."VoxTrace"].map((c, i) => (
                <motion.span key={i} className={i === 0 || i === 3 ? "vt" : ""} style={{ display: "inline-block" }} initial={{ opacity: 0, y: 18, filter: "blur(6px)" }}
                  animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ delay: 3.2 + i * 0.07, duration: 0.8, ease: [0.2, 0.9, 0.25, 1] }}>{c}</motion.span>
              ))}
            </motion.h1>
            <motion.p className="muted" initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 3.9 } }} exit={{ opacity: 0, transition: { duration: 0.2 } }}>Voice authenticity check for compressed audio</motion.p>
          </motion.div>
        )}
      </AnimatePresence>

      {!intro && (
        <motion.header layout className={"brand " + (page === "home" ? "home" : "side")} transition={flow} onClick={() => page !== "home" && go("home")}>
          <motion.span layout className="mark" transition={flow}><Logo size={page === "home" ? 50 : 32} layoutId="logo" /></motion.span>
          <motion.h1 layoutId="word" className="word" transition={flow}><span className="vt">V</span>ox<span className="vt">T</span>race</motion.h1>
        </motion.header>
      )}

      <AnimatePresence>
        {!intro && page !== "home" && (
          <motion.nav className="nav" key="nav" initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 40, opacity: 0 }} transition={flow}>
            <button className="glass-btn round" onClick={() => go(page === "details" ? "result" : "home")} title="Back">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M15 6l-6 6 6 6" /></svg><span>Back</span>
            </button>
            <button className="glass-btn round" onClick={newCheck} title="Start a new check">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 12a8 8 0 1 1-2.34-5.66" /><path d="M20 4v5h-5" /></svg><span>New check</span>
            </button>
          </motion.nav>
        )}
      </AnimatePresence>

      <div className="stage">
        <AnimatePresence initial={false} custom={dir} mode="popLayout">
          <motion.main key={page} className={"page pg-" + page} custom={dir} variants={slide}
            initial={intro ? false : "enter"} animate="center" exit="exit" transition={flow}>
            {page === "home" && <Home onCheck={check} show={!intro} form={form} setForm={setForm} />}
            {page === "result" && res && <Result r={res} onNew={newCheck} />}
            {page === "details" && res && <Details r={res} />}
          </motion.main>
        </AnimatePresence>
      </div>

      <AnimatePresence>{busy && <Busy key="busy" />}</AnimatePresence>
    </TipLayer>
  );
}

/* ---------------- a streak of light that shoots out of a pressed button ---------------- */
export function comet(e) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const r = e.currentTarget.getBoundingClientRect(), c = document.createElement("i");
  c.className = "comet"; c.style.left = e.clientX - 150 + "px"; c.style.top = r.top + r.height / 2 + "px";
  document.body.appendChild(c);
  c.animate([{ transform: "translateX(0) scaleX(.2)", opacity: 0 }, { opacity: 1, offset: 0.12 },
    { transform: `translateX(${innerWidth - e.clientX + 200}px) scaleX(1)`, opacity: 0 }],
    { duration: 1100, easing: "cubic-bezier(.25,.7,.3,1)" }).onfinish = () => c.remove();
}

/* ---------------- a card that lifts off the page and leans towards the pointer ---------------- */
export function Tilt({ children, max = 3, style, ...rest }) {
  const spring = { stiffness: 120, damping: 18 };
  const rx = useSpring(useMotionValue(0), spring), ry = useSpring(useMotionValue(0), spring);
  const move = (e) => {
    if (e.target.closest(".menu")) return;
    const r = e.currentTarget.getBoundingClientRect();
    ry.set(((e.clientX - r.left) / r.width - 0.5) * 2 * max); rx.set(-((e.clientY - r.top) / r.height - 0.5) * 2 * max);
  };
  return (
    <motion.section {...rest} style={{ ...style, rotateX: rx, rotateY: ry, transformPerspective: 1200 }}
      onPointerMove={move} onPointerLeave={() => { rx.set(0); ry.set(0); }}>
      {children}
    </motion.section>
  );
}

/* ---------------- home: one card ---------------- */
function Home({ onCheck, show, form, setForm }) {
  const { file, model, src } = form;
  const set = (k) => (v) => setForm((f) => ({ ...f, [k]: v }));
  const setFile = set("file"), setModel = set("model"), setSrc = set("src");
  const [models, setModels] = useState([]);
  const [notes, setNotes] = useState({});
  const [open, setOpen] = useState(false);
  const [over, setOver] = useState(false);
  const [err, setErr] = useState("");
  const [ready, setReady] = useState(null);
  const input = useRef();

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((l) => { setModels(l); if (l.length && !l.some((m) => m.id === model)) setModel(l[0].id); }).catch(() => {});
    fetch("/api/platforms").then((r) => r.json()).then((l) => { setNotes(Object.fromEntries(l.map((p) => [p.id, p.note]))); setReady(true); }).catch(() => setReady(false));
    const close = () => setOpen(false);
    addEventListener("click", close);
    return () => removeEventListener("click", close);
  }, []);

  const pick = (f) => { if (f) { setFile(f); setErr(""); } };
  const submit = async () => {
    const fd = new FormData(); fd.append("file", file); fd.append("platform", src[0]); fd.append("model", model);
    setErr((await onCheck(fd)) || "");
  };

  return (
    <Tilt className="card glass pop" initial={{ y: 40, opacity: 0 }} animate={show ? { y: 0, opacity: 1 } : {}} transition={{ ...flow, delay: 0.2 }}>
      <p className="lede">Find out whether a voice note is a real person or an AI voice clone.</p>

      <div className={"drop" + (over ? " over" : "") + (file ? " has" : "")} tabIndex={0} role="button"
        onClick={() => input.current.click()} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), input.current.click())}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]); }}>
        <input ref={input} type="file" hidden accept="audio/*,video/*,.opus,.ogg,.m4a,.aac,.webm" onChange={(e) => pick(e.target.files[0])} />
        <AnimatePresence mode="popLayout" initial={false}>
          {file ? (
            <motion.div key="f" className="picked" initial={{ y: 16, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: -16, opacity: 0 }} transition={flow}>
              <span className="bars">{[0, 1, 2, 3, 4].map((i) => <i key={i} style={{ animationDelay: `${-i * 0.22}s` }} />)}</span>
              <span className="fn">{file.name}</span><span className="fs">{(file.size / 1048576).toFixed(2)} MB</span>
              <span className="swap">Change</span>
            </motion.div>
          ) : (
            <motion.div key="e" initial={{ y: 16, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: -16, opacity: 0 }} transition={flow}>
              <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#a37a32" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M12 15V4M7.5 8.5 12 4l4.5 4.5" /><path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15" /></svg>
              <b>Drop your audio here or click to browse</b>
              <span>Voice notes, recordings or videos, up to 10 minutes</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence initial={false}>{file && <Player key={file.name + file.size} file={file} />}</AnimatePresence>

      <label className="lbl">Detection model</label>
      <div className="seg glass-in">
        {models.map((m) => (
          <button key={m.id} type="button" className={m.id === model ? "on" : ""} onClick={() => setModel(m.id)} title={m.desc}>
            {m.id === model && <motion.span layoutId="pill" className="pill" transition={flow} />}
            <b>{m.name}</b><small>{m.id.endsWith("wild") ? "Real-world audio" : "Lab and modern TTS"}</small>
          </button>
        ))}
        {!models.length && <span className="muted pad">{ready === false ? "Detector offline" : "Loading models…"}</span>}
      </div>

      <label className="lbl">Where did you receive it?</label>
      <div className="select" onClick={(e) => e.stopPropagation()}>
        <button type="button" className="sel glass-btn" onClick={() => setOpen(!open)} aria-expanded={open}>
          <Chip s={src} /><span>{src[1]}</span>
          <motion.svg animate={{ rotate: open ? 180 : 0 }} transition={flow} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M6 9l6 6 6-6" /></motion.svg>
        </button>
        <AnimatePresence>
          {open && (
            <motion.ul className="menu" role="listbox" style={{ originY: 0 }}
              initial={{ height: 0, opacity: 0, scaleY: 0.92 }} animate={{ height: "auto", opacity: 1, scaleY: 1 }}
              exit={{ height: 0, opacity: 0, scaleY: 0.95 }} transition={{ type: "spring", stiffness: 70, damping: 16, mass: 1 }}>
              {SOURCES.map((s, i) => (
                <motion.li key={s[0]} role="option" aria-selected={s === src} className={s === src ? "act" : ""}
                  initial={{ opacity: 0, y: -10, rotateX: -35 }} animate={{ opacity: 1, y: 0, rotateX: 0 }}
                  transition={{ type: "spring", stiffness: 80, damping: 11, delay: i * 0.035 }}
                  onClick={() => { setSrc(s); setOpen(false); }}>
                  <Chip s={s} /><div><b>{s[1]}</b><small>{notes[s[0]] || ""}</small></div>
                </motion.li>
              ))}
            </motion.ul>
          )}
        </AnimatePresence>
      </div>

      <button className="cta" disabled={!file} onClick={submit} onPointerDown={comet}
        onPointerMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`); }}>
        Check authenticity
      </button>
      <AnimatePresence>{err && <motion.p className="err" initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}>{err}</motion.p>}</AnimatePresence>
      <p className="foot">Audio is analysed on this computer and deleted straight away.</p>
    </Tilt>
  );
}

/* ---------------- listen to the chosen file before checking it ---------------- */
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
function Player({ file }) {
  const a = useRef(), [url] = useState(() => URL.createObjectURL(file));
  const [on, setOn] = useState(false), [t, setT] = useState(0), [dur, setDur] = useState(0);
  useEffect(() => () => URL.revokeObjectURL(url), [url]);
  const seek = (e) => { const r = e.currentTarget.getBoundingClientRect(); if (dur) a.current.currentTime = ((e.clientX - r.left) / r.width) * dur; };
  return (
    <motion.div className="player" initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={flow}>
      <audio ref={a} src={url} preload="metadata" onLoadedMetadata={(e) => setDur(e.target.duration || 0)}
        onTimeUpdate={(e) => setT(e.target.currentTime)} onPlay={() => setOn(true)} onPause={() => setOn(false)} onEnded={() => setOn(false)} />
      <button type="button" className="glass-btn play" onClick={() => (on ? a.current.pause() : a.current.play())} aria-label={on ? "Pause" : "Play"}>
        {on ? <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1.2" /><rect x="14" y="5" width="4" height="14" rx="1.2" /></svg>
            : <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" /></svg>}
      </button>
      <div className="track" onClick={seek}><motion.i animate={{ width: dur ? `${(t / dur) * 100}%` : 0 }} transition={{ duration: 0.25, ease: "linear" }} /></div>
      <span className="time">{clock(t)} / {dur ? clock(dur) : "–:––"}</span>
    </motion.div>
  );
}

/* ---------------- authenticating overlay ---------------- */
const STEPS = ["Decoding the audio", "Listening in 4-second windows", "Comparing against real and cloned voices", "Weighing the evidence"];
function Busy() {
  const [n, setN] = useState(1);
  useEffect(() => { const t = setInterval(() => setN((k) => (k % 4) + 1), 380); return () => clearInterval(t); }, []);
  const [step, setStep] = useState(0);
  useEffect(() => { const t = setInterval(() => setStep((k) => Math.min(k + 1, STEPS.length - 1)), 1500); return () => clearInterval(t); }, []);
  return (
    <motion.div className="busy" role="status" aria-live="polite"
      initial={{ backdropFilter: "blur(0px)", backgroundColor: "rgba(250,244,231,0)" }}
      animate={{ backdropFilter: "blur(14px)", backgroundColor: "rgba(250,244,231,.55)" }}
      exit={{ backdropFilter: "blur(0px)", backgroundColor: "rgba(250,244,231,0)" }} transition={{ duration: 0.7 }}>
      <motion.div className="busy-in" initial={{ scale: 0.7, y: 20, opacity: 0 }} animate={{ scale: 1, y: 0, opacity: 1 }} exit={{ scale: 0.85, opacity: 0 }} transition={flow}>
        <Logo size={128} mode="busy" />
        <h2>Authenticating<span className="dots">{".".repeat(n)}</span></h2>
        <AnimatePresence mode="wait">
          <motion.p key={step} initial={{ y: 8, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: -8, opacity: 0 }} transition={{ duration: 0.35 }}>{STEPS[step]}</motion.p>
        </AnimatePresence>
      </motion.div>
    </motion.div>
  );
}

/* ---------------- result page ---------------- */
function Count({ to }) {
  const [v, setV] = useState(0);
  useEffect(() => {
    let raf; const t0 = performance.now();
    const step = (t) => { const k = Math.min((t - t0) / 1400, 1); setV(Math.round(to * (1 - Math.pow(1 - k, 3)))); if (k < 1) raf = requestAnimationFrame(step); };
    raf = requestAnimationFrame(step); return () => cancelAnimationFrame(raf);
  }, [to]);
  return v;
}

function Result({ r, onNew }) {
  const human = r.p_real >= 0.5, pct = Math.round((human ? r.p_real : r.p_fake) * 100);
  const C = 2 * Math.PI * 92;
  const tone = r.level === "real" ? "real" : r.level === "fake" ? "fake" : "warn";
  const cal = r.calibration === "unknown" ? "Not stated" : r.calibration === "clean" ? "Uncompressed" : r.calibration.replace("k", " kbps");
  return (
    <section className={"result " + tone}>
      <div className="dial">
        <svg viewBox="0 0 220 220" width="220" height="220">
          <defs>
            <linearGradient id="dr" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#e9c47a" /><stop offset="1" stopColor="#8a5f1c" /></linearGradient>
            <linearGradient id="df" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#e8956a" /><stop offset="1" stopColor="#9c3b22" /></linearGradient>
          </defs>
          <circle cx="110" cy="110" r="92" fill="none" stroke="#e8dcc2" strokeWidth="12" />
          <motion.circle cx="110" cy="110" r="92" fill="none" stroke={human ? "url(#dr)" : "url(#df)"} strokeWidth="12" strokeLinecap="round"
            strokeDasharray={C} initial={{ strokeDashoffset: C }} animate={{ strokeDashoffset: C * (1 - pct / 100) }}
            transition={{ duration: 1.5, ease: [0.16, 0.84, 0.3, 1], delay: 0.3 }} transform="rotate(-90 110 110)" />
        </svg>
        <div className="num"><b><Count to={pct} />%</b><span>{human ? "human" : "AI-generated"}</span></div>
      </div>
      <h2 className="verdict">{r.verdict}</h2>
      <p className="muted">
        {r.suspicious_span && r.level === "warn"
          ? `Mostly human-sounding, but ${r.suspicious_span[0]}–${r.suspicious_span[1]} s scores strongly as AI-generated.`
          : human ? `VoxTrace is ${pct}% confident this is a real person speaking.` : `VoxTrace is ${pct}% confident this voice was generated by AI.`}
      </p>
      <div className="facts">
        <span><small>Model</small>{r.model}</span><span><small>Source</small>{r.platform}</span>
        <span><small>Audio quality</small>{cal}</span><span><small>Length</small>{r.file.decoded_seconds} s</span>
      </div>
      {r.level !== "real" && <ul className="advice">{r.advice.slice(0, 3).map((a) => <li key={a}>{a}</li>)}</ul>}
      <div className="actions">
        <button className="cta" onPointerDown={comet} onClick={() => go("details")}>Check the details</button>
        <button className="glass-btn ghost" onClick={onNew}>Check another file</button>
      </div>
    </section>
  );
}
