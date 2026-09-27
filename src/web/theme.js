// snap playground — theme. "system" follows the OS, light/dark pin it; the
// choice sticks per browser. Loaded in <head> so the first paint already has
// the right tokens; :root[data-theme] carries the resolved theme, and a
// `themechange` event tells canvases to re-read them.
"use strict";
(() => {
  const ORDER = ["system", "light", "dark"];
  const root = document.documentElement;
  const mq = matchMedia("(prefers-color-scheme: dark)");
  let pref = "system";
  try { pref = localStorage.getItem("snap-theme") || "system"; } catch { /* storage blocked: follow the OS */ }

  function apply() {
    const dark = pref === "dark" || (pref === "system" && mq.matches);
    root.dataset.theme = dark ? "dark" : "light";
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", dark ? "#0a0d0b" : "#f4f4ee");
    document.querySelectorAll("[data-set-theme]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.setTheme === pref));
    dispatchEvent(new Event("themechange"));
  }

  // a swap lands every colour in one frame (.theming holds all transitions:
  // controls easing their own colours while the page snaps is the flicker);
  // the view transition then draws the motion — the new theme opens as a
  // circle from the switch that was pressed, or cross-fades when the OS
  // flipped it or the reader asked for less motion
  const reduce = matchMedia("(prefers-reduced-motion: reduce)");
  function swap(from) {
    const was = root.dataset.theme;
    root.classList.add("theming");
    const done = () => requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove("theming")));
    if (!document.startViewTransition) { apply(); return done(); }
    const t = document.startViewTransition(apply);
    t.ready.then(() => {
      if (root.dataset.theme === was) return; // same resolved theme: only the switch moved
      const pseudoElement = "::view-transition-new(root)";
      if (from && !reduce.matches) {
        const [x, y] = from, r = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
        root.animate({ clipPath: [`circle(0 at ${x}px ${y}px)`, `circle(${r}px at ${x}px ${y}px)`] },
          { duration: 560, easing: "cubic-bezier(.16, 1, .3, 1)", pseudoElement });
      } else {
        root.animate({ opacity: [0, 1] }, { duration: reduce.matches ? 160 : 280, easing: "ease-out", pseudoElement });
      }
    }).catch(() => {});
    t.finished.finally(done);
  }

  apply();
  mq.addEventListener("change", () => swap(null));
  document.addEventListener("DOMContentLoaded", apply);
  // picking the active option cycles — on narrow screens only it is shown
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-set-theme]");
    if (!b) return;
    const t = b.dataset.setTheme;
    pref = t === pref ? ORDER[(ORDER.indexOf(t) + 1) % ORDER.length] : t;
    try { localStorage.setItem("snap-theme", pref); } catch { /* not persisted */ }
    const r = b.getBoundingClientRect();
    swap([r.left + r.width / 2, r.top + r.height / 2]);
  });
})();
