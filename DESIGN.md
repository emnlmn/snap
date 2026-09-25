---
name: snap
description: Typed decisions from one pass — paper surface over a dark engine room
colors:
  paper: "#f4f4ee"
  paper-raised: "#ffffff"
  paper-sunk: "#ebebe3"
  ink: "#0a0d0b"
  ink-raised: "#0f1410"
  ink-panel: "#161c17"
  text: "#111611"
  text-soft: "#353c36"
  text-muted: "#555d56"
  ink-text: "#eaede7"
  ink-soft: "#aab2a9"
  ink-faint: "#838c83"
  crocodile: "#2a7a47"
  crocodile-deep: "#1d5e35"
  signal: "#b8ec5f"
  mint: "#7fd79b"
  sand: "#e2b877"
  heat: "#e59a52"
  line: "#deded4"
  line-ink: "#222a24"
typography:
  display:
    fontFamily: "Mona Sans"
    fontWeight: 740
    fontSize: "clamp(2.8rem, 7vw, 6rem)"
    lineHeight: 0.95
    letterSpacing: "-0.035em"
    fontVariation: "stretched 116%"
  headline:
    fontFamily: "Mona Sans"
    fontWeight: 710
    fontSize: "clamp(2rem, 3.9vw, 3.3rem)"
    lineHeight: 1.02
    letterSpacing: "-0.03em"
    fontVariation: "stretched 112%"
  title:
    fontFamily: "Mona Sans"
    fontWeight: 640
    fontSize: "18px"
    lineHeight: 1.3
  body:
    fontFamily: "Mona Sans"
    fontWeight: 400
    fontSize: "16px"
    lineHeight: 1.6
  label:
    fontFamily: "JetBrains Mono"
    fontWeight: 500
    fontSize: "12px"
    letterSpacing: "0.06em"
rounded:
  sm: "7px"
  md: "11px"
  lg: "16px"
  xl: "18px"
spacing:
  gutter: "28px"
  section: "128px"
  section-compact: "88px"
components:
  button-primary:
    background: "{colors.crocodile}"
    color: "#ffffff"
    borderRadius: "{rounded.md}"
    fontWeight: 620
  button-primary-ink:
    background: "{colors.signal}"
    color: "{colors.ink}"
    borderRadius: "{rounded.md}"
  button-ghost:
    background: "{colors.paper-raised}"
    borderColor: "{colors.line}"
    borderRadius: "{rounded.md}"
  card-surface:
    background: "{colors.paper-raised}"
    borderColor: "{colors.line}"
    borderRadius: "{rounded.lg}"
  console-window:
    background: "{colors.ink}"
    borderRadius: "{rounded.xl}"
---

# Design System: snap

## Overview

**Creative North Star: "The Instrument Panel"**

snap is a measurement instrument, not a chatbot — the site dresses like one. The reading surface is a warm paper field; wherever the engine itself is on stage (the live console, the pipeline anatomy, the wire format, the installer) the surface drops into a near-black engine room and the accent flips from crocodile green to signal lime. The split is literal: light is where you read about the product, dark is where you watch it work.

The system refuses the dev-tool landing formula (gradient hero, GIF terminal, icon-card grid). Proof replaces decoration: the hero *is* a recorded request resolving, the architecture diagram *is* a timed race between generated tokens and a single logits read, and every number on the page is a value the engine emitted.

**Key Characteristics:**
- Two zones only: paper for persuasion, ink for machinery — never mixed inside a component
- One accent per zone: crocodile `#2a7a47` on paper, signal lime `#b8ec5f` on ink
- Distributions rendered as real histograms, not metaphors
- Hairline `1px` dividers instead of boxed sections; corners 11–18px, never pill-soft
- Mono type carries all code, data, and measurement — it never decorates prose
- Every claim on the page is sourced from recorded `x_snap` output

## Colors

A warm-paper field with a single green accent; the dark bands run near-black green with a hot lime reserved for the decided answer and the running machine.

### Primary
- **Crocodile** (`#2a7a47`): the brand green — buttons, links, winner bars, deltas on light surfaces. Deep variant `#1d5e35` for hover and emphasized values.

### Secondary
- **Signal Lime** (`#b8ec5f`): exists only on ink. Marks the winning bar in every histogram, active states in the engine-room diagram, step numbers, and the primary CTA on dark. Its rarity is the point.

### Tertiary
- **Mint** (`#7fd79b`): secondary data on ink — numbers, booleans, `200 OK` statuses.
- **Sand** (`#e2b877`): string literals in dark JSON.
- **Heat** (`#e59a52`): the *conventional* pipeline's highlight color in the generate-vs-read diagram; also the residual-risk amber on light.

### Neutral
- **Paper** (`#f4f4ee`): page background.
- **Paper Raised** (`#ffffff`): cards, tables, the command pill.
- **Ink** (`#0a0d0b`) / **Ink Raised** (`#0f1410`) / **Ink Panel** (`#161c17`): the engine-room stack.
- **Text** (`#111611`) on paper; **Ink Text** (`#eaede7`) on ink. Muted tiers step down from each.
- **Line** (`#deded4`) hairlines on paper; **Line Ink** (`#222a24`) on dark.

### Named Rules
**The Signal Economy Rule.** Lime appears only on ink, and only where the machine has decided something: the winning histogram bar, the live model cell, the landed distribution, the step number, the dark CTA. Never as decoration, never on paper.

## Typography

**Display/Body:** Mona Sans (variable, self-hosted) — expanded width (stretched 116%) and heavy weight for display, normal width for text.
**Label/Mono:** JetBrains Mono — code, data, timings, micro-labels only.

**Character:** Editorial heft on top, instrumentation below. Display type is tight-tracked and wide; everything numeric is tabular mono.

### Hierarchy
- **Display** (740, clamp 2.8–6rem, .95, −.035em, stretched): the hero promise and the closing CTA only.
- **Headline** (710, clamp 2–3.3rem, 1.02, −.03em): section theses.
- **Title** (640, 18–24px): card and block headers.
- **Body** (400, 16px, 1.6): prose; ledes at 19px muted.
- **Label** (500 mono, 12px, +.06em, uppercase): pane headers, table heads, type tags, axes.

### Named Rules
**The Mono Is Earned Rule.** Mono appears only where the content is literally machine output — code, JSON, ms timings, probabilities, tab labels. Never for eyebrows over headings or decorative meta text in prose contexts.

## Layout

1200px container, 28px gutters, 128px section rhythm (88px compact). Two-column section headers (`head`): thesis left, supporting paragraph right, 64px column gap. Section hairline borders separate paper chapters; ink bands are full-bleed color changes, not bordered boxes. Breakpoints: 1100px (duel and step grids reflow), 880px (everything stacks to single column, tab lists become horizontal scroll), 520px (answer rows collapse to two-line cells).

## Elevation & Depth

Nearly flat. Depth exists in exactly two places: the hero console casts a real, deep green-black shadow (it is the object being demonstrated), and selected use-case tabs get a faint lift. Everything else is hairline-separated tonal layering — borders, not shadows.

### Shadow Vocabulary
- **Console** (`0 40px 80px -40px rgba(10,30,16,.55)` + inner top hairline): the product window only.
- **Tab lift** (`0 10px 24px -18px rgba(17,30,20,.35)`): selected state only.

### Named Rules
**The Flat-By-Default Rule.** Paper surfaces never carry ambient shadows. Depth is a spotlight reserved for the instrument.

## Shapes

Squircle-adjacent: 7px chips and mono cells, 11px buttons and tabs, 16px cards and panels, 18px the console and diagram frames. Hairline `1px` borders in `--line` / `--ink-line` define every container. No pills, no glass, no gradients-as-decoration (the scan line and token chips are the only gradients, and both are literal).

## Components

### Buttons
- **Shape:** 11px radius, 46px height (38px in nav), 620 weight.
- **Primary:** crocodile on paper, lime-on-ink inside dark bands; hover deepens one step.
- **Ghost:** white-on-paper or transparent-on-ink with hairline border; text-ink.

### Chips / Tabs
- Use-case tabs are full-sentence cards (title + summary), selected = white fill + hairline; on mobile they collapse to a horizontal pill scroll.
- Console tabs are quiet text buttons; selected = ink-panel fill + inset hairline ring.

### Cards / Containers
- 16px radius, white fill on paper, hairline border, no shadow. Dark variants use `ink-panel` on `ink-raised` steps.
- The console is the signature container: 18px radius, window chrome (dots, tabs, endpoint header, status bar), 6-column answer rows.

### The Answer Row (signature component)
One row per question inside the console: mono name + type tag, a mini histogram of the full option distribution, the decided answer in bold white, and its probability in tabular mono. During playback all histograms hold at the baseline; they rise in the same frame the ms counter lands. Hovering a row expands its option-by-option breakdown in the inspect bar.

### Inputs / Fields
- The copyable command pill: white field, mono text, `$` prompt in green, embedded copy button.
- Copy buttons confirm inline ("copied"), never toast.

### Tables
- Hairline-row tables (threat model, benchmarks, compare) with mono uppercase heads; winners marked by weight and crocodile, not by badges.

## Do's and Don'ts

### Do:
- **Do** keep lime exclusively on ink and exclusively on decided output.
- **Do** show distributions, never bare answers — every probability gets a bar.
- **Do** label recordings with model, quantization, and hardware.
- **Do** keep section rhythm generous (128px) — the density lives inside the components.
- **Do** let the JSON and histograms carry the proof; copy stays plainspoken.

### Don't:
- **Don't** use gradient text, glass blur, glow shadows, or floating emoji-icon tiles.
- **Don't** put dark components on paper or light panels inside ink bands — the zones don't mix.
- **Don't** invent numbers, customers, or latency — everything rendered is recorded.
- **Don't** use mono for decoration or display type under 12px for readable copy.
- **Don't** add a second accent hue to either zone.
