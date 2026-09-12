"use client";

import { useMemo, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { PublicConfig } from "@/lib/config";
import { cn } from "@/lib/utils";

/**
 * The sparse contact graph, made visible.
 *
 * This is the one picture that carries the claim: attention exists only on
 * edges, and measured contacts reweight those edges.
 *
 * The edge set is GENERATED FROM THE CONFIG, not read from a Hi-C file. It is
 * an illustration of the graph the builder produces at these settings, and the
 * caption says so — a synthetic picture presented as data would be exactly the
 * kind of thing this project's controls exist to catch.
 */

type Mode = "local" | "distal" | "held" | "drop";

interface Edge {
  i: number;
  j: number;
  strength: number; // detrended contact strength in [0, 1]
  local: boolean;
  held: boolean;
}

/** Small deterministic PRNG so the picture is identical on every render and machine. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildGraph(cfg: PublicConfig): Edge[] {
  const n = cfg.nodesPerSample;
  const rng = mulberry32(cfg.controlSeed);

  // Per-window affinity stands in for anchor strength — a handful of windows act
  // like CTCF anchors and collect most of the long-range edges, which is what
  // real detrended Hi-C looks like.
  const affinity = Array.from({ length: n }, () => rng() ** 2);
  for (let i = 0; i < n; i += 1) if (rng() > 0.94) affinity[i] = 0.85 + rng() * 0.15;

  const seen = new Set<string>();
  const edges: Edge[] = [];
  const push = (i: number, j: number, strength: number, local: boolean) => {
    const key = i < j ? `${i}:${j}` : `${j}:${i}`;
    if (seen.has(key)) return;
    seen.add(key);
    edges.push({ i: Math.min(i, j), j: Math.max(i, j), strength, local, held: false });
  };

  // local neighbours: always present, and the only thing left after structure dropout
  for (let i = 0; i < n; i += 1) {
    for (let d = 1; d <= cfg.localRadius; d += 1) {
      if (i + d < n) push(i, i + d, 0.9 - d * 0.12, true);
    }
  }

  // top-k distal, chosen on detrended strength so the picture is not just "near things touch"
  for (let i = 0; i < n; i += 1) {
    const candidates: { j: number; s: number }[] = [];
    for (let j = 0; j < n; j += 1) {
      const sep = Math.abs(i - j);
      if (sep < cfg.minSeparation) continue;
      const s = Math.sqrt(affinity[i] * affinity[j]) * (0.35 + rng() * 0.65);
      candidates.push({ j, s });
    }
    candidates.sort((a, b) => b.s - a.s);
    for (const c of candidates.slice(0, cfg.topKEdges)) push(i, c.j, c.s, false);
  }

  // hold out a fraction as prediction targets, using the control seed
  const holdRng = mulberry32(cfg.controlSeed + 1);
  for (const e of edges) if (!e.local && holdRng() < cfg.heldOutEdgeFrac) e.held = true;

  return edges;
}

/* Illustrative bias terms. Monotone and clearly captioned, not the trained functions. */
const bHiC = (c: number) => 3.2 * c;
const bDist = (sep: number) => -0.55 * Math.log2(sep + 1);

const VB = { w: 1000, h: 300 };
const TRACK_Y = 250;
const PAD_X = 24;

export function ContactGraph({ config, formulaHtml }: { config: PublicConfig; formulaHtml: string }) {
  const [mode, setMode] = useState<Mode>("distal");
  const [sel, setSel] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const n = config.nodesPerSample;
  const edges = useMemo(() => buildGraph(config), [config]);
  const xOf = (i: number) => PAD_X + (i / (n - 1)) * (VB.w - PAD_X * 2);

  const visible = useMemo(() => {
    if (mode === "drop" || mode === "local") return edges.filter((e) => e.local);
    return edges;
  }, [edges, mode]);

  /** Bias-only attention weights for the selected window, over its visible edges. */
  const weights = useMemo(() => {
    if (sel === null) return null;
    const mine = visible.filter((e) => (e.i === sel || e.j === sel) && !e.held);
    if (!mine.length) return null;
    const logits = mine.map((e) => bHiC(e.strength) + bDist(Math.abs(e.j - e.i)));
    const max = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return new Map(mine.map((e, k) => [`${e.i}:${e.j}`, exps[k] / sum]));
  }, [visible, sel]);

  const onPointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const frac = (event.clientX - rect.left) / rect.width;
    const x = frac * VB.w;
    const i = Math.round(((x - PAD_X) / (VB.w - PAD_X * 2)) * (n - 1));
    setSel(Math.min(n - 1, Math.max(0, i)));
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (step) {
      event.preventDefault();
      setSel((s) => Math.min(n - 1, Math.max(0, (s ?? Math.floor(n / 2)) + step)));
    } else if (event.key === "Home") {
      setSel(0);
    } else if (event.key === "End") {
      setSel(n - 1);
    } else if (event.key === "Escape") {
      setSel(null);
    }
  };

  const heldCount = edges.filter((e) => e.held).length;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <ToggleGroup
          type="single"
          value={mode}
          onValueChange={(v) => v && setMode(v as Mode)}
          variant="outline"
          size="sm"
        >
          <ToggleGroupItem value="local">local only</ToggleGroupItem>
          <ToggleGroupItem value="distal">+ top-{config.topKEdges} distal</ToggleGroupItem>
          <ToggleGroupItem value="held">held-out targets</ToggleGroupItem>
          <ToggleGroupItem value="drop">structure dropout</ToggleGroupItem>
        </ToggleGroup>
        <div className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
          <Badge variant="outline" className="font-mono text-[10px]">
            {n} windows
          </Badge>
          <Badge variant="outline" className="font-mono text-[10px]">
            {visible.length} edges
          </Badge>
        </div>
      </div>

      <div
        tabIndex={0}
        role="application"
        aria-label={`Sparse contact graph over ${n} genomic windows. Use left and right arrows to select a window and see which distal windows it attends to.`}
        onKeyDown={onKeyDown}
        className="focus-visible:ring-ring/70 rounded-xl border border-border/60 bg-card/30 p-2 focus-visible:outline-none focus-visible:ring-[3px]"
      >
        <svg
          ref={svgRef}
          viewBox={`0 0 ${VB.w} ${VB.h}`}
          preserveAspectRatio="xMidYMid meet"
          className="h-auto w-full"
          onPointerMove={onPointerMove}
          onPointerLeave={() => setSel(null)}
        >
          {visible.map((e) => {
            const sep = e.j - e.i;
            const x1 = xOf(e.i);
            const x2 = xOf(e.j);
            const apex = Math.max(24, TRACK_Y - Math.min(215, 26 + sep * 3.1));
            const touches = sel !== null && (e.i === sel || e.j === sel);
            const w = weights?.get(`${e.i}:${e.j}`);

            const highlighted = mode === "held" && e.held;
            const stroke = highlighted
              ? "var(--novel)"
              : e.local
                ? "var(--seq)"
                : "var(--struct)";

            let width = e.local ? 1 : 1.1 + e.strength * 1.5;
            let opacity = e.local ? 0.5 : 0.34;
            if (mode === "held") opacity = e.held ? 0.95 : 0.1;
            if (sel !== null) {
              if (touches && w !== undefined) {
                width = 1 + w * 16;
                opacity = 0.35 + w * 4;
              } else if (touches) {
                opacity = 0.5;
              } else {
                opacity = 0.07;
              }
            }

            return (
              <path
                key={`${e.i}:${e.j}`}
                d={`M ${x1} ${TRACK_Y} Q ${(x1 + x2) / 2} ${apex} ${x2} ${TRACK_Y}`}
                fill="none"
                stroke={stroke}
                strokeWidth={width}
                strokeLinecap="round"
                strokeDasharray={e.held && mode === "held" ? "7 5" : undefined}
                opacity={Math.min(1, opacity)}
                className={cn(
                  "transition-[opacity,stroke-width] duration-200",
                  touches && w !== undefined && w > 0.12 && "edge-pulse",
                )}
              />
            );
          })}

          <line
            x1={PAD_X}
            y1={TRACK_Y}
            x2={VB.w - PAD_X}
            y2={TRACK_Y}
            stroke="var(--border)"
            strokeWidth="1.5"
          />

          {Array.from({ length: n }, (_, i) => (
            <rect
              key={i}
              x={xOf(i) - 2}
              y={TRACK_Y - 5}
              width={4}
              height={10}
              rx={1}
              fill={sel === i ? "var(--primary)" : "var(--muted-foreground)"}
              opacity={sel === null ? 0.55 : sel === i ? 1 : 0.28}
              className="transition-opacity duration-150"
            />
          ))}

          {sel !== null ? (
            <text
              x={Math.min(VB.w - 120, Math.max(60, xOf(sel)))}
              y={TRACK_Y + 30}
              textAnchor="middle"
              fill="var(--primary)"
              fontSize="15"
              fontFamily="var(--font-mono)"
            >
              window {sel}
            </text>
          ) : null}
        </svg>
      </div>

      <div className="grid gap-4 md:grid-cols-[1.15fr_1fr]">
        <div className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <p>{CAPTION[mode](config, heldCount)}</p>
          <p className="text-[13px]">
            Hover the track, or focus it and use the arrow keys, to pick a window. Its edges are
            reweighted by the bias terms and thickened in proportion.{" "}
            <strong className="font-medium text-foreground">
              Only the bias terms are shown — the learned q·k score is not.
            </strong>{" "}
            The edge set here is generated from the training config, not from a measured Hi-C map.
          </p>
        </div>
        <div className="rounded-lg border border-border/60 bg-muted/20 px-3 py-3">
          <div className="overflow-x-auto text-sm" dangerouslySetInnerHTML={{ __html: formulaHtml }} />
        </div>
      </div>
    </div>
  );
}

const CAPTION: Record<Mode, (c: PublicConfig, held: number) => string> = {
  local: (c) =>
    `Sequential neighbours only, within ±${c.localRadius} bins. This is the graph a model sees when structure dropout fires, and it is all the 1D genome can offer on its own.`,
  distal: (c) =>
    `Each window also keeps its ${c.topKEdges} strongest distal partners, at least ${c.minSeparation} bins away. Strength is detrended against genomic separation first, otherwise the top-k would select nothing but near neighbours and the model would learn distance rather than structure.`,
  held: (c, held) =>
    `${held} edges (${Math.round(c.heldOutEdgeFrac * 100)}% of distal edges) are withheld from the encoder and used only as contact-prediction targets. Without this the contact head could read an edge from its own input and echo it back.`,
  drop: (c) =>
    `With probability ${c.structureDropout} the whole distal graph vanishes for a sample and the encoder must work from sequence alone. This is what makes the Hi-C-free claim testable rather than hopeful.`,
};
