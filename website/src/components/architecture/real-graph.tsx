"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import sample from "@/content/graph-sample.json";
import { cn } from "@/lib/utils";

/**
 * One REAL training sample's graph, as each arm's encoder receives it.
 *
 * Unlike ContactGraph, nothing here is generated: graph-sample.json is written
 * by scripts/export_graph_sample.py, which runs the built GM12878 data through
 * the same `build_sample` the training loop calls. Re-run that script after any
 * change to the graph builder, or this picture goes stale.
 *
 * Left: the contact map. Above the diagonal is the measured, detrended
 * strength; below it is what the selected arm actually conditions attention
 * on. Right: the same edges as arcs over the 128 windows, conditioning edges
 * above the track and held-out targets below it.
 */

type Triple = [number, number, number];

interface Sample {
  source: { cell_line: string; chrom: string; start_bin: number; start_bp: number; bin_size: number };
  n: number;
  local_radius: number;
  matrix: Triple[];
  targets: Triple[];
  arms: Record<string, Triple[]>;
}

const DATA = sample as unknown as Sample;

const ARMS = [
  { id: "full", label: "full" },
  { id: "b3_shuffled_hic", label: "B3 shuffled" },
  { id: "b2_distance_only", label: "B2 distance" },
  { id: "b1_random_graph", label: "B1 random" },
  { id: "b0_dna_only", label: "B0 DNA only" },
] as const;

type ArmId = (typeof ARMS)[number]["id"];

const CAPTION: Record<ArmId, string> = {
  full:
    "The real graph: each window's strongest detrended partners, carrying their measured strengths into the attention bias.",
  b3_shuffled_hic:
    "The same edges as full, with their strength values permuted among them. The topology was still picked from the real map, so B3 keeps most of the Hi-C signal and only the magnitudes are scrambled. Full against B3 isolates the value of contact strength, not of structure.",
  b2_distance_only:
    "The same edges again, every strength flattened to 0.5. Like B3, the topology is the real top-k selection; only the separation term of the bias can tell edges apart.",
  b1_random_graph:
    "The same number of distal edges, placed at random (never on a held-out target), with strengths borrowed from the real ones. This is the control that removes the measured topology itself.",
  b0_dna_only:
    "No distal edges at all, only the sequential backbone. B0 also trains no contact head, so its contact numbers need a frozen-embedding probe to mean anything.",
};

function pearson(a: number[], b: number[]): number | null {
  if (a.length < 2) return null;
  const ma = a.reduce((s, x) => s + x, 0) / a.length;
  const mb = b.reduce((s, x) => s + x, 0) / b.length;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let k = 0; k < a.length; k += 1) {
    num += (a[k] - ma) * (b[k] - mb);
    da += (a[k] - ma) ** 2;
    db += (b[k] - mb) ** 2;
  }
  return da > 0 && db > 0 ? num / Math.sqrt(da * db) : null;
}

/** Strength in (0, 1) -> opacity. 0.5 is O/E = 1, so only enrichment lights up. */
const heat = (s: number) => Math.min(1, Math.max(0.04, (s - 0.4) / 0.45));

const CELL = 4;
const ARC = { w: 1000, h: 360, track: 220, pad: 18 };

export function RealGraph() {
  const { n, local_radius: radius, source } = DATA;
  const [arm, setArm] = useState<ArmId>("full");
  const [showTargets, setShowTargets] = useState(true);
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const measured = useMemo(() => {
    const m = new Map<number, number>();
    for (const [i, j, s] of DATA.matrix) m.set(i * n + j, s);
    return m;
  }, [n]);

  const targetKeys = useMemo(() => new Set(DATA.targets.map(([i, j]) => i * n + j)), [n]);
  const edges = DATA.arms[arm];

  const given = useMemo(() => {
    const m = new Map<number, number>();
    for (const [i, j, s] of edges) m.set(i * n + j, s);
    return m;
  }, [edges, n]);

  const stats = useMemo(() => {
    const fullKeys = new Set(DATA.arms.full.map(([i, j]) => i * n + j));
    const shared = edges.filter(([i, j]) => fullKeys.has(i * n + j)).length;
    const pairs = edges.filter(([i, j]) => measured.has(i * n + j));
    const r = pearson(
      pairs.map(([, , s]) => s),
      pairs.map(([i, j]) => measured.get(i * n + j) ?? 0),
    );
    return { shared: edges.length ? shared / edges.length : null, r };
  }, [edges, measured, n]);

  // The contact map. Redrawn on arm, target toggle, or selected window.
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const css = getComputedStyle(canvas);
    const color = (name: string) => css.getPropertyValue(name).trim() || "#888";
    const struct = color("--struct");
    const seq = color("--seq");
    const novel = color("--novel");
    const border = color("--border");
    const primary = color("--primary");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cell = (row: number, col: number, fill: string, alpha: number) => {
      ctx.globalAlpha = alpha;
      ctx.fillStyle = fill;
      ctx.fillRect(col * CELL, row * CELL, CELL, CELL);
    };

    // upper triangle: what was measured
    for (const [i, j, s] of DATA.matrix) cell(i, j, struct, heat(s));

    // lower triangle: what this arm hands the encoder
    for (let i = 0; i < n; i += 1) {
      for (let d = 1; d <= radius && i + d < n; d += 1) cell(i + d, i, seq, 0.55);
    }
    for (const [i, j, s] of edges) cell(j, i, struct, heat(s));
    if (showTargets) for (const [i, j] of DATA.targets) cell(j, i, novel, 0.95);

    ctx.globalAlpha = 1;
    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(n * CELL, n * CELL);
    ctx.stroke();

    if (sel !== null) {
      ctx.globalAlpha = 0.35;
      ctx.fillStyle = primary;
      ctx.fillRect(0, sel * CELL, n * CELL, CELL);
      ctx.fillRect(sel * CELL, 0, CELL, n * CELL);
      ctx.globalAlpha = 1;
    }
  }, [edges, showTargets, sel, n, radius]);

  const onMatrixMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const col = Math.floor(((event.clientX - rect.left) / rect.width) * n);
    const row = Math.floor(((event.clientY - rect.top) / rect.height) * n);
    if (row < 0 || col < 0 || row >= n || col >= n || row === col) {
      setHover(null);
      return;
    }
    setHover({ i: Math.min(row, col), j: Math.max(row, col) });
    setSel(row);
  };

  const onArcMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * ARC.w;
    const i = Math.round(((x - ARC.pad) / (ARC.w - ARC.pad * 2)) * (n - 1));
    setSel(Math.min(n - 1, Math.max(0, i)));
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (step) {
      event.preventDefault();
      setSel((s) => Math.min(n - 1, Math.max(0, (s ?? Math.floor(n / 2)) + step)));
    } else if (event.key === "Escape") {
      setSel(null);
    }
  };

  const xOf = (i: number) => ARC.pad + (i / (n - 1)) * (ARC.w - ARC.pad * 2);
  const arcPath = (i: number, j: number, below: boolean) => {
    const x1 = xOf(i);
    const x2 = xOf(j);
    // Height proportional to separation, never capped: a cap flattens every
    // long-range edge onto one dome and hides exactly the range that matters.
    const rise = 8 + ((j - i) / (n - 1)) * (below ? 250 : 390);
    const apex = below ? ARC.track + rise : ARC.track - rise;
    return `M ${x1} ${ARC.track} Q ${(x1 + x2) / 2} ${apex} ${x2} ${ARC.track}`;
  };

  const hoverKey = hover ? hover.i * n + hover.j : null;
  const sepKb = hover ? ((hover.j - hover.i) * source.bin_size) / 1000 : 0;
  const status =
    hoverKey === null
      ? null
      : targetKeys.has(hoverKey)
        ? "held-out target (predicted, never conditioned on)"
        : given.has(hoverKey)
          ? `conditioning edge, given strength ${given.get(hoverKey)?.toFixed(2)}`
          : hover && hover.j - hover.i <= radius
            ? "local backbone edge"
            : "not an edge for this arm";
  const mb = (bp: number) => (bp / 1e6).toFixed(2);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <ToggleGroup
          type="single"
          value={arm}
          onValueChange={(v) => v && setArm(v as ArmId)}
          variant="outline"
          size="sm"
          className="flex-wrap"
        >
          {ARMS.map((a) => (
            <ToggleGroupItem key={a.id} value={a.id}>
              {a.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-muted-foreground">
          <button
            type="button"
            onClick={() => setShowTargets((v) => !v)}
            aria-pressed={showTargets}
            className={cn(
              "rounded-md border px-2 py-1 transition-colors",
              showTargets ? "border-novel/60 text-novel" : "border-border/60",
            )}
          >
            {showTargets ? "hide" : "show"} {DATA.targets.length} targets
          </button>
          <Badge variant="outline" className="font-mono text-[10px]">
            {edges.length} distal edges
          </Badge>
          <Badge variant="outline" className="font-mono text-[10px]">
            topology shared with full {stats.shared === null ? "—" : `${Math.round(stats.shared * 100)}%`}
          </Badge>
          <Badge variant="outline" className="font-mono text-[10px]">
            r(given, measured) {stats.r === null ? "—" : stats.r.toFixed(2)}
          </Badge>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
        <figure className="flex flex-col gap-2">
          <div className="relative aspect-square w-full max-w-[22rem] rounded-xl border border-border/60 bg-card/30 p-2">
            <canvas
              ref={canvasRef}
              width={n * CELL}
              height={n * CELL}
              onPointerMove={onMatrixMove}
              onPointerLeave={() => {
                setHover(null);
                setSel(null);
              }}
              className="h-full w-full [image-rendering:pixelated]"
              aria-label={`Contact map of ${n} windows. Upper triangle: measured detrended strength. Lower triangle: the edges the ${arm} arm conditions on.`}
              role="img"
            />
            <span className="pointer-events-none absolute top-3 right-3 rounded bg-background/80 px-1 font-mono text-[10px] text-muted-foreground">
              measured
            </span>
            <span className="pointer-events-none absolute bottom-3 left-3 rounded bg-background/80 px-1 font-mono text-[10px] text-muted-foreground">
              encoder input
            </span>
          </div>
          <figcaption className="min-h-[2.5rem] font-mono text-[11px] leading-relaxed text-muted-foreground">
            {hover ? (
              <>
                windows {hover.i}↔{hover.j} · {sepKb} kb apart · measured{" "}
                {measured.get(hoverKey ?? -1)?.toFixed(2) ?? "no pixel"}
                <br />
                {status}
              </>
            ) : (
              "Hover the map to read a pixel."
            )}
          </figcaption>
        </figure>

        <div
          tabIndex={0}
          role="application"
          aria-label="Arc view of the same graph. Use left and right arrows to select a window."
          onKeyDown={onKeyDown}
          className="focus-visible:ring-ring/70 self-start rounded-xl border border-border/60 bg-card/30 p-2 focus-visible:outline-none focus-visible:ring-[3px]"
        >
          <svg
            viewBox={`0 0 ${ARC.w} ${ARC.h}`}
            className="h-auto w-full"
            onPointerMove={onArcMove}
            onPointerLeave={() => setSel(null)}
          >
            <text x={ARC.pad} y={16} fill="var(--muted-foreground)" fontSize="13" fontFamily="var(--font-mono)">
              conditioning edges — attention runs here
            </text>
            {edges.map(([i, j, s]) => {
              const touches = sel !== null && (i === sel || j === sel);
              return (
                <path
                  key={`${i}:${j}`}
                  d={arcPath(i, j, false)}
                  fill="none"
                  stroke="var(--struct)"
                  strokeWidth={touches ? 2.2 : 0.5 + heat(s) * 1.2}
                  opacity={sel === null ? 0.03 + heat(s) * 0.22 : touches ? 0.95 : 0.03}
                  className="transition-opacity duration-150"
                />
              );
            })}
            {showTargets
              ? DATA.targets.map(([i, j]) => {
                  const touches = sel !== null && (i === sel || j === sel);
                  return (
                    <path
                      key={`t${i}:${j}`}
                      d={arcPath(i, j, true)}
                      fill="none"
                      stroke="var(--novel)"
                      strokeWidth={touches ? 2.2 : 0.8}
                      strokeDasharray="5 4"
                      opacity={sel === null ? 0.3 : touches ? 1 : 0.05}
                      className="transition-opacity duration-150"
                    />
                  );
                })
              : null}
            <line
              x1={ARC.pad}
              y1={ARC.track}
              x2={ARC.w - ARC.pad}
              y2={ARC.track}
              stroke="var(--seq)"
              strokeWidth="2.5"
              opacity="0.7"
            />
            {sel !== null ? (
              <circle cx={xOf(sel)} cy={ARC.track} r={5} fill="var(--primary)" />
            ) : null}
            <text
              x={ARC.pad}
              y={ARC.h - 8}
              fill="var(--muted-foreground)"
              fontSize="13"
              fontFamily="var(--font-mono)"
            >
              {showTargets ? "held-out targets — predicted, never attended over" : ""}
            </text>
            <text
              x={ARC.w - ARC.pad}
              y={ARC.h - 8}
              textAnchor="end"
              fill="var(--muted-foreground)"
              fontSize="13"
              fontFamily="var(--font-mono)"
            >
              {sel !== null
                ? `window ${sel} · ${source.chrom}:${mb(source.start_bp + sel * source.bin_size)} Mb`
                : `${source.chrom}:${mb(source.start_bp)}–${mb(source.start_bp + n * source.bin_size)} Mb`}
            </text>
          </svg>
        </div>
      </div>

      <div className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
        <p>{CAPTION[arm]}</p>
        <p className="text-[13px]">
          A real sample: {source.cell_line} {source.chrom}, bins {source.start_bin}–
          {source.start_bin + n - 1}, from a training chromosome, built by the same code the
          training loop runs. The local backbone (±{radius} bins, the blue band) is present in every
          arm and is all that is left under structure dropout. Every arm withholds the same{" "}
          {DATA.targets.length} targets, so the arms are compared on identical questions.
        </p>
      </div>
    </div>
  );
}
