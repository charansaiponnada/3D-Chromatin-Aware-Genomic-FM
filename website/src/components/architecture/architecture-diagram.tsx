"use client";

import { motion, useReducedMotion } from "motion/react";
import { useCallback, useMemo, useState } from "react";

import { DetailPanel } from "@/components/architecture/detail-panel";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import {
  BLOCK_BOX,
  EDGES,
  GRID,
  NODES,
  NODE_BY_ID,
  TOUR_ORDER,
  type ArchEdge,
  type ArchNode,
  type NodeId,
} from "@/content/architecture";
import type { PublicConfig } from "@/lib/config";
import { useMediaQuery } from "@/lib/use-media-query";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ styling */

const KIND_CLASS: Record<ArchNode["kind"], string> = {
  data: "border-border bg-card text-card-foreground",
  module: "border-seq/45 bg-seq-soft/45 text-foreground",
  bias: "border-struct/45 bg-struct-soft/45 text-foreground",
  novel: "border-novel/55 bg-novel-soft/45 text-foreground",
  head: "border-head/45 bg-head-soft/45 text-foreground",
  output: "border-out/45 bg-out-soft/45 text-foreground",
};

const EDGE_STROKE: Record<ArchEdge["kind"], string> = {
  tensor: "var(--muted-foreground)",
  structure: "var(--struct)",
  bias: "var(--struct)",
  loss: "var(--out)",
  target: "var(--novel)",
};

/* --------------------------------------------------------------- geometry */

type Point = { x: number; y: number };

/**
 * Pick the pair of box edges that gives the shortest, least-crossing run, then
 * join them with a cubic. Doing this from the node's declared rectangle rather
 * than from a measured DOM node is what keeps the diagram responsive with no
 * ResizeObserver: the SVG viewBox and the HTML nodes share one coordinate space.
 */
function anchorPair(a: ArchNode, b: ArchNode): [Point, Point] {
  const ac = { x: a.x + a.w / 2, y: a.y + a.h / 2 };
  const bc = { x: b.x + b.w / 2, y: b.y + b.h / 2 };
  const dx = bc.x - ac.x;
  const dy = bc.y - ac.y;

  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0
      ? [{ x: a.x + a.w, y: ac.y }, { x: b.x, y: bc.y }]
      : [{ x: a.x, y: ac.y }, { x: b.x + b.w, y: bc.y }];
  }
  return dy >= 0
    ? [{ x: ac.x, y: a.y + a.h }, { x: bc.x, y: b.y }]
    : [{ x: ac.x, y: a.y }, { x: bc.x, y: b.y + b.h }];
}

function edgePath(a: ArchNode, b: ArchNode): string {
  const [p, q] = anchorPair(a, b);
  const dx = q.x - p.x;
  const dy = q.y - p.y;
  const horizontal = Math.abs(dx) >= Math.abs(dy);
  const c = horizontal
    ? [
        { x: p.x + dx * 0.5, y: p.y },
        { x: q.x - dx * 0.5, y: q.y },
      ]
    : [
        { x: p.x, y: p.y + dy * 0.5 },
        { x: q.x, y: q.y - dy * 0.5 },
      ];
  return `M ${p.x} ${p.y} C ${c[0].x} ${c[0].y}, ${c[1].x} ${c[1].y}, ${q.x} ${q.y}`;
}

/* ------------------------------------------------------------------ layer */

function EdgeLayer({ hovered, active }: { hovered: NodeId | null; active: NodeId | null }) {
  const focus = hovered ?? active;
  const paths = useMemo(
    () =>
      EDGES.map((e) => {
        const a = NODE_BY_ID.get(e.from)!;
        const b = NODE_BY_ID.get(e.to)!;
        return { edge: e, d: edgePath(a, b) };
      }),
    [],
  );

  return (
    <svg
      viewBox={`0 0 ${GRID.w} ${GRID.h}`}
      preserveAspectRatio="xMidYMid meet"
      className="pointer-events-none absolute inset-0 h-full w-full"
      aria-hidden="true"
    >
      <defs>
        {(["tensor", "structure", "bias", "loss", "target"] as const).map((kind) => (
          <marker
            key={kind}
            id={`arrow-${kind}`}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill={EDGE_STROKE[kind]} />
          </marker>
        ))}
      </defs>

      {/* the repeated block */}
      <rect
        x={BLOCK_BOX.x}
        y={BLOCK_BOX.y}
        width={BLOCK_BOX.w}
        height={BLOCK_BOX.h}
        rx="10"
        fill="none"
        stroke="var(--border)"
        strokeWidth="1.5"
        strokeDasharray="7 5"
      />
      <text
        x={BLOCK_BOX.x + 12}
        y={BLOCK_BOX.y + 20}
        fill="var(--muted-foreground)"
        fontSize="12"
        fontFamily="var(--font-mono)"
      >
        encoder block × N
      </text>

      {paths.map(({ edge, d }, i) => {
        const related = !focus || edge.from === focus || edge.to === focus;
        return (
          <path
            key={i}
            d={d}
            fill="none"
            stroke={EDGE_STROKE[edge.kind]}
            strokeWidth={related && focus ? 2.6 : 1.6}
            strokeDasharray={edge.kind === "target" ? "6 5" : undefined}
            markerEnd={`url(#arrow-${edge.kind})`}
            className={cn(
              "transition-opacity duration-300",
              related ? "opacity-90" : "opacity-15",
            )}
          />
        );
      })}
    </svg>
  );
}

/* ------------------------------------------------------------------- node */

function NodeButton({
  node,
  hovered,
  active,
  neighbours,
  onHover,
  onSelect,
}: {
  node: ArchNode;
  hovered: NodeId | null;
  active: NodeId | null;
  neighbours: Set<NodeId>;
  onHover: (id: NodeId | null) => void;
  onSelect: (id: NodeId) => void;
}) {
  const focus = hovered ?? active;
  const dim = focus !== null && focus !== node.id && !neighbours.has(node.id);

  return (
    <HoverCard openDelay={120} closeDelay={60}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          id={`arch-node-${node.id}`}
          aria-expanded={active === node.id}
          aria-current={active === node.id ? "true" : undefined}
          onMouseEnter={() => onHover(node.id)}
          onMouseLeave={() => onHover(null)}
          onFocus={() => onHover(node.id)}
          onBlur={() => onHover(null)}
          onClick={() => onSelect(node.id)}
          style={{
            left: `${(node.x / GRID.w) * 100}%`,
            top: `${(node.y / GRID.h) * 100}%`,
            width: `${(node.w / GRID.w) * 100}%`,
            height: `${(node.h / GRID.h) * 100}%`,
          }}
          className={cn(
            "absolute flex flex-col items-center justify-center rounded-lg border px-2 text-center",
            "transition-[opacity,transform,box-shadow,border-color] duration-300",
            "focus-visible:ring-ring/70 focus-visible:outline-none focus-visible:ring-[3px]",
            "cursor-pointer hover:-translate-y-0.5",
            KIND_CLASS[node.kind],
            dim && "opacity-25",
            active === node.id && "ring-primary/70 ring-2 ring-offset-2 ring-offset-background",
            node.novel && "shadow-[0_0_0_1px_var(--novel)_inset]",
          )}
        >
          <span className="text-[clamp(0.55rem,0.95cqw,0.82rem)] font-medium leading-tight text-balance">
            {node.label}
          </span>
          {node.sub ? (
            <span className="mt-0.5 font-mono text-[clamp(0.45rem,0.72cqw,0.66rem)] leading-tight text-muted-foreground">
              {node.sub}
            </span>
          ) : null}
        </button>
      </HoverCardTrigger>
      <HoverCardContent side="top" className="w-72 text-sm leading-relaxed">
        <p className="font-medium">{node.label}</p>
        <p className="mt-1 text-muted-foreground">{node.short}</p>
        <p className="mt-2 font-mono text-[11px] text-muted-foreground">
          Click for the full explanation
        </p>
      </HoverCardContent>
    </HoverCard>
  );
}

/* ------------------------------------------------------------- the diagram */

export function ArchitectureDiagram({
  mathHtml,
  config,
  full = false,
}: {
  mathHtml: Partial<Record<NodeId, string>>;
  config: PublicConfig;
  full?: boolean;
}) {
  const [hovered, setHovered] = useState<NodeId | null>(null);
  const [active, setActive] = useState<NodeId | null>(null);
  const reduced = useReducedMotion();
  const isDesktop = useMediaQuery("(min-width: 1024px)");

  const neighbours = useMemo(() => {
    const focus = hovered ?? active;
    const set = new Set<NodeId>();
    if (!focus) return set;
    for (const e of EDGES) {
      if (e.from === focus) set.add(e.to);
      if (e.to === focus) set.add(e.from);
    }
    return set;
  }, [hovered, active]);

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.key === "Escape") {
        setActive(null);
        return;
      }
      const step = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1
        : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1
        : 0;
      if (!step) return;
      event.preventDefault();
      const current = hovered ?? active ?? TOUR_ORDER[0];
      const next = TOUR_ORDER[(TOUR_ORDER.indexOf(current) + step + TOUR_ORDER.length) % TOUR_ORDER.length];
      document.getElementById(`arch-node-${next}`)?.focus();
    },
    [hovered, active],
  );

  return (
    <div className={cn("flex flex-col gap-6", full ? "xl:flex-row xl:items-start" : "lg:flex-row lg:items-start")}>
      <div className={cn("flex min-w-0 flex-1 flex-col gap-4")}>
        {/* Screen readers get the pipeline as prose; the canvas below is aria-hidden decoration
            around real buttons, so nothing here is only available visually. */}
        <p className="sr-only">
          The model has two input streams. DNA sequence windows are embedded and encoded by a
          bidirectional Mamba encoder into one embedding per window. Hi-C contact maps are turned
          into a sparse graph over those same windows, a fraction of edges is held out as
          prediction targets, and structure dropout may remove the graph entirely. The remaining
          edges produce a learned bias that is added to attention scores inside the encoder block,
          which repeats N times. Its output feeds three self-supervised heads: masked DNA
          reconstruction, contact-aware contrastive learning, and contact prediction on the
          held-out edges.
        </p>

        {/* Below lg the canvas would be unreadable, so the same data renders as cards. */}
        <div
          role="group"
          aria-label="Interactive model architecture"
          onKeyDown={onKeyDown}
          className="relative hidden aspect-[25/16] w-full rounded-xl border border-border/60 bg-card/30 lg:block"
          style={{ containerType: "inline-size" }}
        >
          <div className="grid-backdrop absolute inset-0 rounded-xl opacity-40" />
          <EdgeLayer hovered={hovered} active={active} />
          {NODES.map((node) => (
            <NodeButton
              key={node.id}
              node={node}
              hovered={hovered}
              active={active}
              neighbours={neighbours}
              onHover={setHovered}
              onSelect={(id) => setActive((prev) => (prev === id ? null : id))}
            />
          ))}
        </div>

        <div className="flex flex-col gap-2 lg:hidden">
          {TOUR_ORDER.map((id) => {
            const node = NODE_BY_ID.get(id)!;
            return (
              <button
                key={id}
                type="button"
                onClick={() => setActive(id)}
                className={cn(
                  "flex flex-col gap-1 rounded-lg border px-3 py-2.5 text-left transition-colors",
                  KIND_CLASS[node.kind],
                )}
              >
                <span className="flex items-center gap-2 text-sm font-medium">
                  {node.label}
                  {node.novel ? (
                    <Badge className="bg-novel-soft text-novel border-novel/40 text-[10px]">
                      novel
                    </Badge>
                  ) : null}
                </span>
                <span className="text-xs leading-relaxed text-muted-foreground">{node.short}</span>
              </button>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 font-mono text-[11px] text-muted-foreground">
          <LegendSwatch className="bg-novel-soft border-novel/60" label="novel to this work" />
          <LegendSwatch className="bg-seq-soft border-seq/50" label="sequence" />
          <LegendSwatch className="bg-struct-soft border-struct/50" label="structure" />
          <LegendSwatch className="bg-head-soft border-head/50" label="objective" />
          <span className="hidden lg:inline">
            hover to preview · click for detail · arrow keys to move · Esc to close
          </span>
        </div>
      </div>

      {/* Desktop detail panel. Always mounted so layout does not jump. */}
      <div
        className={cn(
          "hidden lg:block",
          full ? "w-full xl:w-[26rem] xl:shrink-0" : "lg:w-[21rem] lg:shrink-0",
        )}
      >
        <Card className="sticky top-24 max-h-[calc(100vh-8rem)] overflow-hidden p-5">
          {/* A keyed motion.div rather than AnimatePresence: mode="wait" holds the
              incoming child until the outgoing one finishes exiting, and here that
              exit never resolved, so the panel stayed on its empty state forever.
              Remounting on key change gives the same cross-fade with no stall. */}
          {active ? (
            <motion.div
              key={active}
              initial={{ opacity: 0, y: reduced ? 0 : 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="h-full"
            >
              <DetailPanel
                active={active}
                onSelect={setActive}
                onClose={() => setActive(null)}
                mathHtml={mathHtml}
                config={config}
              />
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.2 }}
              className="flex flex-col gap-3 py-6"
            >
              <h3 className="text-lg font-semibold tracking-tight">Pick a component</h3>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Hover any box for a one-line summary. Click it for what it does, why it is there,
                the maths, and the exact values it reads from the training config.
              </p>
              <p className="text-sm leading-relaxed text-muted-foreground">
                The four boxes outlined in red are the ones this project contributes. Everything
                else is standard, and deliberately so — the comparison is only fair if the parts
                that are not the claim are ordinary.
              </p>
            </motion.div>
          )}
        </Card>
      </div>

      {/* Mobile: the same panel in a bottom sheet. Unmounted on desktop rather
          than hidden, because SheetContent always renders a full-screen overlay
          and `lg:hidden` on the content would leave that overlay dimming the page. */}
      {!isDesktop ? (
      <Sheet open={active !== null} onOpenChange={(open) => !open && setActive(null)}>
        <SheetContent side="bottom" className="max-h-[85vh] overflow-y-auto">
          <SheetTitle className="sr-only">
            {active ? NODE_BY_ID.get(active)!.label : "Component detail"}
          </SheetTitle>
          {active ? (
            <div className="px-4 pb-6">
              <DetailPanel
                active={active}
                onSelect={setActive}
                onClose={() => setActive(null)}
                mathHtml={mathHtml}
                config={config}
              />
            </div>
          ) : null}
        </SheetContent>
      </Sheet>
      ) : null}
    </div>
  );
}

function LegendSwatch({ className, label }: { className: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={cn("inline-block size-3 rounded-sm border", className)} />
      {label}
    </span>
  );
}
