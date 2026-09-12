"use client";

import { ArrowLeft, ArrowRight, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { NODE_BY_ID, TOUR_ORDER, type NodeId } from "@/content/architecture";
import type { PublicConfig } from "@/lib/config";
import { cn } from "@/lib/utils";

export function DetailPanel({
  active,
  onSelect,
  onClose,
  mathHtml,
  config,
  className,
}: {
  active: NodeId;
  onSelect: (id: NodeId) => void;
  onClose: () => void;
  mathHtml: Partial<Record<NodeId, string>>;
  config: PublicConfig;
  className?: string;
}) {
  const node = NODE_BY_ID.get(active)!;
  const index = TOUR_ORDER.indexOf(active);
  const prev = TOUR_ORDER[index - 1];
  const next = TOUR_ORDER[index + 1];

  return (
    <div className={cn("flex h-full flex-col gap-5", className)}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="font-mono text-[10px] uppercase tracking-wider">
              {node.lane}
            </Badge>
            {node.novel ? (
              <Badge className="bg-novel-soft text-novel border-novel/40 text-[10px] uppercase tracking-wider">
                novel
              </Badge>
            ) : null}
          </div>
          <h3 className="text-xl font-semibold tracking-tight text-balance">{node.label}</h3>
          {node.sub ? (
            <p className="font-mono text-xs text-muted-foreground">{node.sub}</p>
          ) : null}
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          aria-label="Close explanation"
          className="shrink-0"
        >
          <X />
        </Button>
      </div>

      <Separator />

      <div className="flex flex-1 flex-col gap-5 overflow-y-auto pr-1">
        <div className="flex flex-col gap-1.5">
          <h4 className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            What it does
          </h4>
          <p className="text-sm leading-relaxed text-pretty">{node.what}</p>
        </div>

        <div className="flex flex-col gap-1.5">
          <h4 className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            Why it is here
          </h4>
          <p className="text-sm leading-relaxed text-pretty text-muted-foreground">{node.why}</p>
        </div>

        {mathHtml[active] ? (
          <div className="flex flex-col gap-2">
            <h4 className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
              Maths
            </h4>
            <div
              className="overflow-x-auto rounded-md border border-border/70 bg-muted/30 px-3 py-3 text-sm"
              dangerouslySetInnerHTML={{ __html: mathHtml[active]! }}
            />
          </div>
        ) : null}

        {node.configRefs?.length ? (
          <div className="flex flex-col gap-2">
            <h4 className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
              From configs/base.yaml
            </h4>
            <ul className="flex flex-col gap-1.5">
              {node.configRefs.map((ref) => (
                <li
                  key={ref}
                  className="flex items-baseline justify-between gap-3 rounded-md border border-border/60 bg-muted/20 px-2.5 py-1.5"
                >
                  <code className="font-mono text-[11px] text-muted-foreground">{ref}</code>
                  <code className="font-mono text-xs font-medium text-primary">
                    {config.flat[ref] ?? "—"}
                  </code>
                </li>
              ))}
            </ul>
            <p className="text-[11px] leading-relaxed text-muted-foreground">
              Read from the training config at build time, not retyped here. If a key is renamed,
              the build fails rather than showing a stale number.
            </p>
          </div>
        ) : null}
      </div>

      <Separator />

      <div className="flex items-center justify-between gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={!prev}
          onClick={() => prev && onSelect(prev)}
        >
          <ArrowLeft data-icon="inline-start" />
          {prev ? NODE_BY_ID.get(prev)!.label : "Start"}
        </Button>
        <span className="font-mono text-[11px] text-muted-foreground">
          {index + 1} / {TOUR_ORDER.length}
        </span>
        <Button
          variant="outline"
          size="sm"
          disabled={!next}
          onClick={() => next && onSelect(next)}
        >
          {next ? NODE_BY_ID.get(next)!.label : "End"}
          <ArrowRight data-icon="inline-end" />
        </Button>
      </div>
    </div>
  );
}
