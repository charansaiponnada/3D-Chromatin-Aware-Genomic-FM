import Link from "next/link";

import { MathBlock } from "@/components/section";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { type PublicConfig, kb } from "@/lib/config";

export function Hero({ config, attentionHtml }: { config: PublicConfig; attentionHtml: string }) {
  return (
    <section className="relative overflow-hidden pt-32 pb-20 md:pt-40 md:pb-28">
      <div className="grid-backdrop pointer-events-none absolute inset-0" aria-hidden="true" />
      <div className="relative mx-auto w-full max-w-6xl px-5 md:px-8">
        <Badge variant="outline" className="font-mono text-[11px]">
          B.Tech Mini Project · 23CS7552 · VR Siddhartha Engineering College
        </Badge>

        <h1 className="mt-6 max-w-4xl text-balance text-4xl font-semibold leading-[1.08] tracking-tight md:text-6xl">
          Teaching a DNA model that the genome{" "}
          <span className="text-primary">folds</span>.
        </h1>

        <p className="mt-6 max-w-2xl text-pretty text-lg leading-relaxed text-muted-foreground">
          Enhancers regulate genes megabases away because chromatin folding brings them into
          contact. Hi-C measures that folding. Most sequence models never see it.{" "}
          <strong className="font-medium text-foreground">ChromGraphFM</strong> lets measured
          contacts decide which distant DNA windows exchange information — inside the encoder,
          throughout pretraining, not bolted on afterwards.
        </p>

        <div className="mt-8 max-w-3xl rounded-xl border border-border/60 bg-card/40 px-5 py-4">
          <MathBlock html={attentionHtml} />
          <p className="mt-1 font-mono text-[11px] text-muted-foreground">
            attention over sparse graph edges only — the three bias terms are the contribution
          </p>
        </div>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Button asChild size="lg">
            <Link href="/architecture">Explore the architecture</Link>
          </Button>
          <Button asChild variant="outline" size="lg">
            <Link href="#gap">Why it is not already done</Link>
          </Button>
        </div>

        <dl className="mt-14 grid grid-cols-2 gap-x-6 gap-y-6 border-t border-border/60 pt-8 md:grid-cols-4">
          <Stat label="Context per sample" value={kb(config.contextBp)} sub={`${config.nodesPerSample} windows × ${kb(config.binSize)}`} />
          <Stat label="Distal edges per window" value={`top-${config.topKEdges}`} sub={`plus ±${config.localRadius} local`} />
          <Stat label="Structure dropout" value={`p = ${config.structureDropout}`} sub="makes Hi-C-free testable" />
          <Stat label="Control arms" value="6 + ours" sub="matched params, data, steps" />
        </dl>
      </div>
    </section>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
        {label}
      </dt>
      <dd className="text-2xl font-semibold tracking-tight">{value}</dd>
      <dd className="font-mono text-[11px] text-muted-foreground">{sub}</dd>
    </div>
  );
}
