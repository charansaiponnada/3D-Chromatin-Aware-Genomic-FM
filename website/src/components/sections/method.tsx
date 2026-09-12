import Link from "next/link";

import { ArchitectureDiagram } from "@/components/architecture/architecture-diagram";
import { ContactGraph } from "@/components/architecture/contact-graph";
import { Reveal } from "@/components/reveal";
import { MathBlock, Section } from "@/components/section";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { NodeId } from "@/content/architecture";
import { DATA_SOURCES } from "@/content/research";
import { type PublicConfig, kb } from "@/lib/config";
import type { PageMath } from "@/lib/math";
import { cn } from "@/lib/utils";

export function Architecture({
  config,
  mathHtml,
  attentionHtml,
}: {
  config: PublicConfig;
  mathHtml: Partial<Record<NodeId, string>>;
  attentionHtml: string;
}) {
  return (
    <Section
      id="architecture"
      eyebrow="07 — Architecture"
      title="The mechanism, component by component"
      lede={
        <>
          Hover any box for a one-line summary; click it for what it does, why it is there, the
          maths, and the exact values it reads from the training config. The four boxes outlined
          in red are this project&apos;s contribution — everything else is deliberately ordinary,
          because a comparison is only fair when the parts that are not the claim are standard.
        </>
      }
    >
      <div className="flex flex-col gap-10">
        <ArchitectureDiagram config={config} mathHtml={mathHtml} />

        <Reveal>
          <div className="flex flex-wrap items-center gap-3">
            <Button asChild variant="outline">
              <Link href="/architecture">Open the full-screen explorer</Link>
            </Button>
            <span className="text-sm text-muted-foreground">
              More room for the diagram and the explanations side by side.
            </span>
          </div>
        </Reveal>

        <Reveal>
          <Card>
            <CardHeader>
              <CardTitle>The sparse contact graph, made visible</CardTitle>
              <CardDescription>
                This is the claim in one picture: attention exists only on edges, and measured
                contacts reweight them. Step through the four states to see what the graph builder
                produces, what is withheld, and what survives structure dropout.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ContactGraph config={config} formulaHtml={attentionHtml} />
            </CardContent>
          </Card>
        </Reveal>
      </div>
    </Section>
  );
}

export function Objectives3({ config, math }: { config: PublicConfig; math: PageMath }) {
  const losses = [
    {
      name: "Masked DNA reconstruction",
      weight: config.lambdaDna,
      html: math.dna,
      role: "regulariser",
      body: `A fraction of bases is masked and reconstructed. Kept as a regulariser rather than a headline number: predicting a masked base from a window ${kb(config.contextBp / 2)} away is close to information-free, so this loss is expected to move very little.`,
    },
    {
      name: "Contact-aware contrastive",
      weight: config.lambdaContrast,
      html: math.contrast,
      role: "distance-matched",
      body: "Contacting windows are pulled together, non-contacting ones pushed apart — with negatives drawn at the same genomic separation as their positive. Sampled freely, the model could minimise this by learning genomic distance, which it can already read off the input.",
    },
    {
      name: "Contact prediction",
      weight: config.lambdaContact,
      html: math.contact,
      role: "primary signal",
      body: `Predict normalised contact strength for the ${Math.round(config.heldOutEdgeFrac * 100)}% of edges the encoder was never shown. This is what forces structural information into the representation, and it is only meaningful because its targets were removed from the conditioning graph first.`,
    },
  ];

  return (
    <Section
      id="objective-fn"
      eyebrow="08 — Pretraining"
      title="Three objectives, one weighted sum"
      lede="Every control arm optimises this same objective with these same weights. Only the structural signal reaching the encoder differs — otherwise a difference in the result would say nothing about whether measured contacts help."
    >
      <div className="flex flex-col gap-8">
        <Reveal>
          <div className="rounded-xl border border-border/60 bg-card/40 px-5 py-4">
            <MathBlock html={math.total} />
          </div>
        </Reveal>

        <div className="grid gap-4 lg:grid-cols-3">
          {losses.map((l, i) => (
            <Reveal key={l.name} delay={i * 0.06}>
              <Card className="h-full">
                <CardHeader>
                  <div className="flex items-center justify-between gap-2">
                    <Badge variant="outline" className="font-mono text-[10px]">
                      λ = {l.weight}
                    </Badge>
                    <span className="font-mono text-[10px] text-muted-foreground">{l.role}</span>
                  </div>
                  <CardTitle className="text-base leading-snug">{l.name}</CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col gap-3">
                  <div
                    className="overflow-x-auto rounded-md border border-border/60 bg-muted/25 px-3 py-2.5 text-[13px]"
                    dangerouslySetInnerHTML={{ __html: l.html }}
                  />
                  <p className="text-sm leading-relaxed text-muted-foreground">{l.body}</p>
                </CardContent>
              </Card>
            </Reveal>
          ))}
        </div>

        <Reveal>
          <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">
            Weights are tuned on the validation chromosomes only. The test chromosomes are touched
            once, at the end, and the number that comes out is the number that gets reported.
          </p>
        </Reveal>
      </div>
    </Section>
  );
}

export function Data({ config }: { config: PublicConfig }) {
  const chroms = [
    ...config.chromsTrain.map((c) => ({ c, split: "train" as const })),
    ...config.chromsVal.map((c) => ({ c, split: "val" as const })),
    ...config.chromsTest.map((c) => ({ c, split: "test" as const })),
  ].sort((a, b) => Number(a.c.replace("chr", "")) - Number(b.c.replace("chr", "")));

  return (
    <Section
      id="data"
      eyebrow="09 — Data"
      title="Public data, chromosome-level splits, a manifest that rebuilds it"
      lede={
        <>
          Git carries the code and a manifest of accessions and checksums — never the data itself.
          hg38 is 3.1 GB and a single contact map can be 30 GB, so the cluster rebuilds the
          dataset from the manifest rather than receiving it.
        </>
      }
    >
      <div className="flex flex-col gap-8">
        <Reveal>
          <div className="overflow-x-auto rounded-xl border border-border/60">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Data</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead>Format</TableHead>
                  <TableHead>Use</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {DATA_SOURCES.map((d) => (
                  <TableRow key={d.data}>
                    <TableCell className="font-medium">{d.data}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">{d.source}</TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {d.format}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">{d.use}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Reveal>

        <Reveal>
          <Card>
            <CardHeader>
              <CardTitle>Chromosome-level splits, never random windows</CardTitle>
              <CardDescription>
                Hi-C is autocorrelated over megabases. A random window split puts near-identical
                regions on both sides of the boundary and inflates every metric, which is how a
                model that has memorised a locus can look like a model that has generalised.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-5">
              <div className="flex flex-wrap gap-1.5">
                {chroms.map(({ c, split }) => (
                  <span
                    key={c}
                    className={cn(
                      "rounded-md border px-2 py-1 font-mono text-[11px]",
                      split === "train" && "border-border bg-muted/40 text-muted-foreground",
                      split === "val" && "border-head/50 bg-head-soft/50 text-foreground",
                      split === "test" && "border-out/50 bg-out-soft/50 text-foreground",
                    )}
                  >
                    {c.replace("chr", "")}
                  </span>
                ))}
              </div>
              <div className="flex flex-wrap gap-x-6 gap-y-2 font-mono text-[11px] text-muted-foreground">
                <LegendDot className="bg-muted/60 border-border" label={`train · ${config.chromsTrain.length} autosomes`} />
                <LegendDot className="bg-head-soft border-head/50" label={`validation · ${config.chromsVal.join(", ")}`} />
                <LegendDot className="bg-out-soft border-out/50" label={`test · ${config.chromsTest.join(", ")}`} />
              </div>
              <div className="grid gap-4 border-t border-border/60 pt-5 sm:grid-cols-2">
                <div>
                  <h4 className="text-sm font-medium">Held-out cell line</h4>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                    Pretrain on several cell lines, evaluate on one the model has never seen. A
                    harder question than a held-out chromosome: it asks whether the model learned
                    sequence-to-structure rules or memorised one cell type&apos;s map.
                  </p>
                </div>
                <div>
                  <h4 className="text-sm font-medium">Hi-C-free inference</h4>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                    Remove the contact graph entirely at test time. This is the setting that
                    separates a transferable representation from a multimodal predictor, and it is
                    only fair because structure dropout made it in-distribution during training.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </Reveal>

        <Reveal>
          <div className="rounded-xl border border-border/60 bg-muted/20 px-5 py-4">
            <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
              Laptop pilot
            </p>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Phases 0–2 run on a laptop with no GPU, on {config.pilotChroms.join(" and ")} only —
              small chromosomes, a few hours of streaming, enough to prove the pipeline is correct
              before any cluster time is spent. The same script builds the full dataset on the
              cluster with one flag.
            </p>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

function LegendDot({ className, label }: { className: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={cn("inline-block size-3 rounded-sm border", className)} />
      {label}
    </span>
  );
}
