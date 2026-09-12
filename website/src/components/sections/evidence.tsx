import { CircleDashed, CircleDot, CheckCircle2, Cpu, Laptop } from "lucide-react";

import { Reveal } from "@/components/reveal";
import { Section } from "@/components/section";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ARMS, PHASES, REFERENCES } from "@/content/research";
import type { PublicConfig } from "@/lib/config";
import type { ArmId, ResultsTable } from "@/lib/results";
import { cn } from "@/lib/utils";

/** A measured number, or an honest "??". Never a plausible-looking placeholder. */
function Metric({ value }: { value?: number }) {
  if (value === undefined) {
    return (
      <Badge variant="outline" className="border-destructive/40 text-destructive font-mono">
        ??
      </Badge>
    );
  }
  return <span className="font-mono tabular-nums">{value.toFixed(3)}</span>;
}

export function Experiments() {
  return (
    <Section
      id="experiments"
      eyebrow="10 — Experiments"
      title="Six controls, one model, identical budgets"
      lede="Same parameter count, same dataset, same optimiser, same schedule, same step budget. Only the structural signal reaching the encoder changes. A result that survives all six is worth something; one that does not survive B3 is worth nothing."
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {ARMS.map((arm, i) => (
          <Reveal key={arm.id} delay={i * 0.04}>
            <Card
              className={cn(
                "h-full",
                arm.ours && "border-primary/50 bg-primary/5",
                arm.decisive && "border-novel/45",
              )}
            >
              <CardHeader>
                <div className="flex items-center justify-between gap-2">
                  <Badge
                    variant={arm.ours ? "default" : "outline"}
                    className="font-mono text-[10px]"
                  >
                    {arm.id}
                  </Badge>
                  {arm.decisive ? (
                    <span className="text-novel font-mono text-[10px] uppercase tracking-wider">
                      decisive
                    </span>
                  ) : null}
                </div>
                <CardTitle className="text-base leading-snug">{arm.name}</CardTitle>
                <CardDescription className="font-mono text-[11px]">
                  {arm.structure}
                </CardDescription>
              </CardHeader>
              <CardContent className="text-sm leading-relaxed text-muted-foreground">
                {arm.tests}
              </CardContent>
            </Card>
          </Reveal>
        ))}
      </div>

      <Reveal>
        <Alert className="border-novel/40 mt-8">
          <AlertTitle className="font-medium">The two arms that decide the project</AlertTitle>
          <AlertDescription className="leading-relaxed">
            <strong className="text-foreground">B3 (shuffled Hi-C)</strong> keeps the signal
            statistics and destroys the biology, so beating it is the difference between learning
            chromatin structure and learning that an extra input helps.{" "}
            <strong className="text-foreground">B2 (distance-only)</strong> keeps genomic
            separation and drops contact strength — matching it would mean the model learned how
            far apart two windows are, which it could already read off the input.
          </AlertDescription>
        </Alert>
      </Reveal>
    </Section>
  );
}

export function Results({ results }: { results: ResultsTable }) {
  const empty = Object.keys(results).length === 0;

  return (
    <Section
      id="results"
      eyebrow="11 — Results"
      title="Nothing measured yet"
      lede="This table is generated from files under results/. Until a run produces one, every cell stays ??. No number appears on this site because it sounds plausible."
    >
      <div className="flex flex-col gap-6">
        {empty ? (
          <Reveal>
            <Alert className="border-destructive/40">
              <AlertTitle className="font-medium">No runs have completed</AlertTitle>
              <AlertDescription className="leading-relaxed">
                Phases 3–6 need the L40S cluster and have not started. The table below shows the
                comparison that will be filled in, in the shape it will be filled in, so the
                claim is committed to before the evidence exists rather than after it.
              </AlertDescription>
            </Alert>
          </Reveal>
        ) : null}

        <Reveal>
          <div className="overflow-x-auto rounded-xl border border-border/60">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[80px]">Arm</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Contact r (long-range)</TableHead>
                  <TableHead className="text-right">Loop AUPRC</TableHead>
                  <TableHead className="text-right">Hi-C-free probe</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {ARMS.map((arm) => {
                  const key = arm.id.toLowerCase() as ArmId;
                  const row = results[key];
                  return (
                    <TableRow key={arm.id} className={arm.ours ? "bg-primary/5" : undefined}>
                      <TableCell className="font-mono text-xs">{arm.id}</TableCell>
                      <TableCell className={cn("text-sm", arm.ours && "font-medium")}>
                        {arm.name}
                      </TableCell>
                      <TableCell className="text-right">
                        <Metric value={row?.contact_r_long} />
                      </TableCell>
                      <TableCell className="text-right">
                        <Metric value={row?.loop_auprc} />
                      </TableCell>
                      <TableCell className="text-right">
                        <Metric value={row?.hic_free_probe} />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </Reveal>

        <Reveal>
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">How these will be read</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-3 text-sm leading-relaxed text-muted-foreground">
                <p>
                  <strong className="text-foreground">Distance-stratified, never one number.</strong>{" "}
                  Hi-C is dominated by short-range contacts, so a single overall correlation can
                  look strong while long-range prediction sits at chance.
                </p>
                <p>
                  <strong className="text-foreground">Distance-matched negatives everywhere.</strong>{" "}
                  Loop and enhancer–promoter negatives are matched on genomic separation, or the
                  model is rewarded for knowing that nearby loci touch more often.
                </p>
              </CardContent>
            </Card>
            <Card className="border-destructive/30">
              <CardHeader>
                <CardTitle className="text-base">What would falsify the hypothesis</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-3 text-sm leading-relaxed text-muted-foreground">
                <p>Ours failing to exceed B3 on long-range contact prediction.</p>
                <p>
                  Ours matching B2, which would mean the model learned genomic separation rather
                  than chromatin structure.
                </p>
                <p className="text-foreground">
                  Either outcome gets reported as measured. A null on a well-controlled experiment
                  is a result about the signal, not a failure of the project.
                </p>
              </CardContent>
            </Card>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

const STATUS_ICON = {
  done: CheckCircle2,
  active: CircleDot,
  planned: CircleDashed,
} as const;

export function Timeline() {
  return (
    <Section
      id="timeline"
      eyebrow="12 — Timeline"
      title="Seven phases, each with a gate"
      lede="The gate decides when a phase ends, not the calendar. Phase 4 carries a stop gate: if the model given real Hi-C does not beat the same model given shuffled Hi-C, the central claim fails and that gets written up."
    >
      <div className="flex flex-col">
        {PHASES.map((p, i) => {
          const Icon = STATUS_ICON[p.status];
          const isStop = p.gate.startsWith("STOP GATE");
          return (
            <Reveal key={p.id} delay={i * 0.04}>
              <div className="group relative flex gap-5 pb-8 last:pb-0">
                {i < PHASES.length - 1 ? (
                  <span className="absolute left-[11px] top-7 h-full w-px bg-border" aria-hidden />
                ) : null}
                <Icon
                  className={cn(
                    "relative z-10 mt-1 size-6 shrink-0 bg-background",
                    p.status === "done" && "text-out",
                    p.status === "active" && "text-primary",
                    p.status === "planned" && "text-muted-foreground/50",
                  )}
                />
                <div className="flex min-w-0 flex-col gap-1.5">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium">{p.id}</span>
                    <span className="font-mono text-[11px] text-muted-foreground">{p.weeks}</span>
                    {p.gpu ? (
                      <Badge variant="outline" className="gap-1 font-mono text-[10px]">
                        <Cpu className="size-3" />
                        cluster
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="gap-1 font-mono text-[10px]">
                        <Laptop className="size-3" />
                        laptop
                      </Badge>
                    )}
                    {p.status === "done" ? (
                      <Badge className="bg-out-soft text-out border-out/40 text-[10px]">
                        complete
                      </Badge>
                    ) : null}
                  </div>
                  <p className="text-sm">{p.deliverable}</p>
                  <p
                    className={cn(
                      "font-mono text-[11px]",
                      isStop ? "text-novel" : "text-muted-foreground",
                    )}
                  >
                    gate — {p.gate.replace("STOP GATE — ", "")}
                    {isStop ? "  ⟵ the project's stop gate" : ""}
                  </p>
                </div>
              </div>
            </Reveal>
          );
        })}
      </div>
    </Section>
  );
}

export function Reproducibility({ config }: { config: PublicConfig }) {
  const rules = [
    {
      title: "No number without a file",
      body: "Nothing appears in a slide, a report, a figure or on this page unless it traces to a file under results/. Placeholders stay ??.",
    },
    {
      title: "Config written before step one",
      body: "Every run writes run_config.yaml next to its metrics.json before training starts, so a run that dies halfway still leaves a complete record of what it was.",
    },
    {
      title: "Controls use a separate seed",
      body: `Shuffles and random graphs draw from control_seed = ${config.controlSeed}, never the training seed, so seed variance and shuffle variance can never be confounded.`,
    },
    {
      title: "No claim without a matched setting",
      body: "No claim of superiority over a published model unless task, data, split, metric, resolution and input setting all match. Published numbers are context, never a head-to-head.",
    },
    {
      title: "The config is the source of truth",
      body: "Every number on this page is read from configs/base.yaml at build time. If a key is renamed, the build fails rather than rendering a stale value.",
    },
  ];

  return (
    <Section
      id="reproducibility"
      eyebrow="13 — Reproducibility"
      title="The rules the project holds itself to"
      lede="These are not aspirations written after the fact — they are enforced by the code. The config loader refuses a chromosome leak between splits; the asset script fails the build on a dangling config reference."
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {rules.map((r, i) => (
          <Reveal key={r.title} delay={i * 0.05}>
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="text-base leading-snug">{r.title}</CardTitle>
              </CardHeader>
              <CardContent className="text-sm leading-relaxed text-muted-foreground">
                {r.body}
              </CardContent>
            </Card>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}

export function References() {
  return (
    <Section id="references" eyebrow="14 — References" title="References">
      <Reveal>
        <ol className="flex flex-col gap-3">
          {REFERENCES.map((r, i) => (
            <li key={i} className="flex gap-3 text-sm leading-relaxed text-muted-foreground">
              <span className="font-mono text-xs text-muted-foreground/60">[{i + 1}]</span>
              <span>{r}</span>
            </li>
          ))}
        </ol>
      </Reveal>
      <Reveal>
        <p className="mt-6 max-w-3xl text-xs leading-relaxed text-muted-foreground">
          Author lists for entries 5, 6 and 7 are to be completed against the published record
          before submission. They are cited here by title and identifier so that every entry is
          verifiable exactly as written.
        </p>
      </Reveal>
    </Section>
  );
}
