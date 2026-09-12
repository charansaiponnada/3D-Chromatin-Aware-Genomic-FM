import { ExternalLink } from "lucide-react";

import { Reveal } from "@/components/reveal";
import { Section } from "@/components/section";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { GAP, LITERATURE } from "@/content/research";

const FAMILY_LABEL: Record<string, string> = {
  "sequence-to-structure": "sequence → structure",
  "hi-c representation": "Hi-C representation",
  "sequence foundation": "sequence foundation",
  closest: "closest precedent",
};

export function Literature() {
  return (
    <Section
      id="literature"
      eyebrow="05 — Literature"
      title="What has already been built"
      lede={
        <>
          Seven models define the space this project sits in. Hover any row — or read the second
          column — for what it leaves open. The claim being avoided here matters as much as the
          one being made.
        </>
      }
    >
      <div className="flex flex-col gap-8">
        <Reveal>
          <div className="overflow-x-auto rounded-xl border border-border/60">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[150px]">Model</TableHead>
                  <TableHead className="w-[160px]">Venue</TableHead>
                  <TableHead>Core idea</TableHead>
                  <TableHead className="w-[38%]">What it does not do</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {LITERATURE.map((p) => (
                  <TableRow
                    key={p.name}
                    className={p.family === "closest" ? "bg-novel-soft/25" : undefined}
                  >
                    <TableCell className="align-top font-medium">
                      <HoverCard openDelay={140}>
                        <HoverCardTrigger asChild>
                          <a
                            href={p.href}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1.5 underline-offset-4 hover:underline"
                          >
                            {p.name}
                            <ExternalLink className="size-3 opacity-60" />
                          </a>
                        </HoverCardTrigger>
                        <HoverCardContent className="w-80 text-sm leading-relaxed">
                          <Badge variant="outline" className="font-mono text-[10px]">
                            {FAMILY_LABEL[p.family]}
                          </Badge>
                          <p className="mt-2">{p.idea}</p>
                          <p className="mt-2 text-muted-foreground">{p.gap}</p>
                        </HoverCardContent>
                      </HoverCard>
                      {p.family === "closest" ? (
                        <Badge className="bg-novel-soft text-novel border-novel/40 mt-1.5 block w-fit text-[10px]">
                          closest precedent
                        </Badge>
                      ) : null}
                    </TableCell>
                    <TableCell className="align-top font-mono text-xs text-muted-foreground">
                      {p.venue}
                    </TableCell>
                    <TableCell className="align-top text-sm leading-relaxed">{p.idea}</TableCell>
                    <TableCell className="align-top text-sm leading-relaxed text-muted-foreground">
                      {p.gap}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Reveal>

        <Reveal>
          <Alert>
            <AlertTitle className="font-medium">
              Evo2HiC came out in November 2025, and it changes what can be claimed
            </AlertTitle>
            <AlertDescription className="leading-relaxed">
              It distils a frozen 7-billion-parameter model into a compact DNA encoder using
              contrastive alignment against Hi-C. That is a direct precedent for &ldquo;Hi-C
              shapes a sequence encoder&rdquo;, and it means the broad version of this project&apos;s
              idea is no longer novel. What remains open is the mechanism: end-to-end training in
              which contacts control contextualisation itself, separated from alignment and late
              fusion under matched compute.
            </AlertDescription>
          </Alert>
        </Reveal>
      </div>
    </Section>
  );
}

export function Gap() {
  return (
    <Section
      id="gap"
      eyebrow="06 — The gap"
      title="What is settled, what is not, and what will not be claimed"
    >
      <div className="grid gap-6 lg:grid-cols-3">
        <Reveal>
          <Card className="h-full">
            <CardHeader>
              <CardTitle className="text-base">Already established</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="flex flex-col gap-2.5 text-sm leading-relaxed text-muted-foreground">
                {GAP.established.map((g) => (
                  <li key={g} className="flex gap-2.5">
                    <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-muted-foreground/50" />
                    {g}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </Reveal>

        <Reveal delay={0.06}>
          <Card className="border-primary/30 h-full">
            <CardHeader>
              <CardTitle className="text-base">Still open</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="flex flex-col gap-2.5 text-sm leading-relaxed">
                {GAP.open.map((g) => (
                  <li key={g} className="flex gap-2.5">
                    <span className="bg-primary mt-1.5 size-1.5 shrink-0 rounded-full" />
                    {g}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </Reveal>

        <Reveal delay={0.12}>
          <Card className="border-destructive/30 h-full">
            <CardHeader>
              <CardTitle className="text-base">Deliberately not claimed</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
                {GAP.notClaimed.map((g) => (
                  <li key={g} className="flex gap-2.5">
                    <span className="text-destructive mt-0.5 shrink-0 font-mono text-xs">✕</span>
                    <span className="line-through decoration-destructive/40">{g}</span>
                  </li>
                ))}
              </ul>
              <p className="text-xs leading-relaxed text-muted-foreground">
                Each of these is contradicted by published work. Claiming any of them would be
                the fastest way to lose a reviewer.
              </p>
            </CardContent>
          </Card>
        </Reveal>
      </div>

      <Reveal>
        <blockquote className="border-primary/60 mt-10 border-l-2 pl-5 text-lg leading-relaxed text-pretty md:text-xl">
          {GAP.statement}
        </blockquote>
      </Reveal>
    </Section>
  );
}
