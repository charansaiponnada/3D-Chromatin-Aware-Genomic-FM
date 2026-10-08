import type { Metadata } from "next";
import Link from "next/link";

import { ArchitectureDiagram } from "@/components/architecture/architecture-diagram";
import { ContactGraph } from "@/components/architecture/contact-graph";
import { RealGraph } from "@/components/architecture/real-graph";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { publicConfig, kb } from "@/lib/config";
import { nodeMathHtml, pageMathHtml } from "@/lib/math";

export const metadata: Metadata = {
  title: "Architecture explorer — ChromGraphFM",
  description:
    "Interactive walkthrough of the ChromGraphFM encoder: contact-biased attention, learned edge bias, structure dropout and held-out target edges.",
};

export default function ArchitecturePage() {
  const config = publicConfig();
  const math = pageMathHtml();
  const nodeMath = nodeMathHtml();

  return (
    <div className="mx-auto w-full max-w-[100rem] px-5 pt-24 pb-20 md:px-8 md:pt-28">
      <div className="flex flex-col gap-2">
        <span className="font-mono text-xs uppercase tracking-[0.2em] text-muted-foreground">
          Architecture explorer
        </span>
        <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
          ChromGraphFM, component by component
        </h1>
        <p className="max-w-3xl text-pretty leading-relaxed text-muted-foreground">
          {config.nodesPerSample} genomic windows of {kb(config.binSize)} each — {kb(config.contextBp)} of
          context — encoded locally, then connected by a sparse graph built from measured Hi-C
          contacts. Hover a component for a summary, click it for the full explanation, the maths
          and the config values behind it. Arrow keys move between components; Escape closes.
        </p>
      </div>

      <div className="mt-10">
        <ArchitectureDiagram config={config} mathHtml={nodeMath} full />
      </div>

      <Card className="mt-14">
        <CardHeader>
          <CardTitle>The sparse contact graph</CardTitle>
        </CardHeader>
        <CardContent>
          <ContactGraph config={config} formulaHtml={math.attention} />
        </CardContent>
      </Card>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>The same graph on real data, arm by arm</CardTitle>
          <CardDescription>
            One measured GM12878 sample, exactly as each experimental arm&apos;s encoder receives
            it. Switch arms to see what each control keeps and what it destroys.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RealGraph />
        </CardContent>
      </Card>

      <div className="mt-12 flex flex-wrap items-center gap-3">
        <Button asChild variant="outline">
          <Link href="/#architecture">Back to the full write-up</Link>
        </Button>
        <span className="text-sm text-muted-foreground">
          The rest of the project — literature, gap, experiments, results — is on the main page.
        </span>
      </div>
    </div>
  );
}
