import { FoldingFigure } from "@/components/folding-figure";
import { Reveal } from "@/components/reveal";
import { Section } from "@/components/section";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { OBJECTIVES, QUESTIONS } from "@/content/research";

export function Problem() {
  return (
    <Section
      id="problem"
      eyebrow="01 — The problem"
      title="Sequence adjacency is not functional adjacency"
      lede={
        <>
          Human DNA is about two metres long and folds into a nucleus roughly six micrometres
          across. The folding is not random: it forms loops, domains and compartments, and it
          decides which regulatory elements can reach which genes.
        </>
      }
    >
      <div className="flex flex-col gap-8">
        <FoldingFigure />
        <Reveal>
          <div className="grid gap-6 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">The measurement exists</CardTitle>
              </CardHeader>
              <CardContent className="text-sm leading-relaxed text-muted-foreground">
                Hi-C sequences pairs of genomic fragments that were physically close in the
                nucleus, producing a contact matrix over genomic bins. Folding is not something
                we have to infer — it has been measured, in many cell types, and released
                publicly.
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">The models do not use it</CardTitle>
              </CardHeader>
              <CardContent className="text-sm leading-relaxed text-muted-foreground">
                Genomic language models read the genome as text. Attention over sequence position
                cannot know that two distant windows are neighbours in space unless something
                tells it, so the model has to rediscover from sequence what an experiment already
                established.
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">So the question is narrow</CardTitle>
              </CardHeader>
              <CardContent className="text-sm leading-relaxed text-muted-foreground">
                Tell it, during pretraining, and measure whether that changes what the encoder
                learns — with controls strong enough that a positive result means something and a
                negative one is still worth reporting.
              </CardContent>
            </Card>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

export function Aim() {
  return (
    <Section
      id="aim"
      eyebrow="02 — Aim"
      title="Condition the representation, do not decorate it"
      lede="Develop and evaluate a genomic representation model, trained from scratch, in which Hi-C contact information directly biases how a DNA sequence encoder exchanges information across distant genomic regions during self-supervised pretraining."
    >
      <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
        <Reveal>
          <Card className="h-full">
            <CardHeader>
              <CardTitle>What already exists, and why it is not this</CardTitle>
              <CardDescription>
                Three families of prior work, none of which does the thing being tested here.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4 text-sm leading-relaxed">
              <p className="text-muted-foreground">
                Sequence-to-structure models predict Hi-C from sequence — the contact map is the
                output. Hi-C foundation models pretrain on contact maps — the sequence is absent.
                The closest precedent aligns a frozen encoder against Hi-C embeddings after the
                fact — the contacts never enter the encoder&apos;s own contextualisation.
              </p>
              <p>
                In none of them does the measured contact map decide{" "}
                <strong className="font-medium">which distal windows communicate</strong> while
                the representation is still being formed. That is the gap this project occupies,
                and it is a narrow one on purpose.
              </p>
            </CardContent>
          </Card>
        </Reveal>
        <Reveal delay={0.08}>
          <Card className="border-primary/30 bg-primary/5 h-full">
            <CardHeader>
              <CardTitle>The bet</CardTitle>
              <CardDescription>Stated in advance, so it can lose.</CardDescription>
            </CardHeader>
            <CardContent className="text-sm leading-relaxed">
              <p>
                If real contacts carry information a sequence model cannot recover on its own,
                conditioning on them should beat a matched model given{" "}
                <strong className="font-medium">shuffled</strong> contacts.
              </p>
              <p className="mt-3 text-muted-foreground">
                If it does not, that null is itself a result about the limits of the signal — and
                the project reports it rather than tuning until the number moves.
              </p>
            </CardContent>
          </Card>
        </Reveal>
      </div>
    </Section>
  );
}

export function Questions() {
  return (
    <Section
      id="questions"
      eyebrow="03 — Research questions"
      title="Five questions, each with a control attached"
      lede="Each question is paired with the arm that answers it, so no question can be answered by a result that does not isolate it."
    >
      <Reveal>
        <Accordion type="single" collapsible defaultValue="RQ1" className="w-full">
          {QUESTIONS.map((q) => (
            <AccordionItem key={q.id} value={q.id}>
              <AccordionTrigger className="text-left">
                <span className="flex items-center gap-3">
                  <Badge variant="outline" className="font-mono text-[10px]">
                    {q.id}
                  </Badge>
                  <span className="font-medium">{q.title}</span>
                </span>
              </AccordionTrigger>
              <AccordionContent className="text-sm leading-relaxed text-muted-foreground">
                {q.body}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </Reveal>
    </Section>
  );
}

export function Objectives() {
  return (
    <Section
      id="objectives"
      eyebrow="04 — Objectives"
      title="Six objectives, mapped to the phase that delivers them"
      lede="Nothing here is aspirational: each objective is the deliverable of a specific phase, and each phase has a gate it must pass before the next one starts."
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {OBJECTIVES.map((o, i) => (
          <Reveal key={o.id} delay={i * 0.05}>
            <Card className="h-full">
              <CardHeader>
                <div className="flex items-center justify-between gap-2">
                  <Badge variant="secondary" className="font-mono text-[10px]">
                    {o.id}
                  </Badge>
                  <span className="font-mono text-[10px] text-muted-foreground">{o.phase}</span>
                </div>
                <CardTitle className="text-base leading-snug">{o.title}</CardTitle>
              </CardHeader>
              <CardContent className="text-sm leading-relaxed text-muted-foreground">
                {o.body}
              </CardContent>
            </Card>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}
