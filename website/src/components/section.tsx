import type { ReactNode } from "react";

import { Reveal } from "@/components/reveal";
import { cn } from "@/lib/utils";

/** A numbered page section with a consistent heading block. */
export function Section({
  id,
  eyebrow,
  title,
  lede,
  children,
  className,
}: {
  id: string;
  eyebrow?: string;
  title: string;
  lede?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      id={id}
      // scroll-mt keeps the heading clear of the sticky header when a hash link lands here
      className={cn("scroll-mt-24 border-t border-border/60 py-20 md:py-28", className)}
    >
      <div className="mx-auto w-full max-w-6xl px-5 md:px-8">
        <Reveal>
          <div className="flex flex-col gap-3">
            {eyebrow ? (
              <span className="font-mono text-xs uppercase tracking-[0.2em] text-muted-foreground">
                {eyebrow}
              </span>
            ) : null}
            <h2 className="text-3xl font-semibold tracking-tight text-balance md:text-4xl">
              {title}
            </h2>
            {lede ? (
              <div className="max-w-3xl text-pretty text-base leading-relaxed text-muted-foreground md:text-lg">
                {lede}
              </div>
            ) : null}
          </div>
        </Reveal>
        <div className="mt-10 md:mt-14">{children}</div>
      </div>
    </section>
  );
}

/** Renders KaTeX HTML produced at build time by src/lib/math.ts. */
export function MathBlock({ html, className }: { html: string; className?: string }) {
  return (
    <div
      className={cn("overflow-x-auto py-1 text-foreground", className)}
      // The HTML comes from our own KaTeX render of a literal string in
      // content/architecture.ts — no user input reaches this.
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
