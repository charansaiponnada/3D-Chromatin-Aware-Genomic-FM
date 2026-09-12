"use client";

import { motion, useReducedMotion } from "motion/react";

/**
 * The one-sentence version of the problem, drawn.
 *
 * Left: the genome as a sequence model reads it — an enhancer and a promoter
 * separated by a megabase of intervening DNA. Right: the same two elements after
 * folding, touching. Same molecule, different notion of "near".
 */
export function FoldingFigure() {
  const reduced = useReducedMotion();
  const draw = {
    hidden: { pathLength: 0, opacity: 0 },
    show: {
      pathLength: 1,
      opacity: 1,
      transition: { duration: reduced ? 0 : 1.1, ease: [0.22, 1, 0.36, 1] as const },
    },
  };

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <figure className="rounded-xl border border-border/60 bg-card/30 p-4">
        <svg viewBox="0 0 420 150" className="h-auto w-full" role="img" aria-labelledby="fig-1d">
          <title id="fig-1d">
            A straight genomic track with an enhancer at one end and a promoter at the other,
            about a megabase apart.
          </title>
          <line x1="20" y1="95" x2="400" y2="95" stroke="var(--border)" strokeWidth="2" />
          {Array.from({ length: 39 }, (_, i) => (
            <rect
              key={i}
              x={20 + i * 10}
              y={91}
              width={3}
              height={8}
              rx={1}
              fill="var(--muted-foreground)"
              opacity={0.35}
            />
          ))}
          <circle cx="45" cy="95" r="8" fill="var(--head)" />
          <circle cx="378" cy="95" r="8" fill="var(--out)" />
          <text x="45" y="122" textAnchor="middle" fontSize="12" fill="var(--head)" fontFamily="var(--font-mono)">
            enhancer
          </text>
          <text x="378" y="122" textAnchor="middle" fontSize="12" fill="var(--out)" fontFamily="var(--font-mono)">
            promoter
          </text>
          <motion.line
            x1="53" y1="72" x2="370" y2="72"
            stroke="var(--muted-foreground)" strokeWidth="1" strokeDasharray="4 4"
            initial={{ opacity: 0 }} whileInView={{ opacity: 0.7 }} viewport={{ once: true }}
          />
          <text x="211" y="62" textAnchor="middle" fontSize="12" fill="var(--muted-foreground)" fontFamily="var(--font-mono)">
            ~1 Mb apart in sequence
          </text>
        </svg>
        <figcaption className="mt-2 font-mono text-[11px] text-muted-foreground">
          what a 1D sequence model sees
        </figcaption>
      </figure>

      <figure className="rounded-xl border border-border/60 bg-card/30 p-4">
        <svg viewBox="0 0 420 150" className="h-auto w-full" role="img" aria-labelledby="fig-3d">
          <title id="fig-3d">
            The same DNA folded into a loop, bringing the enhancer and promoter into physical
            contact.
          </title>
          <motion.path
            d="M 20 118 C 90 118, 120 20, 210 20 C 300 20, 330 118, 400 118"
            fill="none"
            stroke="var(--border)"
            strokeWidth="2.5"
            variants={draw}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-60px" }}
          />
          <motion.path
            d="M 150 44 C 175 30, 245 30, 270 44"
            fill="none"
            stroke="var(--struct)"
            strokeWidth="2"
            strokeDasharray="5 4"
            variants={draw}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-60px" }}
          />
          <circle cx="150" cy="44" r="8" fill="var(--head)" />
          <circle cx="270" cy="44" r="8" fill="var(--out)" />
          <text x="210" y="76" textAnchor="middle" fontSize="12" fill="var(--struct)" fontFamily="var(--font-mono)">
            in contact
          </text>
          <text x="210" y="136" textAnchor="middle" fontSize="12" fill="var(--muted-foreground)" fontFamily="var(--font-mono)">
            Hi-C measures this
          </text>
        </svg>
        <figcaption className="mt-2 font-mono text-[11px] text-muted-foreground">
          what the nucleus actually does
        </figcaption>
      </figure>
    </div>
  );
}
