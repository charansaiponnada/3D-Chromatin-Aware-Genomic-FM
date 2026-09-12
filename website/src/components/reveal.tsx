"use client";

import { motion, useReducedMotion } from "motion/react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/**
 * Scroll-reveal wrapper.
 *
 * `useReducedMotion` collapses this to a plain fade for anyone who has asked
 * their OS for less movement — the content still appears, it just does not
 * travel. A reveal that ignores that setting can trigger nausea, so the check
 * is not decorative.
 */
export function Reveal({
  children,
  className,
  delay = 0,
  y = 18,
  as: _as,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
  y?: number;
  as?: never;
}) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      className={cn(className)}
      initial={{ opacity: 0, y: reduced ? 0 : y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: reduced ? 0.2 : 0.5, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

/** Staggers its children, each one revealing shortly after the last. */
export function RevealGroup({
  children,
  className,
  stagger = 0.06,
}: {
  children: ReactNode[];
  className?: string;
  stagger?: number;
}) {
  return (
    <div className={cn(className)}>
      {children.map((child, i) => (
        <Reveal key={i} delay={i * stagger}>
          {child}
        </Reveal>
      ))}
    </div>
  );
}
