"use client";

import Link from "next/link";
import { motion, useScroll, useSpring } from "motion/react";
import { useEffect, useState } from "react";

import { SECTIONS } from "@/content/research";
import { cn } from "@/lib/utils";

/**
 * Sticky header with a scroll-progress rail, plus a scroll-spy rail on wide
 * screens.
 *
 * IntersectionObserver rather than scroll maths: it does not fire on every
 * frame, so scrolling stays smooth on a laptop trackpad.
 */
export function SiteNav() {
  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 140, damping: 30, mass: 0.4 });
  const [current, setCurrent] = useState<string>(SECTIONS[0].id);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setCurrent(visible[0].target.id);
      },
      // A band across the upper-middle of the viewport: the section the reader
      // is actually looking at, not whatever happens to touch the top edge.
      { rootMargin: "-20% 0px -65% 0px", threshold: 0 },
    );
    for (const s of SECTIONS) {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, []);

  return (
    <>
      <header className="fixed inset-x-0 top-0 z-40 border-b border-border/50 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between gap-4 px-5 md:px-8">
          <Link href="/" className="flex items-baseline gap-2">
            <span className="font-semibold tracking-tight">ChromGraphFM</span>
            <span className="hidden font-mono text-[11px] text-muted-foreground sm:inline">
              3D chromatin-conditioned genomic model
            </span>
          </Link>
          <nav className="flex items-center gap-4 font-mono text-xs">
            <Link
              href="/#architecture"
              className="text-muted-foreground transition-colors hover:text-foreground"
            >
              architecture
            </Link>
            <Link
              href="/architecture"
              className="text-muted-foreground transition-colors hover:text-foreground"
            >
              explorer
            </Link>
            <Link
              href="/#results"
              className="text-muted-foreground transition-colors hover:text-foreground"
            >
              results
            </Link>
          </nav>
        </div>
        <motion.div
          style={{ scaleX: progress }}
          className="h-px origin-left bg-primary"
          aria-hidden="true"
        />
      </header>

      <nav
        aria-label="Section navigation"
        className="fixed left-6 top-1/2 z-30 hidden -translate-y-1/2 flex-col gap-1.5 xl:flex"
      >
        {SECTIONS.map((s) => (
          <a
            key={s.id}
            href={`#${s.id}`}
            className="group flex items-center gap-2.5 py-0.5"
            aria-current={current === s.id ? "true" : undefined}
          >
            <span
              className={cn(
                "h-px transition-all duration-300",
                current === s.id ? "w-6 bg-primary" : "w-3 bg-muted-foreground/40",
              )}
            />
            <span
              className={cn(
                "font-mono text-[10px] transition-colors duration-300",
                current === s.id
                  ? "text-foreground"
                  : "text-muted-foreground/50 group-hover:text-muted-foreground",
              )}
            >
              {s.label}
            </span>
          </a>
        ))}
      </nav>
    </>
  );
}
