import "server-only";

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Results are read from the repo's results/ directory or they do not exist.
 *
 * This mirrors the rule the review decks already follow: no number reaches a
 * reader unless a file produced it. Everything else renders as "??". The
 * temptation to type a plausible placeholder into a slide or a page is exactly
 * how a project ends up defending a number nobody measured.
 */

export type ArmId = "b0" | "b1" | "b2" | "b3" | "b4" | "b5" | "ours";

export interface ArmMetrics {
  contact_r_long?: number;
  loop_auprc?: number;
  hic_free_probe?: number;
  seeds?: number;
}

export type ResultsTable = Partial<Record<ArmId, ArmMetrics>>;

const RESULTS_DIR = join(process.cwd(), "..", "results");

export function loadResults(): ResultsTable {
  const file = join(RESULTS_DIR, "final_comparison.json");
  if (!existsSync(file)) return {};
  try {
    return JSON.parse(readFileSync(file, "utf8")) as ResultsTable;
  } catch {
    // A malformed results file must not silently become "no results" — but it
    // also must not break the build during a review. Empty table, loud log.
    console.warn(`results: could not parse ${file}; rendering ?? instead.`);
    return {};
  }
}

export function hasAnyResults(table: ResultsTable): boolean {
  return Object.keys(table).length > 0;
}
