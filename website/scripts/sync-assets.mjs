/**
 * Copy the repo's generated assets into the site, then check the site cannot
 * quote a config key that no longer exists.
 *
 * Runs on predev and prebuild.
 *
 * Why copy rather than symlink: Windows symlinks need Developer Mode or an
 * elevated shell, so a symlink here would work on one machine and fail on the
 * next. Why copy rather than import across `..`: on a subdirectory deploy the
 * build root is `website/`, and `../figures` escapes it.
 */

import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE = join(HERE, "..");
const REPO = join(SITE, "..");

const figuresSrc = join(REPO, "figures");
const figuresDst = join(SITE, "public", "figures");
const configSrc = join(REPO, "configs", "base.yaml");
const configDst = join(SITE, ".generated", "base.yaml");

function fail(message) {
  console.error(`\n  sync-assets: ${message}\n`);
  process.exit(1);
}

if (!existsSync(figuresSrc)) {
  fail(`figures/ not found at ${figuresSrc}. Run scripts/export_diagrams.py first.`);
}
if (!existsSync(configSrc)) {
  fail(`configs/base.yaml not found at ${configSrc}.`);
}

mkdirSync(figuresDst, { recursive: true });
cpSync(figuresSrc, figuresDst, { recursive: true });

mkdirSync(dirname(configDst), { recursive: true });
const rawConfig = readFileSync(configSrc, "utf8");
writeFileSync(configDst, rawConfig, "utf8");

// --- the check that earns this script its keep -----------------------------
// Architecture nodes cite config keys by dotted path. If a key is renamed in
// base.yaml, the site would quietly render "undefined" in a review room. Fail
// the build instead.
const config = parse(rawConfig);
const archSource = readFileSync(join(SITE, "src", "content", "architecture.ts"), "utf8");

const refs = new Set();
for (const block of archSource.matchAll(/configRefs:\s*\[([^\]]*)\]/g)) {
  for (const ref of block[1].matchAll(/["']([^"']+)["']/g)) refs.add(ref[1]);
}

const missing = [...refs].filter(
  (path) => path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), config) === undefined,
);

if (missing.length) {
  fail(
    `these configRefs in src/content/architecture.ts do not resolve in configs/base.yaml:\n` +
      missing.map((m) => `    - ${m}`).join("\n") +
      `\n  Either the key was renamed in base.yaml or the reference is a typo.`,
  );
}

console.log(
  `  sync-assets: figures copied, base.yaml staged, ${refs.size} config references verified.`,
);
