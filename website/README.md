# ChromGraphFM — project website

Explains the whole research project in one place: problem, aim, research
questions, objectives, literature review, the gap, the architecture,
pretraining objectives, data and splits, experiments, results, timeline.

```bash
npm install
npm run dev      # http://localhost:3000
npm run build    # static export to out/
```

## Two things worth knowing before editing

**Numbers come from `../configs/base.yaml`, never from this folder.** They are
parsed at build time by `src/lib/config.ts`. `scripts/sync-assets.mjs` runs on
predev and prebuild, copies the figures and the config in, and fails the build
if any `configRefs` path in `src/content/architecture.ts` no longer resolves.

**Results render as `??` until a file produces them.** `src/lib/results.ts`
reads `../results/final_comparison.json`; when it is absent every metric shows
`??`. Do not type a number into the site.

## Where things live

| Path | What |
|---|---|
| `src/content/architecture.ts` | The interactive diagram: nodes, edges, and every explanation |
| `src/content/research.ts` | Questions, objectives, literature, arms, phases, references |
| `src/components/architecture/` | The diagram, the detail panel, the contact-graph explorer |
| `src/app/architecture/page.tsx` | Full-screen explorer route |

Static export is configured but no `basePath` is set — see `next.config.ts` for
what GitHub Pages would additionally need.
