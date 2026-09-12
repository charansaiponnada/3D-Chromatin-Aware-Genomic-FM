import { Hero } from "@/components/sections/hero";
import {
  Experiments,
  References,
  Reproducibility,
  Results,
  Timeline,
} from "@/components/sections/evidence";
import { Gap, Literature } from "@/components/sections/literature";
import { Architecture, Data, Objectives3 } from "@/components/sections/method";
import { Aim, Objectives, Problem, Questions } from "@/components/sections/narrative";
import { publicConfig } from "@/lib/config";
import { nodeMathHtml, pageMathHtml } from "@/lib/math";
import { loadResults } from "@/lib/results";

/**
 * Server component. The config is parsed, the maths is rendered and the results
 * are read once at build time, so no `fs`, no `yaml` and no KaTeX engine ever
 * reaches the browser.
 */
export default function Home() {
  const config = publicConfig();
  const math = pageMathHtml();
  const nodeMath = nodeMathHtml();
  const results = loadResults();

  return (
    <>
      <Hero config={config} attentionHtml={math.attention} />
      <Problem />
      <Aim />
      <Questions />
      <Objectives />
      <Literature />
      <Gap />
      <Architecture config={config} mathHtml={nodeMath} attentionHtml={math.attention} />
      <Objectives3 config={config} math={math} />
      <Data config={config} />
      <Experiments />
      <Results results={results} />
      <Timeline />
      <Reproducibility config={config} />
      <References />
    </>
  );
}
