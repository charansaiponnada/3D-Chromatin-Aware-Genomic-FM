import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The React Compiler's memoisation stops motion's AnimatePresence from seeing
  // its children change, so the architecture detail panel never swapped in.
  // The site is a handful of static sections; there is nothing here the compiler
  // needs to optimise.
  reactCompiler: false,

  // Static export is configured now even though the site only runs locally today.
  // It costs one line and forces the discipline that keeps GitHub Pages a
  // configuration change later rather than a rewrite: no API routes, no
  // middleware, no runtime `fs`.
  output: "export",

  // next/image optimisation needs a server; without this the export throws.
  images: { unoptimized: true },

  // For a GitHub Pages *project* site, add:
  //   basePath: "/3D-Chromatin-Aware-Genomic-FM",
  //   assetPrefix: "/3D-Chromatin-Aware-Genomic-FM/",
  //   trailingSlash: true,
  // and create public/.nojekyll, or Pages drops the _next/ directory.
};

export default nextConfig;
