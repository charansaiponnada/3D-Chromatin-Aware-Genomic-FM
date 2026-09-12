import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { SiteNav } from "@/components/site-nav";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ChromGraphFM — a 3D chromatin-conditioned genomic model",
  description:
    "An end-to-end genomic representation model in which measured Hi-C contacts bias attention inside the sequence encoder throughout self-supervised pretraining.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    // The site commits to a single dark treatment rather than shipping a toggle:
    // the contact-graph arcs and the attention pulse were tuned to read as light
    // on a dark ground, and a light variant would need its own colour pass.
    <html
      lang="en"
      className={`dark ${geistSans.variable} ${geistMono.variable} h-full scroll-smooth antialiased`}
      suppressHydrationWarning
    >
      <body className="flex min-h-full flex-col">
        <SiteNav />
        <main className="flex-1">{children}</main>
        <footer className="border-t border-border/60 py-10">
          <div className="mx-auto flex w-full max-w-6xl flex-col gap-2 px-5 md:px-8">
            <p className="text-sm text-muted-foreground">
              ChromGraphFM — B.Tech Mini Project 23CS7552, Department of Computer Science and
              Engineering, VR Siddhartha Engineering College, Vijayawada.
            </p>
            <p className="font-mono text-[11px] text-muted-foreground/70">
              P. Charan Sai · G. Karthik · P. Sanath — under the guidance of Mrs. K. Divya
              Kothapali
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
