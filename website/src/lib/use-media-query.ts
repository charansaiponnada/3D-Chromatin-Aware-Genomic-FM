"use client";

import { useEffect, useState } from "react";

/**
 * Why this exists rather than a CSS breakpoint.
 *
 * The mobile detail view is a Sheet, and shadcn's SheetContent always renders a
 * full-screen SheetOverlay alongside it. Hiding the content with `lg:hidden`
 * leaves that overlay in place, which dims the entire desktop page the moment a
 * component is selected. So the Sheet has to be unmounted, not hidden.
 *
 * Returns false on the server and on first paint, which is the safe direction:
 * the Sheet mounts closed, renders nothing, and unmounts once the effect runs.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const mql = window.matchMedia(query);
    setMatches(mql.matches);
    const onChange = (e: MediaQueryListEvent) => setMatches(e.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}
