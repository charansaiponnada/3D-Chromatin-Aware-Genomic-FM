"use client";

import { useSyncExternalStore } from "react";

/**
 * Why this exists rather than a CSS breakpoint.
 *
 * The mobile detail view is a Sheet, and shadcn's SheetContent always renders a
 * full-screen SheetOverlay alongside it. Hiding the content with `lg:hidden`
 * leaves that overlay in place, which dims the entire desktop page the moment a
 * component is selected. So the Sheet has to be unmounted, not hidden.
 *
 * useSyncExternalStore rather than state-in-an-effect: the store subscription
 * is the whole job, and React handles the server and hydration snapshots —
 * false until the browser takes over, which is the safe direction. The Sheet
 * mounts closed, renders nothing, and unmounts when the real value arrives.
 */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const mql = window.matchMedia(query);
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    },
    () => window.matchMedia(query).matches,
    // Server and first hydration render: assume the narrow layout.
    () => false,
  );
}
