import { useCallback, useEffect, useRef, useState } from "react";

export type AsyncState<T> = { status: "loading" } | { status: "error"; error: unknown } | { status: "ready"; data: T };

// Runs fn on mount and whenever deps change; aborts stale requests; exposes reload().
export function useAsync<T>(fn: (signal: AbortSignal) => Promise<T>, deps: unknown[]): [AsyncState<T>, () => void] {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading" });
  const [tick, setTick] = useState(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  useEffect(() => {
    const ctrl = new AbortController();
    setState((s) => (s.status === "ready" ? s : { status: "loading" }));
    fnRef.current(ctrl.signal).then(
      (data) => { if (!ctrl.signal.aborted) setState({ status: "ready", data }); },
      (error: unknown) => { if (!ctrl.signal.aborted) setState({ status: "error", error }); },
    );
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return [state, reload];
}
