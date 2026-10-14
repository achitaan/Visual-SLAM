"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type ControlPhase = "prepared" | "running" | "stopping" | "cancelled" | "completed" | "failed";

type ControlState = {
  schema: "native_application_control_v1";
  type: "control_hello" | "control_state";
  run_id: string;
  expected_frames: number;
  phase: ControlPhase;
  status: string;
  cleanup_confirmed: boolean;
  incomplete: boolean;
  clear_is_display_only: true;
  capability?: string;
};

const OPEN = 1;
const CAPABILITY = /^[a-f0-9]{64}$/;
const object = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

function localControlUrl(raw: string): string {
  const url = new URL(raw);
  if (url.protocol !== "ws:" || !["127.0.0.1", "localhost"].includes(url.hostname) ||
      !/^\d+$/.test(url.port) || Number(url.port) < 1024 || Number(url.port) > 65535 ||
      url.username || url.password || url.pathname !== "/" || url.search || url.hash) {
    throw new Error("The run link must contain its loopback control endpoint.");
  }
  return url.toString();
}

function parseState(value: unknown, runId: string, expectedFrames: number): ControlState {
  if (!object(value) || value.schema !== "native_application_control_v1" ||
      (value.type !== "control_hello" && value.type !== "control_state") ||
      value.run_id !== runId || value.expected_frames !== expectedFrames ||
      !["prepared", "running", "stopping", "cancelled", "completed", "failed"].includes(String(value.phase)) ||
      typeof value.status !== "string" || typeof value.cleanup_confirmed !== "boolean" ||
      typeof value.incomplete !== "boolean" || value.clear_is_display_only !== true) {
    throw new Error("The control endpoint did not identify this native run.");
  }
  if (value.type === "control_hello" &&
      (typeof value.capability !== "string" || !CAPABILITY.test(value.capability))) {
    throw new Error("The control hello did not contain this run's capability.");
  }
  return value as ControlState;
}

type Props = {
  controlUrl: string;
  runId: string;
  expectedFrames: number;
  emittedFrames: number;
  onClear: () => void;
};

export function NativeOwnedStopControlV2({
  controlUrl,
  runId,
  expectedFrames,
  emittedFrames,
  onClear,
}: Props) {
  const socket = useRef<WebSocket | null>(null);
  const capability = useRef("");
  const stateRef = useRef<ControlState | undefined>(undefined);
  const readyRef = useRef(false);
  const [state, setState] = useState<ControlState>();
  const [connectionReady, setConnectionReady] = useState(false);
  const [note, setNote] = useState("Waiting for the run-bound control hello.");

  useEffect(() => {
    // Clear all per-connection state before creating a replacement. This also
    // makes Strict Mode's setup → cleanup → setup sequence deterministic.
    capability.current = "";
    stateRef.current = undefined;
    readyRef.current = false;
    setState(undefined);
    setConnectionReady(false);

    let endpoint: string;
    try {
      endpoint = localControlUrl(controlUrl);
    } catch (error) {
      setNote(error instanceof Error ? error.message : "Invalid control endpoint.");
      return;
    }

    let disposed = false;
    const current = new WebSocket(endpoint);
    socket.current = current;
    const isCurrent = () => !disposed && socket.current === current;
    const disconnect = (reason: string) => {
      const last = stateRef.current;
      // A transport close cannot revoke an already authenticated terminal
      // outcome. It also cannot supply cleanup evidence for a live run.
      const terminal = last && ((last.phase === "cancelled" && last.cleanup_confirmed) ||
        last.phase === "completed");
      capability.current = "";
      readyRef.current = false;
      setConnectionReady(false);
      if (!terminal) {
        stateRef.current = undefined;
        setState(undefined);
      }
      setNote(terminal ? `${reason}; the last confirmed terminal result is retained.` :
        `${reason}; cleanup is unconfirmed.`);
    };

    current.onopen = () => {
      if (!isCurrent()) return;
      readyRef.current = true;
      setConnectionReady(true);
      setNote("Control connection open; waiting for the control hello.");
    };

    current.onmessage = event => {
      if (!isCurrent()) return;
      try {
        const next = parseState(JSON.parse(String(event.data)), runId, expectedFrames);
        if (next.type === "control_hello") {
          capability.current = next.capability ?? "";
        } else if (!CAPABILITY.test(capability.current)) {
          throw new Error("Control state arrived before the control hello.");
        }
        stateRef.current = next;
        setState(next);
        setConnectionReady(readyRef.current && current.readyState === OPEN);
        setNote(next.status.replaceAll("_", " "));
      } catch (error) {
        if (!isCurrent()) return;
        capability.current = "";
        stateRef.current = undefined;
        readyRef.current = false;
        setConnectionReady(false);
        setState(undefined);
        setNote(error instanceof Error ? error.message : "Invalid control response.");
        current.close();
      }
    };

    current.onerror = () => {
      if (!isCurrent()) return;
      disconnect("Control connection error");
      current.close();
    };

    current.onclose = () => {
      if (!isCurrent()) return;
      socket.current = null;
      disconnect("Control connection closed");
    };

    return () => {
      disposed = true;
      current.onopen = null;
      current.onmessage = null;
      current.onerror = null;
      current.onclose = null;
      current.close();
      if (socket.current === current) {
        socket.current = null;
        capability.current = "";
        stateRef.current = undefined;
        readyRef.current = false;
        setConnectionReady(false);
        setState(undefined);
      }
    };
  }, [controlUrl, runId, expectedFrames]);

  const stopAllowed = Boolean(
    connectionReady &&
    capability.current &&
    CAPABILITY.test(capability.current) &&
    socket.current?.readyState === OPEN &&
    (state?.phase === "prepared" || state?.phase === "running"),
  );

  const status = useMemo(() => {
    if (state?.phase === "cancelled" && state.cleanup_confirmed) {
      return "Cancelled · cleanup confirmed · incomplete";
    }
    if (state?.phase === "stopping") return "Stopping · cleanup unconfirmed";
    if (state?.phase === "completed") return "Run complete";
    if (state?.phase === "failed") return "Run failed · incomplete";
    if (state?.phase === "running") return "Native run active";
    return "Preparing native run";
  }, [state]);

  const stop = () => {
    // Re-read every authority at click time. A render-time boolean can be
    // stale when a replacement socket or a late callback wins the event loop.
    const current = socket.current;
    const currentState = stateRef.current;
    const currentCapability = capability.current;
    if (!readyRef.current || !current || socket.current !== current ||
        current.readyState !== OPEN || !CAPABILITY.test(currentCapability) ||
        !currentState || (currentState.phase !== "prepared" && currentState.phase !== "running")) {
      return;
    }
    current.send(JSON.stringify({
      action: "stop_run",
      run_id: runId,
      capability: currentCapability,
    }));
    setNote("Stop requested; waiting for the sealed whole-process cleanup audit.");
  };

  return <section className="sessionBar" aria-label="Native run control">
    <div className="sessionIdentity">
      <span className={`statusDot ${state?.phase === "running" ? "live" : ""}`} />
      <strong role="status" aria-live="polite" aria-atomic="true">{status}</strong>
      <span className="divider" />
      <span>Validated display prefix: {emittedFrames} / {expectedFrames}</span>
    </div>
    <div className="toolbar">
      <button type="button" className="button primary" disabled={!stopAllowed} onClick={stop}>
        Stop run
      </button>
      <button type="button" className="button" onClick={onClear}>Clear display</button>
    </div>
    <p style={{ width: "100%", margin: "8px 0 0", color: "var(--muted, #94a3b8)" }}>{note}</p>
  </section>;
}

