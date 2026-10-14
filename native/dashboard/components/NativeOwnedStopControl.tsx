"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type ControlState = {
  schema: "native_application_control_v1";
  type: "control_hello" | "control_state";
  run_id: string;
  expected_frames: number;
  phase: "prepared" | "running" | "stopping" | "cancelled" | "completed" | "failed";
  status: string;
  cleanup_confirmed: boolean;
  incomplete: boolean;
  clear_is_display_only: true;
  capability?: string;
};

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
      (typeof value.capability !== "string" || !/^[a-f0-9]{64}$/.test(value.capability))) {
    throw new Error("The control hello did not contain this run's capability.");
  }
  return value as ControlState;
}

export function NativeOwnedStopControl({ controlUrl, runId, expectedFrames, emittedFrames, onClear }:
  { controlUrl: string; runId: string; expectedFrames: number; emittedFrames: number; onClear: () => void }) {
  const socket = useRef<WebSocket | null>(null);
  const capability = useRef("");
  const stateRef = useRef<ControlState | undefined>(undefined);
  const [state, setState] = useState<ControlState>();
  const [note, setNote] = useState("Waiting for the run-bound control hello.");

  useEffect(() => {
    let endpoint: string;
    try { endpoint = localControlUrl(controlUrl); }
    catch (error) { setNote(error instanceof Error ? error.message : "Invalid control endpoint."); return; }
    const current = new WebSocket(endpoint);
    socket.current = current;
    current.onmessage = event => {
      try {
        const next = parseState(JSON.parse(String(event.data)), runId, expectedFrames);
        if (next.type === "control_hello") capability.current = next.capability ?? "";
        else if (!capability.current) throw new Error("Control state arrived before the control hello.");
        stateRef.current = next; setState(next);
        setNote(next.status.replaceAll("_", " "));
      } catch (error) {
        capability.current = "";
        setNote(error instanceof Error ? error.message : "Invalid control response.");
        current.close();
      }
    };
    current.onclose = () => {
      socket.current = null;
      if (!stateRef.current?.cleanup_confirmed && stateRef.current?.phase !== "completed") {
        setNote(previous => previous.includes("cleanup confirmed") ? previous :
          "Control connection closed; cleanup is unconfirmed.");
      }
    };
    return () => { current.close(); if (socket.current === current) socket.current = null; };
  }, [controlUrl, runId, expectedFrames]);

  const stopAllowed = Boolean(capability.current && socket.current?.readyState === WebSocket.OPEN &&
    (state?.phase === "prepared" || state?.phase === "running"));
  const status = useMemo(() => {
    if (state?.phase === "cancelled" && state.cleanup_confirmed) return "Cancelled · cleanup confirmed · incomplete";
    if (state?.phase === "stopping") return "Stopping · cleanup unconfirmed";
    if (state?.phase === "completed") return "Run complete";
    if (state?.phase === "failed") return "Run failed · incomplete";
    if (state?.phase === "running") return "Native run active";
    return "Preparing native run";
  }, [state]);
  const stop = () => {
    if (!stopAllowed || !socket.current) return;
    socket.current.send(JSON.stringify({ action: "stop_run", run_id: runId,
      capability: capability.current }));
    setNote("Stop requested; waiting for the sealed whole-process cleanup audit.");
  };
  return <section className="sessionBar" aria-label="Native run control">
    <div className="sessionIdentity">
      <span className={`statusDot ${state?.phase === "running" ? "live" : ""}`} />
      <strong role="status" aria-live="polite" aria-atomic="true">{status}</strong>
      <span className="divider" /><span>Validated display prefix: {emittedFrames} / {expectedFrames}</span>
    </div>
    <div className="toolbar">
      <button className="button primary" disabled={!stopAllowed} onClick={stop}>Stop run</button>
      <button className="button" onClick={onClear}>Clear display</button>
    </div>
    <p style={{ width: "100%", margin: "8px 0 0", color: "var(--muted, #94a3b8)" }}>{note}</p>
  </section>;
}

