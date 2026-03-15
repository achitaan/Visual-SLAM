"use client";

import { useMemo, useState } from "react";
import { useTelemetry } from "./TelemetryProvider";

export function ControlBar() {
  const { latest, frames, sendControl, showExpected, setShowExpected, restart } = useTelemetry();
  const [overlayEnabled, setOverlayEnabled] = useState(true);
  const mode = latest?.mode ?? "slam";
  const streamEnabled = latest?.stream_enabled ?? true;

  const handleExport = () => {
    const payload = frames.map((frame) => ({
      frame_index: frame.frame_index,
      timestamp: frame.timestamp,
      pose_T_wc: frame.pose_T_wc,
    }));
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "trajectory.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  const statusLabel = useMemo(() => {
    return streamEnabled ? "Streaming" : "Paused";
  }, [streamEnabled]);

  const hasData = frames.length > 0;

  return (
    <div className="panel">
      <h2>Controls</h2>
      <div className="toolbar">
        <button className="button primary" onClick={restart}>
          Restart
        </button>
        <button
          className="button"
          onClick={() => sendControl({ type: "control", action: streamEnabled ? "stop" : "start" })}
        >
          {streamEnabled ? "Pause" : "Resume"}
        </button>
        <button
          className={`button ${mode === "slam" ? "active" : "secondary"}`}
          onClick={() =>
            sendControl({
              type: "control",
              action: "set_mode",
              mode: mode === "vo" ? "slam" : "vo",
            })
          }
        >
          {mode === "vo" ? "VO" : "SLAM"}
        </button>
        <button
          className="button secondary"
          onClick={() => {
            const next = !overlayEnabled;
            setOverlayEnabled(next);
            sendControl({ type: "control", action: "toggle_overlay", enabled: next });
          }}
        >
          Overlays
        </button>
        <button
          className={`button ${showExpected ? "active" : "secondary"}`}
          onClick={() => setShowExpected(!showExpected)}
        >
          GT
        </button>
        <button className="button secondary" onClick={handleExport} disabled={!hasData}>
          Export
        </button>
        <span className={`badge ${streamEnabled ? "live" : "stopped"}`}>
          {statusLabel} ({frames.length})
        </span>
      </div>
    </div>
  );
}
