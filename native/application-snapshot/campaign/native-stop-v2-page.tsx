"use client";

import { useEffect, useState } from "react";
import { NativeOwnedStopControlV2 } from "@/components/NativeOwnedStopControlV2";
import { MapSummary } from "@/components/MapSummary";
import { MapView3D } from "@/components/MapView3D";
import { MetricsPanel } from "@/components/MetricsPanel";
import { TelemetryProvider, useTelemetry } from "@/components/TelemetryProvider";
import { Trajectory2D } from "@/components/Trajectory2D";
import { parseNativeBinding, type NativeRunBinding } from "@/lib/native-binding";

type StopBinding = NativeRunBinding & { controlUrl: string };

function parseStopBinding(search: string): StopBinding {
  const binding = parseNativeBinding(search);
  const params = new URLSearchParams(search);
  const controlUrl = params.get("control") ?? "";
  if (!controlUrl) throw new Error("The native Stop endpoint is missing.");
  return { ...binding, controlUrl };
}

function Workspace({ binding, error }: { binding?: StopBinding; error: string }) {
  const { latest, connected, phase, connectionNote, restart } = useTelemetry();
  const emitted = latest ? latest.frame_index + 1 : 0;
  return <main className="workspace" style={{ maxWidth: 1400, margin: "0 auto" }}>
    <header className="workspaceHeader"><div><div className="breadcrumb">ISOLATED NATIVE STOP V2 QUALIFICATION</div>
      <h1>Owned native run</h1><p>Live Linux stage data with bounded whole-process cleanup.</p></div>
      <div><span className={`connectionBadge ${connected ? "connected" : ""}`}>
        {connected ? "Verified native feed" : phase === "rejected" ? "Native feed rejected" : "Waiting for native data"}
      </span><p>{binding ? `KITTI ${binding.sequence} · expected ${binding.frames} frames` : "No run selected"}</p></div>
    </header>
    {(error || connectionNote) && <p role="status">{error || connectionNote}</p>}
    {binding && <NativeOwnedStopControlV2 controlUrl={binding.controlUrl} runId={binding.runId}
      expectedFrames={binding.frames} emittedFrames={emitted} onClear={restart} />}
    {latest ? <><MetricsPanel/><div className="analysisGrid"><Trajectory2D/><MapView3D/></div><MapSummary/></> :
      <p>No pose is synthesized for readiness. The first plot appears only after a qualified native frame arrives.</p>}
    <footer className="workspaceFooter">Stop is terminal for this run. Cancelled prefixes remain incomplete evidence. Clear affects only this display; it never resumes or restarts native processing.</footer>
  </main>;
}

export default function Page() {
  const [binding, setBinding] = useState<StopBinding>(), [error, setError] = useState("");
  useEffect(() => {
    try { setBinding(parseStopBinding(window.location.search)); setError(""); }
    catch (failure) { setBinding(undefined); setError(failure instanceof Error ? failure.message : "Invalid native Stop link."); }
  }, []);
  return <TelemetryProvider binding={binding}><Workspace binding={binding} error={error}/></TelemetryProvider>;
}

