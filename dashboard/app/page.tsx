"use client";

import { useState } from "react";
import { ControlBar } from "@/components/ControlBar";
import { ErrorGraph } from "@/components/ErrorGraph";
import { EventLog } from "@/components/EventLog";
import { FrameViewer } from "@/components/FrameViewer";
import { MapSummary } from "@/components/MapSummary";
import { MapView3D } from "@/components/MapView3D";
import { MetricsPanel } from "@/components/MetricsPanel";
import { TelemetryProvider } from "@/components/TelemetryProvider";
import { Trajectory2D } from "@/components/Trajectory2D";

type RightTab = "metrics" | "error" | "events";

export default function Home() {
  const [rightTab, setRightTab] = useState<RightTab>("metrics");

  return (
    <TelemetryProvider>
      <div className="layout">
        <div className="stack">
          <ControlBar />
          <FrameViewer />
          <Trajectory2D />
        </div>
        <div className="rightPanel">
          <div className="tabBar">
            <button
              className={`tab ${rightTab === "metrics" ? "active" : ""}`}
              onClick={() => setRightTab("metrics")}
            >
              Metrics
            </button>
            <button
              className={`tab ${rightTab === "error" ? "active" : ""}`}
              onClick={() => setRightTab("error")}
            >
              Error
            </button>
            <button
              className={`tab ${rightTab === "events" ? "active" : ""}`}
              onClick={() => setRightTab("events")}
            >
              Events
            </button>
          </div>

          {rightTab === "metrics" && (
            <>
              <MetricsPanel />
              <MapView3D />
              <MapSummary />
            </>
          )}

          {rightTab === "error" && <ErrorGraph />}

          {rightTab === "events" && <EventLog />}
        </div>
      </div>
    </TelemetryProvider>
  );
}
