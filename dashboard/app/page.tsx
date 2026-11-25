"use client";

import { useState } from "react";

import { BenchmarkProgress } from "@/components/BenchmarkProgress";
import { BenchmarkView } from "@/components/BenchmarkView";
import { ControlBar } from "@/components/ControlBar";
import { ErrorGraph } from "@/components/ErrorGraph";
import { EventLog } from "@/components/EventLog";
import { FrameViewer } from "@/components/FrameViewer";
import { Icon } from "@/components/Icon";
import { MapSummary } from "@/components/MapSummary";
import { MapView3D } from "@/components/MapView3D";
import { MetricsPanel } from "@/components/MetricsPanel";
import { ReconstructionView } from "@/components/ReconstructionView";
import { SavedVisuals } from "@/components/SavedVisuals";
import { TelemetryProvider, useTelemetry } from "@/components/TelemetryProvider";
import { Trajectory2D } from "@/components/Trajectory2D";
import { LiveView } from "@/src/components/LiveView";

type TopTab = "realtime" | "explorer";
type ExplorerView = "benchmarks" | "reconstruction" | "events" | "legacy_live";

const explorerViews = [
  { id: "benchmarks", icon: "runs", label: "Benchmarks" },
  { id: "reconstruction", icon: "layers", label: "Reconstruction" },
  { id: "events", icon: "events", label: "Events" },
  { id: "legacy_live", icon: "camera", label: "Legacy WS" },
] as const;

function Workspace() {
  const [topTab, setTopTab] = useState<TopTab>("realtime");
  const [view, setView] = useState<ExplorerView>("benchmarks");
  const { latest, connected, benchmark } = useTelemetry();

  const titles: Record<ExplorerView, [string, string]> = {
    benchmarks: [
      "Benchmark results",
      "Accuracy measured against KITTI ground-truth trajectories.",
    ],
    reconstruction: [
      "Scene reconstruction",
      "Persistent landmarks and after-run dense point clouds.",
    ],
    events: [
      "Session activity",
      "Tracking changes, keyframes and optimization events.",
    ],
    legacy_live: [
      "Legacy telemetry stream",
      "Existing port-8765 offline benchmark playback monitor.",
    ],
  };

  return (
    <div className="appShell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Visual SLAM home">
          V<span>S</span>
        </a>
        <div className="navItems">
          <button
            className={`navItem ${topTab === "realtime" ? "active" : ""}`}
            aria-label="Live Real-Time SLAM"
            aria-current={topTab === "realtime" ? "page" : undefined}
            onClick={() => setTopTab("realtime")}
          >
            <Icon name="live" size={21} />
            <span>Real-Time</span>
          </button>
          {explorerViews.map((item) => (
            <button
              key={item.id}
              className={`navItem ${
                topTab === "explorer" && view === item.id ? "active" : ""
              }`}
              aria-label={item.label}
              aria-current={
                topTab === "explorer" && view === item.id ? "page" : undefined
              }
              onClick={() => {
                setTopTab("explorer");
                setView(item.id);
              }}
            >
              <Icon name={item.icon} size={21} />
              <span>{item.label}</span>
            </button>
          ))}
        </div>
        <div className="sidebarFooter">
          <span className={`statusDot ${connected ? "live" : ""}`} />
          <span>{connected ? "Online" : "Offline"}</span>
        </div>
      </aside>

      <main className="workspace">
        <header className="workspaceHeader">
          <div>
            <div className="breadcrumb">
              VISUAL SLAM<span>/</span>
              {topTab === "realtime"
                ? "LIVE REAL-TIME ENGINE"
                : "OFFLINE BENCHMARK EXPLORER"}
            </div>
            <h1>
              {topTab === "realtime"
                ? "Live Real-Time SLAM"
                : titles[view][0]}
            </h1>
            <p>
              {topTab === "realtime"
                ? "Decoupled async local BA, live 3D frustum & keyframe-anchored point cloud, and coalesced video streaming."
                : titles[view][1]}
            </p>
          </div>
          <div className="headerContext">
            <div
              className="segmented topModeSwitch"
              role="tablist"
              aria-label="Workspace mode"
            >
              <button
                role="tab"
                aria-selected={topTab === "realtime"}
                className={topTab === "realtime" ? "selected" : ""}
                onClick={() => setTopTab("realtime")}
              >
                Live Real-Time SLAM
              </button>
              <button
                role="tab"
                aria-selected={topTab === "explorer"}
                className={topTab === "explorer" ? "selected" : ""}
                onClick={() => setTopTab("explorer")}
              >
                Offline Benchmark Explorer
              </button>
            </div>
            {topTab === "explorer" && (
              <span className="pill">
                {latest?.translation_scale === "metric"
                  ? "Stereo · Metric"
                  : latest
                  ? "Monocular · Arbitrary scale"
                  : benchmark?.active
                  ? `KITTI ${benchmark.active.sequence} · ${
                      benchmark.active.sensor === "stereo"
                        ? "Stereo"
                        : "Monocular"
                    }`
                  : "Offline explorer"}
              </span>
            )}
          </div>
        </header>

        {topTab === "realtime" ? (
          <LiveView />
        ) : view === "legacy_live" ? (
          <>
            <BenchmarkProgress />
            <SavedVisuals />
            {(!benchmark || latest) && <ControlBar />}
            <MetricsPanel />
            <div className="visualGrid">
              <FrameViewer />
              <Trajectory2D />
            </div>
            <div className="analysisGrid">
              <MapView3D />
              <ErrorGraph />
            </div>
            <MapSummary />
            <footer className="workspaceFooter">
              <span>
                Camera-to-world pose convention · coordinates in reported scale
              </span>
              <span>
                Pause controls streaming. The pipeline continues processing.
              </span>
            </footer>
          </>
        ) : view === "benchmarks" ? (
          <>
            <BenchmarkProgress />
            <SavedVisuals />
            <BenchmarkView />
          </>
        ) : view === "reconstruction" ? (
          <ReconstructionView />
        ) : (
          <EventLog />
        )}
      </main>
    </div>
  );
}

export default function Home() {
  return (
    <TelemetryProvider>
      <Workspace />
    </TelemetryProvider>
  );
}
