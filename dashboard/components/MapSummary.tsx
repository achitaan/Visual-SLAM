"use client";

import { useTelemetry } from "./TelemetryProvider";

export function MapSummary() {
  const { latest } = useTelemetry();
  return (
    <div className="panel">
      <h2>Map Summary</h2>
      <div className="metricsGrid">
        <div className="metricCard">
          <div>Keyframes</div>
          <strong>{latest?.map?.keyframes ?? 0}</strong>
        </div>
        <div className="metricCard">
          <div>Map Points</div>
          <strong>{latest?.map?.map_points ?? 0}</strong>
        </div>
        <div className="metricCard">
          <div>Mode</div>
          <strong>{latest?.mode ?? "vo"}</strong>
        </div>
      </div>
    </div>
  );
}
