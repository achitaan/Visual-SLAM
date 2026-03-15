"use client";

import { useMemo } from "react";
import { useTelemetry } from "./TelemetryProvider";

function extractXYZ(pose: number[][] | undefined | null) {
  if (!pose || pose.length < 3) {
    return null;
  }
  return [pose[0][3] ?? 0, pose[1][3] ?? 0, pose[2][3] ?? 0];
}

export function MetricsPanel() {
  const { latest } = useTelemetry();
  const tracking = latest?.tracking;
  const lastEvent = latest?.events && latest.events.length > 0 ? latest.events[latest.events.length - 1] : undefined;
  const trackingStatus = lastEvent?.type === "relocalized"
    ? "Relocalized"
    : lastEvent?.type === "relocalization_failed"
      ? "Reloc Failed"
      : lastEvent?.type === "tracking_lost"
        ? "Tracking Lost"
        : "OK";

  const errorStats = useMemo(() => {
    if (!latest?.expected_pose_T_wc || !latest?.pose_T_wc) {
      return null;
    }
    const est = extractXYZ(latest.pose_graph?.optimized_pose_T_wc ?? latest.pose_T_wc);
    const exp = extractXYZ(latest.expected_pose_T_wc);
    if (!est || !exp) {
      return null;
    }
    const dx = est[0] - exp[0];
    const dy = est[1] - exp[1];
    const dz = est[2] - exp[2];
    const err = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return { rmse: err };
  }, [latest]);

  return (
    <div className="panel">
      <h2>Live Metrics</h2>
      <div className="metricsGrid">
        <div className="metricCard">
          <div>FPS</div>
          <strong>{latest?.fps?.toFixed(1) ?? "--"}</strong>
        </div>
        <div className="metricCard">
          <div>Matches</div>
          <strong>{tracking?.num_matches ?? "--"}</strong>
        </div>
        <div className="metricCard">
          <div>Inliers</div>
          <strong>{tracking?.num_inliers ?? "--"}</strong>
        </div>
        <div className="metricCard">
          <div>Inlier Ratio</div>
          <strong>
            {tracking?.inlier_ratio !== undefined && tracking?.inlier_ratio !== null
              ? `${(tracking.inlier_ratio * 100).toFixed(1)}%`
              : "--"}
          </strong>
        </div>
        <div className="metricCard">
          <div>Reproj Error</div>
          <strong>{tracking?.reprojection_error?.toFixed(2) ?? "--"}</strong>
        </div>
        <div className="metricCard">
          <div>Keyframes</div>
          <strong>{latest?.map?.keyframes ?? 0}</strong>
        </div>
        <div className="metricCard">
          <div>Tracking</div>
          <strong>{trackingStatus}</strong>
        </div>
        {errorStats ? (
          <div className="metricCard">
            <div>GT Error</div>
            <strong>{errorStats.rmse.toFixed(3)} m</strong>
          </div>
        ) : null}
      </div>
    </div>
  );
}
