"use client";

import { useEffect, useMemo, useRef } from "react";
import { useTelemetry } from "./TelemetryProvider";

function extractXYZ(pose: number[][] | undefined | null) {
  if (!pose || pose.length < 3) {
    return null;
  }
  return [pose[0][3] ?? 0, pose[1][3] ?? 0, pose[2][3] ?? 0];
}

export function ErrorGraph() {
  const { frames } = useTelemetry();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const errors = useMemo(() => {
    return frames
      .map((frame) => {
        const est = extractXYZ(frame.pose_graph?.optimized_pose_T_wc ?? frame.pose_T_wc);
        const exp = extractXYZ(frame.expected_pose_T_wc);
        if (!est || !exp) return null;
        const dx = est[0] - exp[0];
        const dy = est[1] - exp[1];
        const dz = est[2] - exp[2];
        return {
          frame: frame.frame_index,
          error: Math.sqrt(dx * dx + dy * dy + dz * dz),
        };
      })
      .filter((e): e is { frame: number; error: number } => e !== null);
  }, [frames]);

  const stats = useMemo(() => {
    if (errors.length === 0) return null;
    const vals = errors.map((e) => e.error);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const max = Math.max(...vals);
    const min = Math.min(...vals);
    const rmse = Math.sqrt(vals.reduce((a, b) => a + b * b, 0) / vals.length);
    return { mean, max, min, rmse, count: vals.length };
  }, [errors]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = 380;
    const height = 200;
    canvas.width = width;
    canvas.height = height;

    ctx.fillStyle = "#1e1e1e";
    ctx.fillRect(0, 0, width, height);

    if (errors.length < 2) {
      ctx.fillStyle = "#6a6a6a";
      ctx.font = "12px sans-serif";
      ctx.fillText("Waiting for ground truth data...", 20, height / 2);
      return;
    }

    const pad = { top: 20, right: 15, bottom: 30, left: 50 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const maxError = Math.max(...errors.map((e) => e.error), 0.1);
    const minFrame = errors[0].frame;
    const maxFrame = errors[errors.length - 1].frame;
    const frameRange = Math.max(maxFrame - minFrame, 1);

    ctx.strokeStyle = "#3a3a3a";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
    }

    ctx.fillStyle = "#6a6a6a";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const val = maxError * (1 - i / 4);
      const y = pad.top + (plotH / 4) * i;
      ctx.fillText(val.toFixed(2), pad.left - 5, y + 3);
    }

    ctx.textAlign = "center";
    ctx.fillText("Frame", width / 2, height - 5);
    ctx.save();
    ctx.translate(12, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Error (m)", 0, 0);
    ctx.restore();

    ctx.strokeStyle = "#ef5350";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    errors.forEach((e, i) => {
      const x = pad.left + ((e.frame - minFrame) / frameRange) * plotW;
      const y = pad.top + plotH - (e.error / maxError) * plotH;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    if (stats) {
      const meanY = pad.top + plotH - (stats.mean / maxError) * plotH;
      ctx.strokeStyle = "#4fc3f7";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.left, meanY);
      ctx.lineTo(width - pad.right, meanY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [errors, stats]);

  return (
    <div className="panel">
      <h2>Error Estimation</h2>
      <canvas ref={canvasRef} style={{ width: "100%", height: "auto" }} />
      {stats && (
        <div className="metricsGrid" style={{ marginTop: 8 }}>
          <div className="metricCard">
            <div>RMSE</div>
            <strong>{stats.rmse.toFixed(3)} m</strong>
          </div>
          <div className="metricCard">
            <div>Mean</div>
            <strong>{stats.mean.toFixed(3)} m</strong>
          </div>
          <div className="metricCard">
            <div>Max</div>
            <strong>{stats.max.toFixed(3)} m</strong>
          </div>
          <div className="metricCard">
            <div>Frames</div>
            <strong>{stats.count}</strong>
          </div>
        </div>
      )}
    </div>
  );
}
