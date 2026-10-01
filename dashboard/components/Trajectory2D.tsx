"use client";
import { useEffect, useMemo, useRef } from "react";
import { useTelemetry } from "./TelemetryProvider";

export function Trajectory2D() {
  const { frames, latest, showExpected } = useTelemetry();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const metric = latest?.translation_scale === "metric";
  const { points, truth } = useMemo(() => {
    const position = (pose: number[][]) => [pose[0][3], pose[2][3]];
    return { points: frames.map(f => position(f.pose_graph?.optimized_pose_T_wc ?? f.pose_T_wc)),
      truth: showExpected && metric ? frames.flatMap(f => f.expected_pose_T_wc ? [position(f.expected_pose_T_wc)] : []) : [] };
  }, [frames, metric, showExpected]);
  useEffect(() => {
    const canvas = canvasRef.current, ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const w = 520, h = 300;
    canvas.width = w; canvas.height = h;
    ctx.fillStyle = "#fafbfd"; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#e9edf2";
    for (let i = 1; i < 6; i++) {
      ctx.beginPath(); ctx.moveTo(i * w / 6, 0); ctx.lineTo(i * w / 6, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * h / 6); ctx.lineTo(w, i * h / 6); ctx.stroke();
    }
    ctx.font = "12px Segoe UI"; ctx.fillStyle = "#98a4b2";
    if (points.length < 2) { ctx.textAlign = "center"; ctx.fillText("Waiting for trajectory data", w / 2, h / 2); return; }
    const all = [...points, ...truth], xs = all.map(p => p[0]), zs = all.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
    const scale = Math.min((w - 70) / Math.max(1, maxX - minX), (h - 60) / Math.max(1, maxZ - minZ));
    const project = (p: number[]) => [w / 2 + (p[0] - (minX + maxX) / 2) * scale, h / 2 - (p[1] - (minZ + maxZ) / 2) * scale];
    const draw = (path: number[][], color: string) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
      path.forEach((p, i) => { const [x, y] = project(p); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke();
    };
    draw(truth, "#11a58b"); draw(points, "#5375dd");
    const [x, y] = project(points[points.length - 1]);
    ctx.fillStyle = "#5375dd"; ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    const units = Math.pow(10, Math.floor(Math.log10(90 / scale)));
    ctx.strokeStyle = "#8391a3"; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(w - 25 - units * scale, h - 18); ctx.lineTo(w - 25, h - 18); ctx.stroke();
    ctx.fillStyle = "#8391a3"; ctx.font = "10px Segoe UI"; ctx.textAlign = "right";
    ctx.fillText(`${units} ${metric ? "m" : "units"}`, w - 25, h - 25);
  }, [points, truth, metric]);
  return <section className="panel trajectoryPanel">
    <div className="panelHeading"><div><span className="eyebrow">TOP DOWN · X / Z</span><h2>Trajectory</h2></div><div className="chartLegend"><span><i />Estimate</span>{metric && showExpected && <span><i className="truth" />Ground truth</span>}</div></div>
    <div className="chartBody"><canvas ref={canvasRef} role="img" aria-label="Top-down trajectory with equal axis scale" /></div>
    <div className="chartFooter">{metric ? "Raw coordinates · meters · equal axis scale" : "Monocular coordinates · arbitrary scale"}</div>
  </section>;
}
