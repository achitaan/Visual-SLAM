"use client";
import { useEffect, useRef, useState } from "react";
import { useTelemetry } from "./TelemetryProvider";

const initialView = { yaw: .6, pitch: -.65, zoom: 1 };
export function MapView3D() {
  const { latest, frames } = useTelemetry();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const [view, setView] = useState(initialView);
  useEffect(() => {
    const canvas = canvasRef.current, ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const w = 700, h = 280;
    canvas.width = w; canvas.height = h;
    ctx.fillStyle = "#fafbfd"; ctx.fillRect(0, 0, w, h);
    const rotate = (p: number[]) => {
      const x = Math.cos(view.yaw) * p[0] + Math.sin(view.yaw) * p[2];
      const z = -Math.sin(view.yaw) * p[0] + Math.cos(view.yaw) * p[2];
      return [x, Math.cos(view.pitch) * p[1] - Math.sin(view.pitch) * z];
    };
    const path = latest?.trajectory ?? frames.map(f => { const p = f.pose_graph?.optimized_pose_T_wc ?? f.pose_T_wc; return [p[0][3], p[1][3], p[2][3]]; });
    const landmarks = latest?.map_points ?? [], all = [...path, ...landmarks].map(rotate);
    const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
    const minX = Math.min(...xs, 0), maxX = Math.max(...xs, 0), minY = Math.min(...ys, 0), maxY = Math.max(...ys, 0);
    const scale = Math.min((w - 100) / Math.max(1, maxX - minX), (h - 65) / Math.max(1, maxY - minY)) * view.zoom;
    const project = (p: number[]) => { const q = rotate(p); return [w / 2 + (q[0] - (minX + maxX) / 2) * scale, h / 2 + (q[1] - (minY + maxY) / 2) * scale]; };
    ctx.strokeStyle = "#e9edf2"; ctx.lineWidth = 1;
    for (let i = 1; i < 8; i++) { ctx.beginPath(); ctx.moveTo(i * w / 8, 0); ctx.lineTo(i * w / 8, h); ctx.stroke(); }
    for (let i = 1; i < 5; i++) { ctx.beginPath(); ctx.moveTo(0, i * h / 5); ctx.lineTo(w, i * h / 5); ctx.stroke(); }
    ctx.fillStyle = "#97b6d0";
    landmarks.forEach(p => { const [x, y] = project(p); ctx.fillRect(x, y, 2, 2); });
    ctx.strokeStyle = "#5375dd"; ctx.lineWidth = 2.2; ctx.beginPath();
    path.forEach((p, i) => { const [x, y] = project(p); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke();
    if (path.length) { const [x, y] = project(path[path.length - 1]); ctx.fillStyle = "#11a58b"; ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill(); }
    else { ctx.fillStyle = "#98a4b2"; ctx.font = "12px Segoe UI"; ctx.textAlign = "center"; ctx.fillText("Waiting for pose and map data", w / 2, h / 2); }
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((axis, i) => {
      const [x, y] = rotate(axis), colors = ["#cd7676", "#11a58b", "#5375dd"];
      ctx.strokeStyle = colors[i]; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(w - 50, 45); ctx.lineTo(w - 50 + x * 24, 45 + y * 24); ctx.stroke();
      ctx.fillStyle = colors[i]; ctx.font = "10px Segoe UI"; ctx.textAlign = "center"; ctx.fillText("XYZ"[i], w - 50 + x * 32, 49 + y * 32);
    });
  }, [latest, frames, view]);
  return <section className="panel mapPanel"><div className="panelHeading"><div><span className="eyebrow">SPATIAL VIEW</span><h2>Trajectory & landmarks</h2></div><button className="button" onClick={() => setView(initialView)}>Reset view</button></div>
    <div className="chartBody"><canvas ref={canvasRef} role="img" aria-label="Rotatable 3D trajectory and available landmarks" style={{ touchAction: "none" }}
      onPointerDown={e => { drag.current = { x: e.clientX, y: e.clientY }; e.currentTarget.setPointerCapture(e.pointerId); }}
      onPointerMove={e => { if (!drag.current) return; const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y; drag.current = { x: e.clientX, y: e.clientY }; setView(v => ({ ...v, yaw: v.yaw + dx * .006, pitch: Math.max(-1.5, Math.min(1.5, v.pitch + dy * .006)) })); }}
      onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}
    /></div><div className="chartFooter mapToolbar"><span>Drag to orbit · {latest?.map_points?.length ?? 0} landmarks</span><div><button className="button" aria-label="Zoom out map" onClick={() => setView(v => ({ ...v, zoom: Math.max(.3, v.zoom / 1.3) }))}>−</button><button className="button" aria-label="Zoom in map" onClick={() => setView(v => ({ ...v, zoom: Math.min(8, v.zoom * 1.3) }))}>+</button></div></div>
  </section>;
}
