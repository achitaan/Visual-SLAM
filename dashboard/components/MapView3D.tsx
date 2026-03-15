"use client";

import { useEffect, useRef, useState } from "react";
import { useTelemetry } from "./TelemetryProvider";

type Vec3 = { x: number; y: number; z: number };

function project(point: Vec3, yaw: number, pitch: number, scale: number, width: number, height: number) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);

  const x1 = cy * point.x + sy * point.z;
  const z1 = -sy * point.x + cy * point.z;
  const y2 = cp * point.y - sp * z1;
  const z2 = sp * point.y + cp * z1;

  const depth = Math.max(0.1, z2 + 5.0);
  const sx = (x1 / depth) * scale + width / 2;
  const sy2 = (y2 / depth) * scale + height / 2;
  return { x: sx, y: sy2 };
}

function projectGizmo(point: Vec3, yaw: number, pitch: number, scale: number, originX: number, originY: number) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);

  const x1 = cy * point.x + sy * point.z;
  const z1 = -sy * point.x + cy * point.z;
  const y2 = cp * point.y - sp * z1;
  const z2 = sp * point.y + cp * z1;

  const depth = Math.max(0.3, z2 + 3.0);
  const sx = (x1 / depth) * scale + originX;
  const sy2 = (y2 / depth) * scale + originY;
  return { x: sx, y: sy2 };
}

export function MapView3D() {
  const { latest, frames } = useTelemetry();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const draggingRef = useRef(false);
  const lastPosRef = useRef({ x: 0, y: 0 });
  const [view, setView] = useState({ yaw: 0.6, pitch: -0.4, zoom: 200 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    const width = 420;
    const height = 320;
    canvas.width = width;
    canvas.height = height;

    ctx.fillStyle = "#0f1420";
    ctx.fillRect(0, 0, width, height);

    const yaw = view.yaw;
    const pitch = view.pitch;
    const scale = view.zoom;

    const mapPoints = latest?.map_points ?? [];
    ctx.fillStyle = "rgba(79,156,255,0.6)";
    for (const p of mapPoints) {
      const [x, y, z] = p;
      const projected = project({ x, y, z }, yaw, pitch, scale, width, height);
      ctx.fillRect(projected.x, projected.y, 2, 2);
    }

    if (frames.length > 1) {
      ctx.strokeStyle = "#00d084";
      ctx.lineWidth = 2;
      ctx.beginPath();
      frames.forEach((frame, idx) => {
        const pose = frame.pose_graph?.optimized_pose_T_wc ?? frame.pose_T_wc;
        const point = { x: pose[0][3], y: pose[1][3], z: pose[2][3] };
        const projected = project(point, yaw, pitch, scale, width, height);
        if (idx === 0) {
          ctx.moveTo(projected.x, projected.y);
        } else {
          ctx.lineTo(projected.x, projected.y);
        }
      });
      ctx.stroke();
    }

    if (latest?.events && latest.events.some((event) => event.type === "relocalized")) {
      ctx.fillStyle = "#ffb74d";
      const pose = latest.pose_graph?.optimized_pose_T_wc ?? latest.pose_T_wc;
      const point = { x: pose[0][3], y: pose[1][3], z: pose[2][3] };
      const projected = project(point, yaw, pitch, scale, width, height);
      ctx.beginPath();
      ctx.arc(projected.x, projected.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Orientation gizmo (top-right)
    const gizmoSize = 70;
    const originX = width - gizmoSize;
    const originY = gizmoSize;
    ctx.strokeStyle = "#2c3246";
    ctx.lineWidth = 1;
    ctx.strokeRect(width - gizmoSize * 2, 10, gizmoSize * 2 - 10, gizmoSize * 2 - 10);

    const axisScale = 40;
    const centerX = width - gizmoSize;
    const centerY = gizmoSize;
    const axes = [
      { axis: "X", color: "#ff6b6b", vec: { x: 1, y: 0, z: 0 } },
      { axis: "Y", color: "#00d084", vec: { x: 0, y: 1, z: 0 } },
      { axis: "Z", color: "#4f9cff", vec: { x: 0, y: 0, z: 1 } },
    ];

    axes.forEach(({ axis, color, vec }) => {
      const tip = projectGizmo(vec, yaw, pitch, axisScale, centerX, centerY);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(tip.x, tip.y);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#e6e6e6";
      ctx.font = "10px Segoe UI";
      ctx.fillText(axis, tip.x + 6, tip.y - 6);
    });
  }, [latest, frames, view]);

  const onMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    draggingRef.current = true;
    lastPosRef.current = { x: event.clientX, y: event.clientY };
  };

  const onMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!draggingRef.current) {
      return;
    }
    const dx = event.clientX - lastPosRef.current.x;
    const dy = event.clientY - lastPosRef.current.y;
    lastPosRef.current = { x: event.clientX, y: event.clientY };
    setView((prev) => ({
      ...prev,
      yaw: prev.yaw + dx * 0.005,
      pitch: Math.max(-1.4, Math.min(1.4, prev.pitch + dy * 0.005)),
    }));
  };

  const onMouseUp = () => {
    draggingRef.current = false;
  };

  const onWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const delta = event.deltaY > 0 ? -10 : 10;
    setView((prev) => ({
      ...prev,
      zoom: Math.max(80, Math.min(600, prev.zoom + delta)),
    }));
  };

  const onClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const width = canvas.width;
    const gizmoSize = 70;
    const gizmoLeft = width - gizmoSize * 2;
    const gizmoTop = 10;
    const gizmoRight = width - 10;
    const gizmoBottom = gizmoSize * 2;
    if (x >= gizmoLeft && x <= gizmoRight && y >= gizmoTop && y <= gizmoBottom) {
      if (y < gizmoTop + gizmoSize) {
        // top half -> top-down
        setView((prev) => ({ ...prev, yaw: 0, pitch: -1.2 }));
      } else if (x < gizmoLeft + gizmoSize) {
        // left half -> side view
        setView((prev) => ({ ...prev, yaw: -1.57, pitch: -0.3 }));
      } else {
        // right/bottom -> front view
        setView((prev) => ({ ...prev, yaw: 0.0, pitch: -0.3 }));
      }
    }
  };

  return (
    <div className="panel">
      <h2>3D Map View</h2>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "auto", cursor: "grab" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onWheel={onWheel}
        onClick={onClick}
      />
    </div>
  );
}
