"use client";

import { useEffect, useMemo, useRef } from "react";
import { useTelemetry } from "./TelemetryProvider";

function extractXYZ(pose: number[][]) {
  if (!pose || pose.length < 3) {
    return { x: 0, y: 0, z: 0 };
  }
  return { x: pose[0][3] ?? 0, y: pose[1][3] ?? 0, z: pose[2][3] ?? 0 };
}

type Vec2 = { x: number; z: number };

function toXZ(p: { x: number; y: number; z: number }): Vec2 {
  return { x: p.x, z: p.z };
}

function umeyama2D(src: Vec2[], dst: Vec2[]) {
  const n = Math.min(src.length, dst.length);
  if (n < 2) {
    return null;
  }
  const srcMean = {
    x: src.reduce((sum, p) => sum + p.x, 0) / n,
    z: src.reduce((sum, p) => sum + p.z, 0) / n,
  };
  const dstMean = {
    x: dst.reduce((sum, p) => sum + p.x, 0) / n,
    z: dst.reduce((sum, p) => sum + p.z, 0) / n,
  };

  let sxx = 0;
  let szz = 0;
  let sxz = 0;
  let szx = 0;
  let varSrc = 0;
  for (let i = 0; i < n; i++) {
    const xs = src[i].x - srcMean.x;
    const zs = src[i].z - srcMean.z;
    const xd = dst[i].x - dstMean.x;
    const zd = dst[i].z - dstMean.z;
    sxx += xd * xs;
    szz += zd * zs;
    sxz += xd * zs;
    szx += zd * xs;
    varSrc += xs * xs + zs * zs;
  }
  if (varSrc < 1e-6) {
    return null;
  }
  const a = sxx + szz;
  const b = sxz - szx;
  const rNorm = Math.hypot(a, b);
  const cos = a / rNorm;
  const sin = b / rNorm;
  const scale = rNorm / varSrc;
  const tx = dstMean.x - scale * (cos * srcMean.x - sin * srcMean.z);
  const tz = dstMean.z - scale * (sin * srcMean.x + cos * srcMean.z);
  return { scale, cos, sin, tx, tz };
}

function applyAlignment2D(p: Vec2, alignment: ReturnType<typeof umeyama2D>) {
  if (!alignment) {
    return p;
  }
  const { scale, cos, sin, tx, tz } = alignment;
  const x = scale * (cos * p.x - sin * p.z) + tx;
  const z = scale * (sin * p.x + cos * p.z) + tz;
  return { x, z };
}

export function Trajectory2D() {
  const { frames, showExpected } = useTelemetry();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const { points, expectedPoints, alignment } = useMemo(() => {
    const latestOptimized = [...frames]
      .reverse()
      .find((frame) => frame.pose_graph?.optimized_poses && frame.pose_graph.optimized_poses.length > 0)
      ?.pose_graph?.optimized_poses;

    const est3d = latestOptimized
      ? latestOptimized.map((pose) => extractXYZ(pose))
      : frames.map((frame) => {
          const optimized = frame.pose_graph?.optimized_pose_T_wc;
          return extractXYZ(optimized ?? frame.pose_T_wc);
        });
    const exp3d = showExpected
      ? frames
          .map((frame) => frame.expected_pose_T_wc)
          .filter((pose): pose is number[][] => Boolean(pose))
          .map((pose) => extractXYZ(pose))
      : [];

    const est2d = est3d.map(toXZ);
    const exp2d = exp3d.map(toXZ);
    const windowSize = 300;
    const estWin = est2d.slice(-windowSize);
    const expWin = exp2d.slice(-windowSize);
    const align = expWin.length >= 2 ? umeyama2D(estWin, expWin) : null;
    let estAligned = align ? est2d.map((p) => applyAlignment2D(p, align)) : est2d;
    if (exp2d.length > 0 && estAligned.length > 0) {
      const dx = exp2d[0].x - estAligned[0].x;
      const dz = exp2d[0].z - estAligned[0].z;
      estAligned = estAligned.map((p) => ({ x: p.x + dx, z: p.z + dz }));
    }
    return { points: estAligned, expectedPoints: exp2d, alignment: align };
  }, [frames, showExpected]);

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

    ctx.fillStyle = "#1e1e1e";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (height / 4) * i);
      ctx.lineTo(width, (height / 4) * i);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo((width / 4) * i, 0);
      ctx.lineTo((width / 4) * i, height);
      ctx.stroke();
    }

    if (frames.length < 2) {
      ctx.fillStyle = "#6a6a6a";
      ctx.font = "12px sans-serif";
      ctx.fillText("Waiting for data...", 20, height / 2);
      return;
    }

    const allPoints = expectedPoints.length > 0 ? [...points, ...expectedPoints] : points;
    const xs = allPoints.map((p) => p.x);
    const zs = allPoints.map((p) => p.z);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);

    const pad = 20;
    const scaleX = (width - pad * 2) / Math.max(1e-6, maxX - minX);
    const scaleZ = (height - pad * 2) / Math.max(1e-6, maxZ - minZ);

    if (showExpected && expectedPoints.length > 1) {
      ctx.strokeStyle = "#66bb6a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      expectedPoints.forEach((p, index) => {
        const x = pad + (p.x - minX) * scaleX;
        const y = height - (pad + (p.z - minZ) * scaleZ);
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }

    ctx.strokeStyle = "#4fc3f7";
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p, index) => {
      const x = pad + (p.x - minX) * scaleX;
      const y = height - (pad + (p.z - minZ) * scaleZ);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    const last = points[points.length - 1];
    const lastX = pad + (last.x - minX) * scaleX;
    const lastY = height - (pad + (last.z - minZ) * scaleZ);
    ctx.fillStyle = "#ef5350";
    ctx.beginPath();
    ctx.arc(lastX, lastY, 5, 0, Math.PI * 2);
    ctx.fill();

    const rangeX = maxX - minX;
    const rangeZ = maxZ - minZ;
    const maxRange = Math.max(rangeX, rangeZ, 1);
    const scaleBarWorld = Math.pow(10, Math.floor(Math.log10(maxRange / 2)));
    const scaleBarPx = scaleBarWorld * Math.min(scaleX, scaleZ);
    
    ctx.strokeStyle = "#888";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(width - pad - scaleBarPx, height - 12);
    ctx.lineTo(width - pad, height - 12);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(width - pad - scaleBarPx, height - 8);
    ctx.lineTo(width - pad - scaleBarPx, height - 16);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(width - pad, height - 8);
    ctx.lineTo(width - pad, height - 16);
    ctx.stroke();

    ctx.fillStyle = "#888";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${scaleBarWorld}m`, width - pad - scaleBarPx / 2, height - 2);

    ctx.fillStyle = "#4fc3f7";
    ctx.fillRect(pad, height - 14, 10, 10);
    ctx.fillStyle = "#888";
    ctx.textAlign = "left";
    ctx.fillText("Est", pad + 14, height - 5);

    if (showExpected) {
      ctx.fillStyle = "#66bb6a";
      ctx.fillRect(pad + 45, height - 14, 10, 10);
      ctx.fillStyle = "#888";
      ctx.fillText("GT", pad + 59, height - 5);
    }
  }, [frames, showExpected, points, expectedPoints, alignment]);

  return (
    <div className="panel">
      <h2>Trajectory (Top-Down)</h2>
      <canvas ref={canvasRef} style={{ width: "100%", height: "auto" }} />
    </div>
  );
}
