"use client";

import { useEffect, useRef } from "react";
import { useTelemetry } from "./TelemetryProvider";

export function FrameViewer() {
  const { latest } = useTelemetry();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!latest?.image || !canvasRef.current) {
      return;
    }
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      if (latest.features && latest.features.length > 0) {
        for (const feature of latest.features) {
          ctx.fillStyle = feature.inlier ? "#00d084" : "#ff6b6b";
          ctx.beginPath();
          ctx.arc(feature.x, feature.y, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    };
    img.src = `data:image/${latest.image.encoding};base64,${latest.image.data_base64}`;
  }, [latest]);

  return (
    <div className="panel">
      <h2>Camera Frame</h2>
      <canvas ref={canvasRef} style={{ width: "100%", height: "auto" }} />
    </div>
  );
}
