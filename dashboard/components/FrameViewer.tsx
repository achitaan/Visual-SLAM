"use client";
import { useEffect, useRef, useState } from 'react';
import { TelemetryFrame } from '@/lib/types';
import { Icon } from './Icon';
import { useTelemetry } from './TelemetryProvider';

export function FrameViewer() {
  const { latest } = useTelemetry();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [cameraFrame, setCameraFrame] = useState<TelemetryFrame>();
  useEffect(() => {
    if (!latest || (cameraFrame?.run_id && latest.run_id !== cameraFrame.run_id)) setCameraFrame(undefined);
    if (latest?.image) setCameraFrame(latest);
  }, [latest, cameraFrame?.run_id]);
  const image = cameraFrame?.image;
  useEffect(() => {
    if (!image || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      canvas.width = img.width; canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      if (latest?.overlay_enabled !== false) for (const feature of cameraFrame?.features ?? []) {
        ctx.fillStyle = feature.inlier ? '#37d6ae' : '#f97475';
        ctx.beginPath(); ctx.arc(feature.x, feature.y, 1.8, 0, Math.PI * 2); ctx.fill();
      }
    };
    img.src = `data:image/${image.encoding};base64,${image.data_base64}`;
    return () => { cancelled = true; };
  }, [image, cameraFrame, latest?.overlay_enabled]);
  const progress = latest?.total_frames ? (latest.frame_index + 1) / latest.total_frames * 100 : 0;
  return <section className="panel cameraPanel">
    <div className="panelHeading"><div><span className="eyebrow">VISUAL INPUT</span><h2>Camera feed</h2></div><span className="pill">{image ? `${image.width} × ${image.height}` : 'No feed'}</span></div>
    <div className={`cameraStage ${image ? 'hasImage' : ''}`}>
      <canvas ref={canvasRef} hidden={!image} role="img" aria-label="Latest received camera frame with optional feature overlay" />
      {!image && <div className="emptyState"><Icon name="camera" size={32} /><strong>Waiting for camera frames</strong><span>Connect an odometry run to see the camera feed.</span></div>}
      {image && <span className="cameraTag">{cameraFrame?.translation_scale === 'metric' ? 'STEREO · RECTIFIED LEFT' : 'MONOCULAR'}{latest?.stream_enabled === false ? ' · STREAM PAUSED' : ''}</span>}
    </div>
    <div className="frameFooter"><span><span className="statusDot" />{cameraFrame ? `Camera frame ${(cameraFrame.frame_index + 1).toLocaleString()}` : 'Frame —'}{latest?.total_frames ? ` / ${latest.total_frames.toLocaleString()}` : ''}</span><span>{latest?.overlay_enabled === false ? 0 : cameraFrame?.features?.length ?? 0} displayed features</span></div>
    <div className="progressTrack"><div style={{ width: `${progress}%` }} /></div>
  </section>;
}
