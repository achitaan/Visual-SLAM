"use client";
import { useTelemetry } from './TelemetryProvider';

export function MapSummary() {
  const { latest } = useTelemetry();
  return <div className="mapSummary"><span>Map state</span><strong>{latest?.map.keyframes.toLocaleString() ?? '—'}<span>keyframes</span></strong><strong>{latest?.map.map_points.toLocaleString() ?? '—'}<span>landmarks</span></strong><span className="mapNote">{latest?.translation_scale === 'metric' && latest?.map.map_points === 0 ? 'Stereo landmarks are not yet mapped.' : latest?.translation_scale === 'arbitrary' ? 'Map scale is arbitrary.' : 'Camera-to-world coordinates'}</span></div>;
}
