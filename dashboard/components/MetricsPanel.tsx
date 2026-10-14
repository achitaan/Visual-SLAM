"use client";
import { useTelemetry } from './TelemetryProvider';

export function MetricsPanel() {
  const { latest, connected } = useTelemetry();
  const tracking = latest?.tracking;
  const ratio = tracking?.inlier_ratio;
  const positionError = latest?.translation_scale === 'metric' && latest.expected_pose_T_wc
    ? Math.hypot(...[0, 1, 2].map(i => latest.pose_T_wc[i][3] - latest.expected_pose_T_wc![i][3])) : undefined;
  const quality = !latest ? 'Awaiting data' : tracking?.state ? ({ initializing: 'Initializing', tracking: 'Tracking stable', lost: 'Tracking lost', relocalized: 'Relocalized' }[tracking.state]) : tracking?.tracking_ok === false ? 'Tracking lost' : 'Tracking stable';
  const cards = [
    { label: 'Processing rate', value: latest?.fps?.toFixed(1) ?? '—', unit: 'fps', note: connected ? 'Including frame pacing' : 'Last received frame' },
    { label: 'Feature matches', value: tracking?.num_matches?.toLocaleString() ?? '—', unit: '', note: latest?.mode_locked ? 'Image-to-map correspondences' : 'Consecutive camera frames' },
    { label: 'Geometric inliers', value: tracking?.num_inliers?.toLocaleString() ?? '—', unit: '', note: 'Accepted pose correspondences' },
    { label: 'Inlier ratio', value: ratio == null ? '—' : (ratio * 100).toFixed(1), unit: '%', note: quality, tone: tracking?.tracking_ok === false ? 'warning' : 'positive' },
    { label: 'Keyframes', value: latest?.map.keyframes.toLocaleString() ?? '—', unit: '', note: 'Retained reference views' },
    { label: 'Position difference', value: positionError?.toFixed(2) ?? '—', unit: 'm', note: latest?.translation_scale === 'arbitrary' ? 'Metric scale unavailable' : 'Raw distance to ground truth' },
  ];
  return <div className="metricStrip">{cards.map(card => <div key={card.label} className={`statCard ${card.tone ?? ''}`}><div className="statLabel">{card.label}</div><div className="statValue">{card.value}<span>{card.unit}</span></div><div className="statNote">{card.note}</div></div>)}</div>;
}
