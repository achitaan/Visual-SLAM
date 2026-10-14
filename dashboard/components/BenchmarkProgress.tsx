"use client";
import { useTelemetry } from './TelemetryProvider';

export function BenchmarkProgress() {
  const { benchmark, connected } = useTelemetry();
  if (!benchmark) return null;
  const active = benchmark.active, progress = active?.progress;
  return <section className="panel" style={{ padding: 24, marginBottom: 20 }}>
    <div className="panelHeading"><div><span className="eyebrow">FULL KITTI · SHARED SLAM</span><h2>Benchmark progress</h2></div><span className="pill">{benchmark.completed} / {benchmark.total} runs completed</span></div>
    <p>{benchmark.paused ? 'Paused for stereo reliability rework · displayed artifacts are saved results' : benchmark.finished ? 'Batch completed' : !connected ? 'Monitor disconnected · last received checkpoint' : benchmark.running ? `Running sequence ${active?.sequence} · ${active?.sensor === 'stereo' ? 'Stereo' : 'Monocular'}` : 'Benchmark worker is not running'}</p>
    {progress && <><progress value={progress.frames} max={progress.total_frames} style={{ width: '100%', accentColor: '#5375dd' }} /><p>{progress.frames.toLocaleString()} / {progress.total_frames.toLocaleString()} frames at last checkpoint · {progress.state} · {progress.landmarks.toLocaleString()} landmarks</p><p style={{ color: '#69788a', fontSize: 13 }}>Checkpoint received {new Date(progress.updated_at * 1000).toLocaleTimeString()}. Progress is recorded every 50 frames.</p></>}
    {benchmark.active && !benchmark.stream_available && <p style={{ color: '#69788a', fontSize: 13 }}>This run started before live snapshots were enabled. Image, trajectory and map streaming begins with the next run.</p>}
    <details><summary>Current batch results and queue</summary><div style={{ overflowX: 'auto', marginTop: 12 }}><table style={{ width: '100%', textAlign: 'left', fontSize: 13 }}><thead><tr>{['Sequence', 'Sensor', 'Status', 'Frames', 'Lost', 'ATE (m)', 'Alignment', 'Drift (%)'].map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{benchmark.rows.map(row => <tr key={`${row.sequence}-${row.sensor}`}><td>{row.sequence}</td><td>{row.sensor}</td><td>{row.status.replaceAll('_', ' ')}</td><td>{row.frames ?? '—'}</td><td>{row.lost_frames ?? '—'}</td><td>{row.ate_rmse_m?.toFixed(3) ?? '—'}</td><td>{row.alignment ?? '—'}</td><td>{row.translation_percent?.toFixed(3) ?? '—'}</td></tr>)}</tbody></table></div><p style={{ fontSize: 12 }}>Monocular accuracy uses fitted scale (Sim(3)); its map remains in arbitrary units. Meter-based drift is unavailable for unscaled monocular output.</p></details>
  </section>;
}
