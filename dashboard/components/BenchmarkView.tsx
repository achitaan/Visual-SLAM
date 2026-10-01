"use client";
import { useState } from 'react';
import data from '@/data/benchmarks.json';
import { Icon } from './Icon';
import { GraphExperimentView } from './GraphExperimentView';

type Run = {
  sequence: string; frames: number; total_frames: number; complete: boolean;
  ate_rmse_m: number; raw_ate_rmse_m: number; translation_percent: number | null;
  rotation_deg_per_m: number | null; segment_count: number; lost_pairs: number | null;
  processing_fps: number | null; updated_at: string;
  trajectory: { estimated: number[][]; ground_truth: number[][] };
};
const runs = data.runs as Run[];
const number = (value: number | null, digits = 2) => value == null ? '—' : value.toFixed(digits);

function TrajectoryComparison({ run }: { run: Run }) {
  const points = [...run.trajectory.estimated, ...run.trajectory.ground_truth];
  const xs = points.map(p => p[0]), zs = points.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
  const scale = Math.min(720 / Math.max(maxX - minX, 1), 225 / Math.max(maxZ - minZ, 1));
  const offsetX = 55 + (720 - (maxX - minX) * scale) / 2;
  const offsetY = 25 + (225 - (maxZ - minZ) * scale) / 2;
  const path = (values: number[][]) => values.map(p => `${offsetX + (p[0] - minX) * scale},${250 - (offsetY - 25) - (p[1] - minZ) * scale}`).join(' ');
  return <svg viewBox="0 0 820 285" className="benchmarkPlot" role="img" aria-label={`Raw estimated and ground-truth trajectories for KITTI sequence ${run.sequence}`}>
    {Array.from({ length: 6 }, (_, i) => <g key={i}><line x1={55 + i * 144} x2={55 + i * 144} y1="20" y2="250" stroke="#e8edf1" /><line x1="55" x2="775" y1={20 + i * 46} y2={20 + i * 46} stroke="#e8edf1" /></g>)}
    <polyline points={path(run.trajectory.ground_truth)} fill="none" stroke="#11a58b" strokeWidth="2.6" strokeLinejoin="round" />
    <polyline points={path(run.trajectory.estimated)} fill="none" stroke="#5375dd" strokeWidth="2.3" strokeLinejoin="round" />
    <text x="410" y="278" textAnchor="middle" fill="#75818f" fontSize="11">X / Z · meters · equal axis scale</text>
  </svg>;
}

export function BenchmarkView() {
  const [filter, setFilter] = useState('all');
  const [selected, setSelected] = useState(runs[0]?.sequence);
  const visible = runs.filter(run => filter === 'all' || (filter === 'full' ? run.complete : !run.complete));
  const current = visible.find(run => run.sequence === selected) ?? visible[0];
  const segments = runs.reduce((sum, run) => sum + run.segment_count, 0);
  const meanDrift = segments ? runs.reduce((sum, run) => sum + (run.translation_percent ?? 0) * run.segment_count, 0) / segments : null;
  return <>
    <div className="benchmarkBanner"><div><span className="eyebrow">GROUND TRUTH EVALUATION</span><h2>KITTI odometry</h2><p>Measured stereo results from the sequences dataset. Partial runs are labeled explicitly.</p></div><span className="pill"><Icon name="layers" size={14} />SIFT · StereoSGBM · PnP</span></div>
    <div className="benchmarkStats"><div className="statCard"><div className="statLabel">Sequences measured</div><div className="statValue">{runs.length}</div><div className="statNote">{runs.filter(run => run.complete).length} complete · {runs.filter(run => !run.complete).length} partial</div></div><div className="statCard"><div className="statLabel">Frames evaluated</div><div className="statValue">{runs.reduce((sum, run) => sum + run.frames, 0).toLocaleString()}</div><div className="statNote">Compared with corresponding ground truth</div></div><div className="statCard"><div className="statLabel">Mean translation drift</div><div className="statValue">{number(meanDrift)}<span>%</span></div><div className="statNote">Weighted by evaluated segments</div></div><div className="statCard"><div className="statLabel">Drift segments</div><div className="statValue">{segments}</div><div className="statNote">100–800 m · no scale alignment</div></div></div>
    <GraphExperimentView />
    <section className="panel"><div className="panelHeading"><div><span className="eyebrow">ACCURACY REPORT</span><h2>Sequence results</h2></div><select aria-label="Benchmark coverage filter" value={filter} onChange={event => setFilter(event.target.value)}><option value="all">All runs</option><option value="full">Complete runs</option><option value="partial">Partial runs</option></select></div>
      <div className="tableScroll"><table className="dataTable benchmarkTable"><thead><tr><th>Sequence</th><th>Coverage</th><th>ATE <span>m · SE(3)</span></th><th>Translation <span>%</span></th><th>Rotation <span>°/m</span></th><th>Tracking losses</th></tr></thead><tbody>{visible.map(run => <tr key={run.sequence} className={selected === run.sequence ? 'selectedRow' : ''}><td><button className="sequenceLink" onClick={() => setSelected(run.sequence)}>KITTI {run.sequence}<Icon name="arrow" size={13} /></button></td><td><span className={`pill ${run.complete ? 'teal' : 'amber'}`}>{run.complete ? 'Complete' : 'Partial'}</span><span className="coverageCount">{run.frames.toLocaleString()} / {run.total_frames.toLocaleString()} frames</span></td><td className="tableNumber">{number(run.ate_rmse_m)}</td><td className="tableNumber">{number(run.translation_percent)}</td><td className="tableNumber">{number(run.rotation_deg_per_m, 4)}</td><td>{run.lost_pairs == null ? <span className="muted">Not recorded</span> : <span className={`pill ${run.lost_pairs ? 'amber' : 'teal'}`}>{run.lost_pairs} / {run.frames - 1} pairs</span>}</td></tr>)}</tbody></table></div>
      {!visible.length && <div className="emptyState"><strong>No matching runs</strong><span>Choose a different coverage filter.</span></div>}
      <div className="reportFootnote">ATE uses rigid 3D alignment. Drift uses the original metric trajectory. These rows measure raw stereo odometry; offline graph comparisons are shown above. Live tracking and landmark correction remain unfinished.</div>
    </section>
    {current && <div className="benchmarkDetail"><section className="panel"><div className="panelHeading"><div><span className="eyebrow">SEQUENCE {current.sequence}</span><h2>Trajectory comparison</h2></div><div className="chartLegend"><span><i className="estimated" />Estimate</span><span><i className="truth" />Ground truth</span></div></div><TrajectoryComparison run={current} /><div className="reportFootnote">Raw coordinates in meters. The plot is not aligned or rescaled to ground truth.</div></section><section className="panel runDetails"><div className="panelHeading"><div><span className="eyebrow">RUN DETAILS</span><h2>Evaluation context</h2></div></div><dl><div><dt>Dataset</dt><dd>KITTI · {current.sequence}</dd></div><div><dt>Frames</dt><dd>{current.frames} / {current.total_frames}</dd></div><div><dt>Raw position RMSE</dt><dd>{number(current.raw_ate_rmse_m)} m</dd></div><div><dt>Valid drift segments</dt><dd>{current.segment_count}</dd></div><div><dt>Wall throughput</dt><dd>{number(current.processing_fps)} fps</dd></div><div><dt>Recorded</dt><dd>{new Date(current.updated_at).toLocaleDateString()}</dd></div></dl><p className="muted">Throughput includes tracking, image reads and any downloads during tracking. Cache preparation is recorded separately in the run log.</p></section></div>}
  </>;
}
