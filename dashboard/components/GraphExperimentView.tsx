"use client";
import data from '@/data/benchmarks.json';

type Metrics = { ate_rmse_m: number; raw_ate_rmse_m: number; translation_percent: number | null; rotation_deg_per_m: number | null };
type Experiment = { sequence: string; frames: number; keyframe_count: number; verified_loop_count: number; candidate_pairs_tested: number; before: Metrics; after: Metrics; ground_truth_used_for_constraints: boolean; trajectory: { raw: number[][]; corrected: number[][]; ground_truth: number[][] } };
const experiments = (data as { graph_experiments?: Experiment[] }).graph_experiments ?? [];
const number = (value: number | null, digits = 2) => value == null ? '—' : value.toFixed(digits);

export function GraphExperimentView() {
  return <>{experiments.map(run => {
    const all = Object.values(run.trajectory).flat(), xs = all.map(p => p[0]), zs = all.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
    const scale = Math.min(660 / Math.max(1, maxX - minX), 230 / Math.max(1, maxZ - minZ));
    const path = (values: number[][]) => values.map(p => `${360 + (p[0] - (minX + maxX) / 2) * scale},${140 - (p[1] - (minZ + maxZ) / 2) * scale}`).join(' ');
    return <section className="panel graphExperiment" key={run.sequence}><div className="panelHeading"><div><span className="eyebrow">LOOP CONSTRAINT EXPERIMENT · KITTI {run.sequence}</span><h2>Pose graph: before & after</h2></div><span className={`pill ${run.verified_loop_count ? 'teal' : 'amber'}`}>{run.verified_loop_count} verified image loops</span></div>
      <div className="graphComparison"><div><div className="chartLegend graphLegend"><span><i />Raw VO</span><span><i className="corrected" />Graph corrected</span><span><i className="truth" />Ground truth</span></div><svg viewBox="0 0 720 285" className="benchmarkPlot" role="img" aria-label={`Raw, graph-corrected and ground-truth trajectories for KITTI ${run.sequence}`}>
        {Array.from({length:6}, (_, i) => <g key={i}><line x1={30 + i * 132} x2={30 + i * 132} y1="25" y2="255" stroke="#e8edf1" /><line x1="30" x2="690" y1={25 + i * 46} y2={25 + i * 46} stroke="#e8edf1" /></g>)}
        <polyline points={path(run.trajectory.ground_truth)} fill="none" stroke="#11a58b" strokeWidth="2" /><polyline points={path(run.trajectory.raw)} fill="none" stroke="#5375dd" strokeWidth="2" /><polyline points={path(run.trajectory.corrected)} fill="none" stroke="#d39a4b" strokeWidth="2" /><text x="360" y="280" textAnchor="middle" fill="#75818f" fontSize="11">Raw X / Z · meters · equal axis scale</text>
      </svg></div><div className="tableScroll"><table className="dataTable"><thead><tr><th>Metric</th><th>Raw VO</th><th>Graph</th></tr></thead><tbody><tr><td>Aligned ATE · m</td><td className="tableNumber">{number(run.before.ate_rmse_m)}</td><td className="tableNumber">{number(run.after.ate_rmse_m)}</td></tr><tr><td>Raw RMSE · m</td><td>{number(run.before.raw_ate_rmse_m)}</td><td>{number(run.after.raw_ate_rmse_m)}</td></tr><tr><td>Translation drift · %</td><td>{number(run.before.translation_percent)}</td><td>{number(run.after.translation_percent)}</td></tr><tr><td>Rotation drift · °/m</td><td>{number(run.before.rotation_deg_per_m, 4)}</td><td>{number(run.after.rotation_deg_per_m, 4)}</td></tr></tbody></table><p className="graphContext">{run.frames.toLocaleString()} frames · {run.keyframe_count} graph vertices · {run.candidate_pairs_tested} candidate pairs checked</p></div></div>
      <div className="reportFootnote">{run.ground_truth_used_for_constraints ? 'Ground-truth-assisted diagnostic.' : 'Loop transforms come from bidirectional stereo PnP verification; ground truth is used only for evaluation.'} Batch graph correction includes intermediate frames. Live estimator and landmark feedback remain unfinished.</div>
    </section>;
  })}</>;
}
