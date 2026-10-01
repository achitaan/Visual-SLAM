"use client";
import { useTelemetry } from './TelemetryProvider';
import { Icon } from './Icon';

export function ControlBar() {
  const { latest, frames, connected, sendControl, showExpected, setShowExpected, restart } = useTelemetry();
  const mode = latest?.mode ?? 'vo';
  const streaming = latest?.stream_enabled ?? true;
  const overlay = latest?.overlay_enabled ?? true;
  const complete = Boolean(latest?.total_frames && latest.frame_index + 1 >= latest.total_frames);
  const status = complete ? 'Run complete' : !connected ? 'Offline' : streaming ? 'Streaming' : 'Stream paused';
  const exportTrajectory = () => {
    const payload = frames.map(({ frame_index, timestamp, pose_T_wc }) => ({ frame_index, timestamp, pose_T_wc }));
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `trajectory-${latest?.sequence ?? 'session'}.json`; link.click(); URL.revokeObjectURL(url);
  };
  return <div className="sessionBar">
    <div className="sessionIdentity"><span className={`statusDot ${connected && streaming ? 'live' : ''}`} /><strong>{status}</strong><span className="divider" /><span>{latest?.sequence ? `KITTI · ${latest.sequence === 'sample' ? 'Sample' : 'Sequence ' + latest.sequence}` : 'No active sequence'}</span></div>
    <div className="toolbar">
      <div className="segmented" aria-label="Pipeline mode">{(['vo', 'slam'] as const).map(value => <button key={value} className={mode === value ? 'selected' : ''} disabled={!connected} onClick={() => sendControl({ type: 'control', action: 'set_mode', mode: value })}>{value.toUpperCase()}</button>)}</div>
      <button className={`button toggle ${overlay ? 'active' : ''}`} disabled={!connected} aria-pressed={overlay} onClick={() => sendControl({ type: 'control', action: 'toggle_overlay', enabled: !overlay })}><Icon name="layers" size={15} />Features</button>
      <button className={`button toggle ${showExpected ? 'active' : ''}`} aria-pressed={showExpected} onClick={() => setShowExpected(!showExpected)}>Ground truth</button>
      <button className="button primary" disabled={!connected} onClick={() => sendControl({ type: 'control', action: streaming ? 'stop' : 'start' })}><Icon name={streaming ? 'pause' : 'play'} size={15} />{streaming ? 'Pause stream' : 'Resume stream'}</button>
      <button className="button iconButton" title="Clear view and reconnect" aria-label="Clear view" onClick={restart}><Icon name="reset" size={16} /></button>
      <button className="button" onClick={exportTrajectory} disabled={!frames.length}><Icon name="export" size={15} />Export</button>
    </div>
  </div>;
}
