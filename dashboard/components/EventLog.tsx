"use client";
import { useMemo } from 'react';
import { useTelemetry } from './TelemetryProvider';
import { Icon } from './Icon';

export function EventLog() {
  const { frames } = useTelemetry();
  const events = useMemo(() => frames.flatMap(frame => frame.events ?? []).slice(-100).reverse(), [frames]);
  return <section className="panel eventPanel"><div className="panelHeading"><div><span className="eyebrow">SESSION ACTIVITY</span><h2>Event log</h2></div><span className="pill">{events.length} events</span></div>
    {!events.length ? <div className="emptyState"><Icon name="events" size={30} /><strong>No events recorded</strong><span>Keyframes, tracking changes and graph updates appear here.</span></div> : <div className="tableScroll"><table className="dataTable"><thead><tr><th>Time</th><th>Event</th><th>Detail</th><th>Severity</th></tr></thead><tbody>{events.map((event, index) => <tr key={`${event.timestamp}-${index}`}><td className="mono muted">{new Date(event.timestamp * 1000).toLocaleTimeString()}</td><td><span className="eventType">{event.type.replaceAll('_', ' ')}</span></td><td>{event.message}</td><td><span className={`pill ${event.severity === 'error' ? 'danger' : event.severity === 'warn' ? 'amber' : ''}`}>{event.severity ?? 'info'}</span></td></tr>)}</tbody></table></div>}
  </section>;
}
