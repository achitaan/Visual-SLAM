"use client";
import { useState } from 'react';
import plots from '@/data/saved-visuals.json';

export function SavedVisuals() {
  const [selected, setSelected] = useState(plots[0]?.file);
  const plot = plots.find(item => item.file === selected) ?? plots[0];
  if (!plot) return null;
  return <section className="panel" style={{ marginBottom: 20 }}>
    <div className="panelHeading"><div><span className="eyebrow">SAVED RUNS · ACTUAL RESULTS</span><h2>Trajectories, errors & maps</h2></div></div>
    <div style={{ padding: '0 20px 16px' }}><label>Saved visualization <select aria-label="Saved visualization" value={plot.file} onChange={event => setSelected(event.target.value)} style={{ marginLeft: 12, maxWidth: '100%' }}>{plots.map(item => <option key={item.file} value={item.file}>{item.label}</option>)}</select></label><p style={{ color: '#69788a' }}>{plot.group}. These are retained results, separate from the active benchmark.</p><a href={`/saved-visuals/${plot.file}`} target="_blank" rel="noreferrer"><img src={`/saved-visuals/${plot.file}`} alt={plot.label} style={{ width: '100%', height: 'auto', display: 'block' }} /></a></div>
  </section>;
}
