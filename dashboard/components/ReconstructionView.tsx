"use client";
import { useEffect, useRef, useState } from 'react';
import saved from '@/data/saved-reconstructions.json';

type Preview = { revision: number; translation_scale: string; depth_source: string; depth_model?: string; training_domain?: string; sparse: number[][]; dense: number[][]; dense_colors?: number[][]; trajectory: number[][] };
export function ReconstructionView() {
  const [data, setData] = useState<Preview | null>(null);
  const [error, setError] = useState('');
  const [sparse, setSparse] = useState(true), [dense, setDense] = useState(true);
  const [view, setView] = useState({ yaw: 0.3, pitch: -0.25, zoom: 1 });
  const canvas = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  async function loadSaved(url: string) {
    if (!url) return;
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error('Saved reconstruction is unavailable.');
      await load(new File([await response.blob()], 'preview.json', { type: 'application/json' }));
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to open saved reconstruction.'); }
  }
  async function load(file?: File) {
    if (!file) return;
    try {
      if (file.size > 12 * 1024 * 1024) throw new Error('Preview exceeds the 12 MB limit.');
      const value = JSON.parse(await file.text()) as Preview;
      for (const key of ['sparse', 'dense', 'trajectory'] as const) {
        if (!Array.isArray(value[key]) || value[key].length > 50000 || value[key].some(p => !Array.isArray(p) || p.length !== 3 || p.some(v => !Number.isFinite(v)))) throw new Error('Invalid point-cloud preview.');
      }
      if (value.dense_colors && (value.dense_colors.length !== value.dense.length || value.dense_colors.some(c => !Array.isArray(c) || c.length !== 3 || c.some(v => !Number.isFinite(v) || v < 0 || v > 255)))) throw new Error('Invalid point colors.');
      setData(value); setError(''); setView({ yaw: 0.3, pitch: -0.25, zoom: 1 });
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to read preview.'); }
  }
  function download(points: number[][], name: string) {
    const header = `ply\nformat ascii 1.0\nelement vertex ${points.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n`;
    const url = URL.createObjectURL(new Blob([header, points.map(p => p.join(' ')).join('\n')], { type: 'text/plain' }));
    const link = document.createElement('a'); link.href = url; link.download = name; link.click(); URL.revokeObjectURL(url);
  }
  useEffect(() => {
    const el = canvas.current, context = el?.getContext('2d');
    if (!el || !context) return;
    const draw = () => {
      const width = el.clientWidth, height = 520, ratio = window.devicePixelRatio || 1;
      el.width = width * ratio; el.height = height * ratio; context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.fillStyle = '#fafbfd'; context.fillRect(0, 0, width, height);
      if (!data) return;
      const rotate = (p: number[]) => { const x = Math.cos(view.yaw)*p[0]+Math.sin(view.yaw)*p[2], z = -Math.sin(view.yaw)*p[0]+Math.cos(view.yaw)*p[2]; return [x, Math.cos(view.pitch)*p[1]-Math.sin(view.pitch)*z]; };
      const all = [...data.sparse, ...data.dense, ...data.trajectory].map(rotate);
      if (!all.length) return;
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of all) { minX = Math.min(minX,p[0]); maxX = Math.max(maxX,p[0]); minY = Math.min(minY,p[1]); maxY = Math.max(maxY,p[1]); }
      const scale = Math.min((width-50)/Math.max(maxX-minX,0.001),470/Math.max(maxY-minY,0.001))*view.zoom;
      const project = (p:number[]) => { const q=rotate(p); return [width/2+(q[0]-(minX+maxX)/2)*scale,260+(q[1]-(minY+maxY)/2)*scale]; };
      if (dense) data.dense.forEach((p,i) => { const q=project(p), c=data.dense_colors?.[i]; context.fillStyle=c ? `rgb(${c[0]},${c[1]},${c[2]})` : '#8ba9b5'; context.fillRect(q[0],q[1],1.5,1.5); });
      if (sparse) { context.fillStyle='#cd852b'; data.sparse.forEach(p => { const q=project(p); context.fillRect(q[0],q[1],3,3); }); }
      context.strokeStyle='#4775d1'; context.lineWidth=2; context.beginPath(); data.trajectory.forEach((p,i)=>{ const q=project(p); if(i) context.lineTo(q[0],q[1]); else context.moveTo(q[0],q[1]); }); context.stroke();
    };
    draw(); const observer=new ResizeObserver(draw); observer.observe(el); return () => observer.disconnect();
  }, [data, sparse, dense, view]);
  return <section className="panel"><div className="panelHeader"><div><h2>Reconstruction</h2><p>Load a saved preview.json. Drag to orbit; scroll to zoom.</p></div><label>Open reconstruction <input type="file" accept=".json" onChange={e=>void load(e.target.files?.[0])} /></label></div>
    <div style={{ padding: '12px 20px' }}><label>Saved reconstruction <select aria-label="Saved reconstruction" defaultValue="" onChange={event => void loadSaved(event.target.value)} style={{ marginLeft: 12 }}><option value="">Choose a saved run</option>{saved.map(item => <option key={item.url} value={item.url}>{item.label}</option>)}</select></label><p>Retained outputs from earlier runs; separate from live tracking.</p></div>
    {error && <p role="alert">{error}</p>}
    {data && <div style={{padding:'12px 20px',display:'flex',gap:12}}><button onClick={()=>download(data.sparse,'sparse-preview.ply')}>Download sparse preview</button><button disabled={!data.dense.length} onClick={()=>download(data.dense,'dense-preview.ply')}>Download dense preview</button><span>Full-resolution PLY files remain in the saved run directory.</span></div>}
    <div style={{display:'flex',gap:20,padding:'12px 20px'}}><label><input type="checkbox" checked={sparse} onChange={e=>setSparse(e.target.checked)} /> Sparse landmarks</label><label><input type="checkbox" checked={dense} onChange={e=>setDense(e.target.checked)} /> Dense reconstruction</label><button onClick={()=>setView({yaw:.3,pitch:-.25,zoom:1})}>Reset view</button></div>
    <canvas ref={canvas} style={{width:'100%',height:520,touchAction:'none'}} aria-label="Sparse and dense reconstruction with camera trajectory" onPointerDown={e=>{drag.current={x:e.clientX,y:e.clientY};e.currentTarget.setPointerCapture(e.pointerId);}} onPointerUp={()=>{drag.current=null;}} onPointerMove={e=>{if(drag.current){const dx=e.clientX-drag.current.x,dy=e.clientY-drag.current.y;setView(v=>({...v,yaw:v.yaw+dx*.006,pitch:Math.max(-1.5,Math.min(1.5,v.pitch+dy*.006))}));drag.current={x:e.clientX,y:e.clientY};}}} onWheel={e=>setView(v=>({...v,zoom:Math.max(.2,Math.min(8,v.zoom*Math.exp(-e.deltaY*.001)))}))} />
    <p style={{padding:20}}>{data ? `${data.sparse.length.toLocaleString()} sparse · ${data.dense.length.toLocaleString()} dense preview points · ${data.translation_scale} scale · ${data.depth_source} · map revision ${data.revision}${data.depth_model ? ` · ${data.depth_model} · trained on ${data.training_domain}` : ''}` : 'Run SLAM to export a sparse preview, or offline reconstruction to include dense points.'}</p>
  </section>;
}
