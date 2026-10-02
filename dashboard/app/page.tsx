"use client";

import { useState } from 'react';

import { BenchmarkView } from '@/components/BenchmarkView';

import { SavedVisuals } from '@/components/SavedVisuals';
import { BenchmarkProgress } from '@/components/BenchmarkProgress';

import { ControlBar } from '@/components/ControlBar';

import { ErrorGraph } from '@/components/ErrorGraph';

import { EventLog } from '@/components/EventLog';

import { FrameViewer } from '@/components/FrameViewer';

import { Icon } from '@/components/Icon';

import { MapSummary } from '@/components/MapSummary';

import { MapView3D } from '@/components/MapView3D';

import { MetricsPanel } from '@/components/MetricsPanel';

import { TelemetryProvider, useTelemetry } from '@/components/TelemetryProvider';

import { Trajectory2D } from '@/components/Trajectory2D';

import { ReconstructionView } from '@/components/ReconstructionView';

type View = 'live' | 'benchmarks' | 'events' | 'reconstruction';

const views = [{ id: 'live', icon: 'live', label: 'Live' }, { id: 'benchmarks', icon: 'runs', label: 'Benchmarks' }, { id: 'reconstruction', icon: 'layers', label: 'Reconstruction' }, { id: 'events', icon: 'events', label: 'Events' }] as const;

function Workspace() {

  const [view, setView] = useState<View>('live');

  const { latest, connected, benchmark } = useTelemetry();

  const titles = { live: ['Live workspace', 'A clear view of camera tracking and trajectory quality.'], benchmarks: ['Benchmark results', 'Accuracy measured against KITTI ground-truth trajectories.'], events: ['Session activity', 'Tracking changes, keyframes and optimization events.'], reconstruction: ['Scene reconstruction', 'Persistent landmarks and after-run dense point clouds.'] };

  return <div className="appShell">

    <aside className="sidebar"><a className="brand" href="/" aria-label="Visual SLAM home">V<span>S</span></a><div className="navItems">{views.map(item => <button key={item.id} className={`navItem ${view === item.id ? 'active' : ''}`} aria-label={item.label} aria-current={view === item.id ? 'page' : undefined} onClick={() => setView(item.id)}><Icon name={item.icon} size={21} /><span>{item.label}</span></button>)}</div><div className="sidebarFooter"><span className={`statusDot ${connected ? 'live' : ''}`} /><span>{connected ? 'Online' : 'Offline'}</span></div></aside>

    <main className="workspace"><header className="workspaceHeader"><div><div className="breadcrumb">VISUAL SLAM<span>/</span>EXPERIMENT CONSOLE</div><h1>{titles[view][0]}</h1><p>{titles[view][1]}</p></div><div className="headerContext"><span className="pill">{latest?.translation_scale === 'metric' ? 'Stereo · Metric' : latest ? 'Monocular · Arbitrary scale' : benchmark?.active ? `KITTI ${benchmark.active.sequence} · ${benchmark.active.sensor === 'stereo' ? 'Stereo' : 'Monocular'}` : 'No active session'}</span><span className={`connectionBadge ${connected ? 'connected' : ''}`}><span className="statusDot" />{connected ? 'Backend connected' : 'Backend offline'}</span></div></header>

      {view === 'live' ? <><BenchmarkProgress /><SavedVisuals />{(!benchmark || latest) && <ControlBar />}<MetricsPanel /><div className="visualGrid"><FrameViewer /><Trajectory2D /></div><div className="analysisGrid"><MapView3D /><ErrorGraph /></div><MapSummary /><footer className="workspaceFooter"><span>Camera-to-world pose convention · coordinates in reported scale</span><span>Pause controls streaming. The pipeline continues processing.</span></footer></> : view === 'benchmarks' ? <><BenchmarkProgress /><SavedVisuals /><BenchmarkView /></> : view === 'reconstruction' ? <ReconstructionView /> : <EventLog />}

    </main>

  </div>;

}

export default function Home() { return <TelemetryProvider><Workspace /></TelemetryProvider>; }
