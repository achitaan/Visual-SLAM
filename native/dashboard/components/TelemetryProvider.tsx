"use client";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { BenchmarkStatus, ControlMessage, TelemetryFrame } from '@/lib/types';
import { TelemetrySocket } from '@/lib/ws';
import { NativeTelemetryGate, type NativeRunBinding } from '@/lib/native-binding';
type Phase = 'unbound' | 'awaiting' | 'verified' | 'offline' | 'rejected';
type ContextValue = {
  benchmark?: BenchmarkStatus; latest?: TelemetryFrame; frames: TelemetryFrame[];
  connected: boolean; phase: Phase; connectionNote: string; showExpected: boolean;
  setShowExpected: (v: boolean) => void; sendControl: (m: ControlMessage) => void; restart: () => void;
};
const TelemetryContext = createContext<ContextValue | undefined>(undefined);
export function TelemetryProvider({ children, binding }: { children: React.ReactNode; binding?: NativeRunBinding }) {
  const [latest, setLatest] = useState<TelemetryFrame>(), [frames, setFrames] = useState<TelemetryFrame[]>([]);
  const [connected, setConnected] = useState(false), [phase, setPhase] = useState<Phase>('unbound');
  const [connectionNote, setConnectionNote] = useState(''), [showExpected, setShowExpected] = useState(true);
  const [reconnectKey, setReconnectKey] = useState(0), socketRef = useRef<TelemetrySocket | null>(null), verified = useRef(false);
  const url = binding?.url, runId = binding?.runId, sequence = binding?.sequence, count = binding?.frames;
  useEffect(() => {
    setLatest(undefined); setFrames([]); setConnected(false); verified.current = false; setConnectionNote('');
    if (!url || !runId || !sequence || !count) { setPhase('unbound'); return; }
    const gate = new NativeTelemetryGate({ url, runId, sequence, frames: count });
    let rejected = false;
    const socket = new TelemetrySocket(url); socketRef.current = socket; setPhase('awaiting');
    const unsubStatus = socket.onStatus(open => {
      if (!open) { verified.current = false; setConnected(false); setPhase(p => p === 'rejected' ? p : 'offline'); }
      else setPhase(p => p === 'rejected' ? p : 'awaiting');
    });
    const unsubBenchmark = socket.onBenchmark(() => {
      rejected = true;
      verified.current = false; setConnected(false); setPhase('rejected');
      setLatest(undefined); setFrames([]);
      setConnectionNote('This endpoint is not the requested native run.'); socket.close();
    });
    const unsub = socket.onFrame(frame => {
      if (rejected) return;
      if (!gate.accept(frame)) {
        rejected = true;
        verified.current = false; setConnected(false); setPhase('rejected'); setConnectionNote(gate.reason);
        setLatest(undefined); setFrames([]); socket.close(); return;
      }
      verified.current = true; setConnected(true); setPhase('verified'); setLatest(frame);
      setFrames(prev => {
        if (prev.at(-1)?.frame_index === frame.frame_index) return prev;
        const compact = { ...frame, image: null, features: null, map_points: null, trajectory: undefined, pose_graph: null };
        return [...prev, compact];
      });
    });
    socket.connect();
    return () => {
      unsub(); unsubStatus(); unsubBenchmark(); verified.current = false; socket.close();
      if (socketRef.current === socket) socketRef.current = null;
      setConnected(false);
    };
  }, [url, runId, sequence, count, reconnectKey]);
  const sendControl = useCallback((message: ControlMessage) => { if (verified.current) socketRef.current?.sendControl(message); }, []);
  const restart = useCallback(() => { verified.current = false; socketRef.current?.close(); setReconnectKey(k => k + 1); }, []);
  const value = useMemo(() => ({ latest, frames, connected, phase, connectionNote, showExpected, setShowExpected, sendControl, restart }),
    [latest, frames, connected, phase, connectionNote, showExpected, sendControl, restart]);
  return <TelemetryContext.Provider value={value}>{children}</TelemetryContext.Provider>;
}
export function useTelemetry() {
  const context = useContext(TelemetryContext);
  if (!context) throw new Error('useTelemetry must be used within TelemetryProvider');
  return context;
}
