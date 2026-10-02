"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { BenchmarkStatus, ControlMessage, TelemetryFrame } from "@/lib/types";
import { TelemetrySocket } from "@/lib/ws";

type TelemetryContextValue = {
  benchmark?: BenchmarkStatus;
  latest?: TelemetryFrame;
  frames: TelemetryFrame[];
  connected: boolean;
  showExpected: boolean;
  setShowExpected: (value: boolean) => void;
  sendControl: (message: ControlMessage) => void;
  restart: () => void;
};

const TelemetryContext = createContext<TelemetryContextValue | undefined>(undefined);

export function TelemetryProvider({ children }: { children: React.ReactNode }) {
  const [latest, setLatest] = useState<TelemetryFrame>();
  const [benchmark, setBenchmark] = useState<BenchmarkStatus>();
  const [frames, setFrames] = useState<TelemetryFrame[]>([]);
  const [connected, setConnected] = useState(false);
  const [showExpected, setShowExpected] = useState(true);
  const [reconnectKey, setReconnectKey] = useState(0);
  const socketRef = useRef<TelemetrySocket | null>(null);

  useEffect(() => {
    const url = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8765";
    const socket = new TelemetrySocket(url);
    socketRef.current = socket;

    const unsubStatus = socket.onStatus(setConnected);
    const unsubBenchmark = socket.onBenchmark(status => {
      setBenchmark(status);
      if (status.active) {
        setLatest(prev => prev?.run_id === status.active?.run_id ? prev : undefined);
        setFrames(prev => prev.at(-1)?.run_id === status.active?.run_id ? prev : []);
      }
    });

    const unsub = socket.onFrame((frame) => {
      setLatest(frame);
      setFrames((prev) => {
        if (frame.run_id && prev.at(-1)?.run_id !== frame.run_id) prev = [];
        // Control acknowledgements update latest without duplicating poses.
        if (prev.at(-1)?.frame_index === frame.frame_index) return prev;
        // Keep heavy image/map payloads only in latest, not trajectory history.
        const compact = { ...frame, image: null, features: null, map_points: null, trajectory: undefined,
          pose_graph: frame.pose_graph ? { ...frame.pose_graph, optimized_poses: undefined } : null };
        return [...prev.slice(-1999), compact];
      });
    });
    socket.connect();

    return () => {
      unsub();
      unsubStatus();
      unsubBenchmark();
      socket.close();
      setConnected(false);
    };
  }, [reconnectKey]);

  const sendControl = (message: ControlMessage) => {
    socketRef.current?.sendControl(message);
  };

  const restart = useCallback(() => {
    setFrames([]);
    setLatest(undefined);
    setBenchmark(undefined);
    socketRef.current?.close();
    setReconnectKey((k) => k + 1);
  }, []);

  const value = useMemo(
    () => ({ latest, benchmark, frames, connected, showExpected, setShowExpected, sendControl, restart }),
    [latest, benchmark, frames, connected, showExpected, restart]
  );

  return <TelemetryContext.Provider value={value}>{children}</TelemetryContext.Provider>;
}

export function useTelemetry() {
  const ctx = useContext(TelemetryContext);
  if (!ctx) {
    throw new Error("useTelemetry must be used within TelemetryProvider");
  }
  return ctx;
}
