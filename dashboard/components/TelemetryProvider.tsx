"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { ControlMessage, TelemetryFrame } from "@/lib/types";
import { TelemetrySocket } from "@/lib/ws";

type TelemetryContextValue = {
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
  const [frames, setFrames] = useState<TelemetryFrame[]>([]);
  const [connected, setConnected] = useState(false);
  const [showExpected, setShowExpected] = useState(true);
  const [reconnectKey, setReconnectKey] = useState(0);
  const socketRef = useRef<TelemetrySocket | null>(null);

  useEffect(() => {
    const url = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8765";
    const socket = new TelemetrySocket(url);
    socketRef.current = socket;

    socket.connect();
    setConnected(true);

    const unsub = socket.onFrame((frame) => {
      setLatest(frame);
      setFrames((prev) => {
        const next = prev.length > 2000 ? prev.slice(-2000) : prev;
        return [...next, frame];
      });
    });

    return () => {
      unsub();
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
    socketRef.current?.close();
    setReconnectKey((k) => k + 1);
  }, []);

  const value = useMemo(
    () => ({ latest, frames, connected, showExpected, setShowExpected, sendControl, restart }),
    [latest, frames, connected, showExpected, restart]
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
