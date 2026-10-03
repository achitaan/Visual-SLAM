import { BenchmarkStatus, ControlMessage, TelemetryFrame } from './types';

type TelemetryListener = (frame: TelemetryFrame) => void;

export class TelemetrySocket {
  private ws?: WebSocket;
  private listeners = new Set<TelemetryListener>();
  private benchmarkListeners = new Set<(status: BenchmarkStatus) => void>();
  private statusListeners = new Set<(connected: boolean) => void>();
  private retry?: ReturnType<typeof setTimeout>;
  private disposed = false;

  constructor(private url: string) {}

  connect() {
    if (this.disposed || (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING))) return;
    const socket = new WebSocket(this.url);
    this.ws = socket;
    socket.onopen = () => this.setConnected(true);
    socket.onerror = () => this.setConnected(false);
    socket.onclose = () => {
      this.setConnected(false);
      if (!this.disposed) this.retry = setTimeout(() => this.connect(), 1000);
    };
    socket.onmessage = (event) => {
      try {
        const frame = JSON.parse(event.data);
        if (frame?.kind === 'benchmark_status') {
          if (frame.schema_version !== 1 || !Number.isInteger(frame.completed) || !Number.isInteger(frame.total) || frame.completed < 0 || frame.total < frame.completed || !Array.isArray(frame.rows) || typeof frame.running !== 'boolean' || typeof frame.stream_available !== 'boolean') return;
          if (frame.active != null && (typeof frame.active.run_id !== 'string' || typeof frame.active.sequence !== 'string' || typeof frame.active.sensor !== 'string')) return;
          this.benchmarkListeners.forEach(listener => listener(frame));
          return;
        }
        if (frame?.schema_version !== 1 || !Number.isInteger(frame.frame_index) || !Array.isArray(frame.pose_T_wc)) return;
        this.listeners.forEach((listener) => listener(frame));
      } catch { /* Ignore malformed payloads. */ }
    };
  }

  private setConnected(value: boolean) {
    this.statusListeners.forEach((listener) => listener(value));
  }

  onStatus(listener: (connected: boolean) => void) {
    this.statusListeners.add(listener);
    return () => { this.statusListeners.delete(listener); };
  }

  onFrame(listener: TelemetryListener) {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  onBenchmark(listener: (status: BenchmarkStatus) => void) {
    this.benchmarkListeners.add(listener);
    return () => { this.benchmarkListeners.delete(listener); };
  }

  sendControl(message: ControlMessage) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(message));
  }

  close() {
    this.disposed = true;
    clearTimeout(this.retry);
    this.ws?.close();
    this.setConnected(false);
  }
}
