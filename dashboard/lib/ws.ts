import { ControlMessage, TelemetryFrame } from './types';

type TelemetryListener = (frame: TelemetryFrame) => void;

export class TelemetrySocket {
  private ws?: WebSocket;
  private listeners = new Set<TelemetryListener>();
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
        const frame = JSON.parse(event.data) as TelemetryFrame;
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
