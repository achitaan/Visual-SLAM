import { ControlMessage, TelemetryFrame } from "./types";

type TelemetryListener = (frame: TelemetryFrame) => void;

export class TelemetrySocket {
  private ws?: WebSocket;
  private listeners: Set<TelemetryListener> = new Set();
  private url: string;

  constructor(url: string) {
    this.url = url;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    this.ws = new WebSocket(this.url);
    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as TelemetryFrame;
        this.listeners.forEach((listener) => listener(data));
      } catch {
        // Ignore malformed payloads.
      }
    };
  }

  onFrame(listener: TelemetryListener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  sendControl(message: ControlMessage) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    this.ws.send(JSON.stringify(message));
  }

  close() {
    this.ws?.close();
  }
}
