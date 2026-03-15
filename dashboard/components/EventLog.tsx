"use client";

import { useMemo } from "react";
import { useTelemetry } from "./TelemetryProvider";

export function EventLog() {
  const { frames } = useTelemetry();

  const events = useMemo(() => {
    const collected = frames.flatMap((frame) => frame.events ?? []);
    return collected.slice(-50).reverse();
  }, [frames]);

  return (
    <div className="panel">
      <h2>Event Log</h2>
      <div className="stack">
        {events.length === 0 ? (
          <div>No events yet.</div>
        ) : (
          events.map((event, index) => (
            <div key={`${event.timestamp}-${index}`} className="metricCard">
              <div>{new Date(event.timestamp * 1000).toLocaleTimeString()}</div>
              <strong>{event.type}</strong>
              <div>{event.message}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
