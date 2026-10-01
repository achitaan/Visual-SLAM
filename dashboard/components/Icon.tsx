import type { CSSProperties } from "react";

const paths = {
  live: "M3 12h4l3-8 4 16 3-8h4",
  runs: "M5 20V10m7 10V4m7 16v-7",
  events: "M5 6h14M5 12h14M5 18h9",
  camera: "M4 7h3l2-3h6l2 3h3v13H4V7Zm8 3a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Z",
  arrow: "M7 17 17 7M7 7h10v10",
  export: "M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5",
  pause: "M9 5v14M15 5v14",
  play: "m8 5 11 7-11 7V5Z",
  reset: "M4 9a8 8 0 1 1 0 6M4 3v6h6",
  layers: "m12 3 10 6-10 6L2 9l10-6Zm-10 12 10 6 10-6M2 12l10 6 10-6",
};

export function Icon({ name, size = 18, style }: { name: keyof typeof paths; size?: number; style?: CSSProperties }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={style}><path d={paths[name]} /></svg>;
}
