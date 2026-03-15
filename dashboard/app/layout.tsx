import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Visual SLAM Dashboard",
  description: "Real-time VO/SLAM telemetry dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
