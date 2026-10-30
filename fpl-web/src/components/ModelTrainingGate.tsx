"use client";

import { useEffect, useState, type ReactNode } from "react";
import ModelTrainingScreen from "./ModelTrainingScreen";

type Status = {
  state: "idle" | "checking" | "training" | "predicting" | "ready" | "error";
  message: string;
};

export default function ModelTrainingGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status | null>(null);

  useEffect(() => {
    let mounted = true;

    async function checkStatus() {
      try {
        const response = await fetch("/api/pipeline-status", { cache: "no-store" });
        if (!response.ok) return;
        const data: Status = await response.json();
        if (mounted) setStatus(data);
      } catch {
        // Keep the site usable during a temporary connection problem.
      }
    }

    void checkStatus();
    const timer = window.setInterval(checkStatus, 4000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  // Only interrupt the entire website for training. Daily CP refreshes can
  // run in the background using previously saved predictions.
  if (status?.state === "training") {
    return <ModelTrainingScreen state={status.state} message={status.message} />;
  }
  return <>{children}</>;
}
