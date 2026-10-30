"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ModelTrainingScreen from "@/components/ModelTrainingScreen";

type Status = {
  state: "idle" | "checking" | "training" | "predicting" | "ready" | "error";
  message: string;
  hasPredictions: boolean;
};

export default function ModelLoadingPage() {
  const router = useRouter();
  const [status, setStatus] = useState<Status>({
    state: "checking",
    message: "Connecting to the prediction pipeline...",
    hasPredictions: false,
  });

  useEffect(() => {
    let mounted = true;
    const params = new URLSearchParams(window.location.search);
    const managerId = params.get("managerId") || "";
    const returnPath = /^[1-9][0-9]{0,9}$/.test(managerId)
      ? `/my-team?managerId=${managerId}&auto=1`
      : "/my-team";

    async function check() {
      try {
        const response = await fetch("/api/pipeline-status", { cache: "no-store" });
        if (!response.ok) throw new Error("Could not check the pipeline status");
        const data: Status = await response.json();
        if (!mounted) return;
        if (data.state === "idle") {
          setStatus({ state: "error", message: "No pipeline job is running. Return to My Team and try again.", hasPredictions: false });
          return;
        }
        setStatus(data);
        if (data.state === "ready" && data.hasPredictions) {
          router.replace(returnPath);
        }
      } catch {
        if (mounted) {
          setStatus({
            state: "error",
            message: "Cannot reach the backend. Check that Next.js and Python are running.",
            hasPredictions: false,
          });
        }
      }
    }

    void check();
    const timer = window.setInterval(check, 3000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [router]);

  return <ModelTrainingScreen state={status.state} message={status.message} />;
}
