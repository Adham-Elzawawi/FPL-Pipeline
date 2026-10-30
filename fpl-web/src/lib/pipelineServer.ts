import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const projectDir = path.resolve(process.cwd(), "..");
const outputDir = path.join(projectDir, "outputs");
const statusFile = path.join(outputDir, "pipeline_status.json");
const predictionFile = path.join(outputDir, "current_prediction.json");

type PipelineState = "idle" | "checking" | "training" | "predicting" | "ready" | "error";

type PipelineStatus = {
  state: PipelineState;
  message: string;
  updatedAt?: string;
  pid?: number | null;
  hasPredictions: boolean;
};

function readStatus(): Partial<PipelineStatus> | null {
  try {
    return JSON.parse(fs.readFileSync(statusFile, "utf8"));
  } catch {
    return null;
  }
}

function processExists(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    // EPERM means the process is running but we can't signal it.
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

export function getPipelineStatus(): PipelineStatus {
  const saved = readStatus();
  const hasPredictions = fs.existsSync(predictionFile);

  if (!saved || !saved.state) {
    return {
      state: hasPredictions ? "ready" : "idle",
      message: hasPredictions ? "Predictions are available" : "Models have not been prepared yet",
      hasPredictions,
    };
  }

  const active = ["checking", "training", "predicting"].includes(saved.state);
  if (active) {
    const updated = Date.parse(saved.updatedAt || "");
    const justStarted = Number.isFinite(updated) && Date.now() - updated < 90000;
    const running = saved.pid ? processExists(saved.pid) : justStarted;
    if (!running) {
      return {
        state: "error",
        message: "The pipeline stopped unexpectedly. Check outputs/pipeline.log.",
        hasPredictions,
      };
    }
  }

  if (saved.state === "ready" && !hasPredictions) {
    return {
      state: "error",
      message: "The pipeline finished but no predictions were generated.",
      hasPredictions,
    };
  }

  return {
    state: saved.state,
    message: saved.message || "Processing FPL data",
    updatedAt: saved.updatedAt,
    hasPredictions,
  };
}

export function startPipelineIfNeeded(): PipelineStatus {
  const status = getPipelineStatus();
  if (["checking", "training", "predicting"].includes(status.state)) {
    return status;
  }
  if (status.hasPredictions) return status;

  fs.mkdirSync(outputDir, { recursive: true });
  const logFd = fs.openSync(path.join(outputDir, "pipeline.log"), "a");
  const python = process.env.FPL_PYTHON || "python3";

  // Publish the initial state before spawning to avoid overwriting
  // a faster status update from Python.
  fs.writeFileSync(statusFile, JSON.stringify({
    state: "checking",
    message: "Starting FPL model preparation",
    updatedAt: new Date().toISOString(),
    pid: null,
  }));

  try {
    const worker = spawn(python, ["scheduler.py"], {
      cwd: projectDir,
      detached: true,
      stdio: ["ignore", logFd, logFd],
    });

    worker.on("error", (error) => {
      fs.writeFileSync(statusFile, JSON.stringify({
        state: "error",
        message: error.message,
        updatedAt: new Date().toISOString(),
      }));
    });
    worker.unref();
  } catch (error) {
    fs.writeFileSync(statusFile, JSON.stringify({
      state: "error",
      message: error instanceof Error ? error.message : "Could not start Python",
      updatedAt: new Date().toISOString(),
    }));
    throw error;
  } finally {
    fs.closeSync(logFd);
  }

  return getPipelineStatus();
}
