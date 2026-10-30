import { NextResponse } from "next/server";
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { getPipelineStatus, startPipelineIfNeeded } from "@/lib/pipelineServer";

// This API runs on the same machine as the Python FPL pipeline.
// Node.js is the default runtime when cacheComponents is enabled.

const projectDir = path.resolve(process.cwd(), "..");
const outputDir = path.join(projectDir, "outputs", "manager_advice");
const predictionFile = path.join(projectDir, "outputs", "current_prediction.json");

function getId(value: unknown): string | null {
  const id = String(value ?? "").trim();
  return /^[1-9][0-9]{0,9}$/.test(id) ? id : null;
}

type StoredData = {
  state?: string;
  message?: string;
  updatedAt?: string;
  predictionGeneratedAt?: string;
  gameweeks?: number[];
  shapVersion?: number;
  generated_at?: string;
  gameweek?: number;
};

function loadJson(file: string): StoredData | null {
  if (!fs.existsSync(file)) return null;
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

function filesFor(id: string) {
  return {
    advice: path.join(outputDir, `manager_${id}.json`),
    status: path.join(outputDir, `manager_${id}.status.json`),
    log: path.join(outputDir, `manager_${id}.log`),
  };
}

function getState(id: string) {
  const pipeline = getPipelineStatus();
  if (["checking", "training", "predicting"].includes(pipeline.state)) {
    return { state: "training", message: pipeline.message };
  }
  const files = filesFor(id);
  const advice = loadJson(files.advice);
  const status = loadJson(files.status);
  const prediction = loadJson(predictionFile);

  const started = Date.parse(String(status?.updatedAt ?? ""));
  const running = status?.state === "running" &&
    Number.isFinite(started) && Date.now() - started < 15 * 60 * 1000;

  if (running) return { state: "processing" };

  const fresh = advice && prediction &&
    advice.predictionGeneratedAt === prediction.generated_at &&
    advice.gameweeks?.[0] === prediction.gameweek &&
    advice.shapVersion === 1;

  if (fresh) return { state: "ready", advice };

  if (status?.state === "running") {
    return { state: "error", error: "Optimization stopped or took too long. Try again." };
  }

  if (status?.state === "error") {
    return { state: "error", error: status.message || "Optimization failed." };
  }

  return { state: "idle" };
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const id = getId(url.searchParams.get("managerId"));
  if (!id) {
    return NextResponse.json({ error: "Enter a valid FPL manager ID." }, { status: 400 });
  }
  return NextResponse.json(getState(id), {
    headers: { "Cache-Control": "no-store" },
  });
}

export async function POST(request: Request) {
  let body: { managerId?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid request." }, { status: 400 });
  }

  const id = getId(body.managerId);
  if (!id) {
    return NextResponse.json({ error: "Enter a valid FPL manager ID." }, { status: 400 });
  }

  if (!fs.existsSync(predictionFile)) {
    // Initialize the saved model and predictions on the first request.
    try {
      const pipeline = startPipelineIfNeeded();
      if (pipeline.state === "error") {
        return NextResponse.json({ state: "error", error: pipeline.message }, { status: 503 });
      }
      return NextResponse.json({ state: "training", message: pipeline.message }, {
        status: 202,
        headers: { "Cache-Control": "no-store" },
      });
    } catch (error) {
      return NextResponse.json({
        state: "error", error: error instanceof Error ? error.message : "Could not start Python",
      }, { status: 500 });
    }
  }

  const current = getState(id);
  if (current.state === "training" || current.state === "processing" || current.state === "ready") {
    return NextResponse.json(current, {
      headers: { "Cache-Control": "no-store" },
    });
  }

  // No shell is used. The supplied ID is always a validated integer.
  fs.mkdirSync(outputDir, { recursive: true });
  const files = filesFor(id);
  const fd = fs.openSync(files.log, "a");
  const python = process.env.FPL_PYTHON || "python3";

  fs.writeFileSync(files.status, JSON.stringify({
    state: "running", updatedAt: new Date().toISOString(),
  }));

  try {
    const worker = spawn(
      python,
      ["manager_web_job.py", "--id", id],
      {
        cwd: projectDir,
        detached: true,
        stdio: ["ignore", fd, fd],
      }
    );

    worker.on("error", (error) => {
      fs.writeFileSync(files.status, JSON.stringify({
        state: "error", message: error.message,
        updatedAt: new Date().toISOString(),
      }));
    });
    worker.unref();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not start Python.";
    return NextResponse.json({ error: message }, { status: 500 });
  } finally {
    fs.closeSync(fd);
  }

  return NextResponse.json({ state: "processing" }, {
    status: 202,
    headers: { "Cache-Control": "no-store" },
  });
}
