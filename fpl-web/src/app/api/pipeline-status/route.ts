import { NextResponse } from "next/server";
import { getPipelineStatus } from "@/lib/pipelineServer";

// Read-only status endpoint. Starting a model is handled by the manager API.
export async function GET() {
  return NextResponse.json(getPipelineStatus(), {
    headers: { "Cache-Control": "no-store" },
  });
}
