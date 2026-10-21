
import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";

export async function GET() {
  const file = path.join(
    process.cwd(),
    "..",
    "outputs",
    "next_4_fixtures.csv"
  );

  if (!fs.existsSync(file)) {
    return NextResponse.json(
      { error: "Run the Python predictor first" },
      { status: 404 }
    );
  }

  const csv = fs.readFileSync(file, "utf8");

  const fixtures = parse(csv, {
    columns: true,
    skip_empty_lines: true,
    cast: true,
  });

  return NextResponse.json(fixtures, {
    headers: { "Cache-Control": "no-store" },
  });
}
