import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";

export type Player = {
  id: string;
  gameweek: string;
  web_name: string;
  team_name: string;
  position: string;
  opponents: string;
  price_m: string;
  expected_points: string;
  captain?: string;
};


export function getOptimalTeam(weeks: number): Player[] {
  const file = path.join(
    process.cwd(),
    "..",
    "outputs",
    `optimal_xi_${weeks}gw.csv`
  );

  if (!fs.existsSync(file)) {
    return [];
  }

  const csv = fs.readFileSync(file, "utf8");

  return parse(csv, {
    columns: true,
    skip_empty_lines: true,
  });
}


export function getPlayers(type: "player_ratings" | "starting_xi"): Player[] {
  const folder = path.join(process.cwd(), "..", "outputs");

  if (!fs.existsSync(folder)) return [];

  const files = fs.readdirSync(folder)
    .filter((file) => file.match(new RegExp(`^gw_\\d+_${type}\\.csv$`)))
    .sort((a, b) => Number(b.split("_")[1]) - Number(a.split("_")[1]));

  if (files.length === 0) return [];

  const csv = fs.readFileSync(path.join(folder, files[0]), "utf8");

  return parse(csv, {
    columns: true,
    skip_empty_lines: true,
  });
}
