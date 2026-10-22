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

type PredictionType = "player_ratings" | "starting_xi";


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




export function getAvailableGameweeks(type: PredictionType): number[] {
  const folder = path.join(process.cwd(), "..", "outputs");

  if (!fs.existsSync(folder)) return [];

  const regex = new RegExp(`^gw_(\\d+)_${type}\\.csv$`);

  return fs.readdirSync(folder)
    .map((file) => file.match(regex))
    .filter((match): match is RegExpMatchArray => match !== null)
    .map((match) => Number(match[1]))
    .sort((a, b) => a - b);
}

export function getPlayers(type: PredictionType, gw?: number): Player[] {
  const folder = path.join(process.cwd(), "..", "outputs");

  const selectedGw = gw ?? getAvailableGameweeks(type)[0];

  if (selectedGw === undefined) return [];

  const file = path.join(folder, `gw_${selectedGw}_${type}.csv`);

  if (!fs.existsSync(file)) return [];

  const csv = fs.readFileSync(file, "utf8");

  return parse(csv, {
    columns: true,
    skip_empty_lines: true,
  });
}

