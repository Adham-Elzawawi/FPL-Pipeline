
import {
  getPlayers,
  getCurrentGameweek,
  type Player,
} from "@/lib/predictions";

import PlayersTable from "@/components/players/PlayersTable";


export default function PlayersPage() {
  const nextGw = getCurrentGameweek();

  const gameweeks = nextGw === undefined
    ? []
    : Array.from({ length: 4 }, (_, i) => nextGw + i)
        .filter((gw) => gw <= 38);

  const playersByGw: Record<number, Player[]> = {};

  for (const gw of gameweeks) {
    playersByGw[gw] = getPlayers("player_ratings", gw);
  }

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">
        Player Predictions
      </h1>

      <PlayersTable
        gameweeks={gameweeks}
        playersByGw={playersByGw}
      />
    </div>
  );
}
