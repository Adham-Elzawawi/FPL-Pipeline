
import { getPlayers } from "@/lib/predictions";


export default function Home() {
  const players = getPlayers("player_ratings");
  const topPlayers = players.slice(0, 5);

  return (
    <div>
      <h1 className="text-3xl font-bold mb-3">FPL Dashboard</h1>
      <p className="text-slate-400 mb-8">
        Gameweek {players[0]?.gameweek || "-"} Predictions
      </p>

      <h2 className="text-xl font-semibold mb-4">Top 5 Players</h2>

      {topPlayers.length === 0 && <p>No predictions yet. Run main.py first.</p>}

      <div className="grid gap-3">
        {topPlayers.map((player) => (
          <div key={player.id} className="bg-slate-900 rounded-lg p-4 flex justify-between">
            <div>
              <p className="font-semibold">{player.web_name}</p>
              <p className="text-slate-400 text-sm">
                {player.team_name} · {player.position}
              </p>
            </div>
            <p className="text-green-400 font-bold text-xl">
              {player.expected_points} xPts
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
