
import { getPlayers } from "@/lib/predictions";


export default function TeamPage() {
  const team = getPlayers("starting_xi");

  const total = team.reduce(
    (sum, player) => sum + Number(player.expected_points), 0
  );

  const captain = team.find((player) => player.captain === "C");
  const totalPoints = total + Number(captain?.expected_points || 0);

  return (
    <div>
      <h1 className="text-3xl font-bold mb-3">Optimal Starting XI</h1>

      <p className="text-green-400 text-xl mb-8">
        Predicted points: {totalPoints.toFixed(2)}
      </p>

      {team.length === 0 && <p>No optimized team yet. Run main.py first.</p>}

      <div className="grid gap-3">
        {team.map((player) => (
          <div key={player.id} className="bg-slate-900 p-4 rounded-lg flex justify-between">
            <div>
              <p className="font-bold">
                {player.web_name} {player.captain && `(${player.captain})`}
              </p>
              <p className="text-slate-400 text-sm">
                {player.position} · {player.team_name} · £{player.price_m}m
              </p>
            </div>

            <p className="text-green-400 font-semibold">
              {player.expected_points} xPts
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
