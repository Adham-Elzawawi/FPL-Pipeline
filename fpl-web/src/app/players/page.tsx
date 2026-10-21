
import { getPlayers } from "@/lib/predictions";


export default function PlayersPage() {
  const players = getPlayers("player_ratings");

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Player Rankings</h1>

      {players.length === 0 && <p>No predictions yet. Run main.py first.</p>}

      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead className="bg-slate-800">
            <tr>
              <th className="p-3">Player</th>
              <th className="p-3">Team</th>
              <th className="p-3">Position</th>
              <th className="p-3">Fixture</th>
              <th className="p-3">Price</th>
              <th className="p-3">xPts</th>
            </tr>
          </thead>
          <tbody>
            {players.map((player) => (
              <tr key={player.id} className="border-b border-slate-800">
                <td className="p-3">{player.web_name}</td>
                <td className="p-3">{player.team_name}</td>
                <td className="p-3">{player.position}</td>
                <td className="p-3">{player.opponents}</td>
                <td className="p-3">£{player.price_m}m</td>
                <td className="p-3 text-green-400 font-bold">
                  {player.expected_points}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
