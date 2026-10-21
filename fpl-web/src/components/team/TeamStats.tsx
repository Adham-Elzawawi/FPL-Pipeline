
import type { Player } from "@/lib/predictions";

type Props = {
  players: Player[];
};

export default function TeamStats({ players }: Props) {
  const defenders = players.filter(p => p.position === "DEF").length;
  const midfielders = players.filter(p => p.position === "MID").length;
  const forwards = players.filter(p => p.position === "FWD").length;

  const formation = `${defenders}-${midfielders}-${forwards}`;

  const captain = players.find(p => p.captain === "C");

  const points =
    players.reduce((sum, p) => sum + Number(p.expected_points), 0) +
    Number(captain?.expected_points || 0);

  const cost = players.reduce(
    (sum, p) => sum + Number(p.price_m), 0
  );

  const stats = [
    { label: "Formation", value: formation },
    { label: "Predicted Points", value: points.toFixed(2) },
    { label: "Team Cost", value: `£${cost.toFixed(1)}m` },
  ];

  return (
    <div className="grid grid-cols-3 gap-3 mb-6">
      {stats.map(stat => (
        <div
          key={stat.label}
          className="bg-slate-900 rounded-xl p-4 text-center"
        >
          <p className="text-slate-400 text-xs mb-2">
            {stat.label}
          </p>

          <p className="text-lg sm:text-xl font-bold">
            {stat.value}
          </p>
        </div>
      ))}
    </div>
  );
}
