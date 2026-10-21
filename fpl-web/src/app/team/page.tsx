
import { getOptimalTeam } from "@/lib/predictions";
import TeamHorizon from "@/components/team/TeamHorizon";

export default function TeamPage() {
  const teams = [1, 2, 3, 4].map((weeks) =>
    getOptimalTeam(weeks)
  );

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">
        Optimal XI
      </h1>

      <p className="text-slate-400 mb-6">
        Find the best team for the next 1 to 4 gameweeks.
      </p>

      <TeamHorizon teams={teams} />
    </div>
  );
}
