
import { getPlayers } from "@/lib/predictions";
import FootballPitch from "@/components/team/FootballPitch";
import TeamStats from "@/components/team/TeamStats";

export default function TeamPage() {
  const players = getPlayers("starting_xi");

  if (players.length === 0) {
    return <p>No team found. Run your Python predictor first.</p>;
  }

  return (
    <div className="max-w-4xl mx-auto">

      <h1 className="text-3xl font-bold mb-2">
        Optimal XI
      </h1>

      <p className="text-slate-400 mb-6">
        Your highest predicted scoring team
      </p>

      <TeamStats players={players} />

      <FootballPitch players={players} />

      <p className="text-slate-400 text-sm text-center mt-4">
        C = Captain (2x points) · VC = Vice-Captain
      </p>

    </div>
  );
}
