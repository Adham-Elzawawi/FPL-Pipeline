
import type { Player } from "@/lib/predictions";
import PlayerShirt from "./PlayerShirt";
import { teamNames } from "@/lib/teamColors";

type Props = {
  player: Player;
};

export default function PlayerCard({ player }: Props) {
  return (
    <div className="flex flex-col items-center w-[62px] sm:w-24 text-center">

      <div className="relative">
        <PlayerShirt team={player.team_name} />

        {player.captain && (
          <span className="absolute -top-1 -right-2 bg-yellow-400 text-black text-xs font-bold rounded-full w-5 h-5 flex items-center justify-center">
            {player.captain}
          </span>
        )}
      </div>

      <div className="bg-slate-950 text-white rounded-t w-full p-1">
        <p className="text-[10px] sm:text-xs font-bold truncate">
          {player.web_name}
        </p>

        <p
          className="text-[9px] text-slate-300 truncate"
          title={teamNames[player.team_name] || player.team_name}
        >
          {teamNames[player.team_name] || player.team_name}
        </p>
      </div>

      <div className="bg-white text-slate-900 rounded-b w-full p-1">
        <p className="text-[10px] sm:text-xs font-semibold">
          {Number(player.expected_points).toFixed(2)} xPts
        </p>
      </div>

    </div>
  );
}
