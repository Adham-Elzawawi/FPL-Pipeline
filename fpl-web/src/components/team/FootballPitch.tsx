
import type { Player } from "@/lib/predictions";
import PlayerCard from "./PlayerCard";

type Props = {
  players: Player[];
};

export default function FootballPitch({ players }: Props) {
  const goalkeepers = players.filter(p => p.position === "GK");
  const defenders = players.filter(p => p.position === "DEF");
  const midfielders = players.filter(p => p.position === "MID");
  const forwards = players.filter(p => p.position === "FWD");

  const rows = [forwards, midfielders, defenders, goalkeepers];

  return (
    <div
      className="relative rounded-xl overflow-hidden shadow-2xl"
      style={{
        background:
          "repeating-linear-gradient(to bottom, #167c49 0px, #167c49 85px, #198c53 85px, #198c53 170px)",
      }}
    >
      {/* Pitch lines */}
      <div className="absolute inset-3 sm:inset-6 border-2 border-white/30" />

      <div className="absolute top-1/2 left-3 right-3 border-t-2 border-white/30" />

      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-24 h-24 rounded-full border-2 border-white/30" />

      <div className="absolute top-3 left-1/2 -translate-x-1/2 w-40 h-14 border-2 border-t-0 border-white/30" />

      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 w-40 h-14 border-2 border-b-0 border-white/30" />

      {/* Players */}
      <div className="relative z-10 flex flex-col justify-around h-[560px] sm:h-[700px] py-8">

        {rows.map((row, index) => (
          <div
            key={index}
            className="flex justify-evenly items-center w-full"
          >
            {row.map(player => (
              <PlayerCard key={player.id} player={player} />
            ))}
          </div>
        ))}

      </div>
    </div>
  );
}
