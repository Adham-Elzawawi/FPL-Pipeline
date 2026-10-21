
"use client";

import { useState } from "react";
import type { Player } from "@/lib/predictions";
import FootballPitch from "./FootballPitch";
import TeamStats from "./TeamStats";

type Props = {
  teams: Player[][];
};

export default function TeamHorizon({ teams }: Props) {
  const [weeks, setWeeks] = useState(1);
  const players = teams[weeks - 1];

  return (
    <div>
      {players.length === 0 ? (
        <p className="text-slate-400 mb-6">
          No predictions found for this period.
        </p>
      ) : (
        <>
          <TeamStats players={players} />
          <FootballPitch players={players} />

          <p className="text-slate-400 text-sm text-center mt-4">
            C = Captain (2x points) · VC = Vice-Captain
          </p>
        </>
      )}

      {/* Slider at the bottom of the page */}
      <div className="bg-slate-900 border border-slate-700 rounded-xl p-6 mt-8">
        <div className="flex justify-between items-center mb-5">
          <h2 className="font-semibold">
            Optimization Period
          </h2>

          <span className="text-green-400 font-bold">
            {weeks} {weeks === 1 ? "GW" : "GWs"}
          </span>
        </div>

        <input
          type="range"
          min={1}
          max={4}
          step={1}
          value={weeks}
          onChange={(e) => setWeeks(Number(e.target.value))}
          className="w-full accent-green-500 cursor-pointer"
        />

        <div className="flex justify-between text-xs text-slate-400 mt-3">
          <span>1 GW</span>
          <span>2 GWs</span>
          <span>3 GWs</span>
          <span>4 GWs</span>
        </div>
      </div>
    </div>
  );
}
