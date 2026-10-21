
"use client";

import { useEffect, useState } from "react";

type Fixture = {
  team: number;
  team_name: string;
  opponent_name: string;
  round: number;
  was_home: number;
  att_fdr: number;
  def_fdr: number;
  att_fdr_raw: number;
  def_fdr_raw: number;
};

function getColor(difficulty: number) {
  if (difficulty <= 1) return "bg-green-800 text-white";
  if (difficulty === 2) return "bg-green-500 text-slate-950";
  if (difficulty === 3) return "bg-yellow-400 text-slate-950";
  if (difficulty === 4) return "bg-orange-500 text-slate-950";
  return "bg-red-700 text-white";
}

export default function FixturesPage() {
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [mode, setMode] = useState<"attack" | "defence">("attack");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/fixtures", { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error("Could not load fixtures");
        return res.json();
      })
      .then((data) => setFixtures(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const gameweeks = [...new Set(fixtures.map((f) => f.round))]
    .sort((a, b) => a - b)
    .slice(0, 4);

  const teams = [...new Map(
    fixtures.map((f) => [f.team, f.team_name])
  ).entries()].sort((a, b) => a[1].localeCompare(b[1]));

  return (
    <div className="pb-28">
      <h1 className="text-3xl font-bold mb-2">
        Dynamic Fixture Difficulty
      </h1>

      <p className="text-slate-400 mb-6">
        Next 4 gameweeks — {mode === "attack" ? "Attacking" : "Defensive"} FDR
      </p>

      {loading && <p>Loading fixtures...</p>}
      {error && <p className="text-red-400">{error}</p>}

      {!loading && !error && (
        <div className="overflow-x-auto">
          <table className="w-full border-separate border-spacing-1 text-center">
            <thead>
              <tr>
                <th className="p-3 text-left">Team</th>
                {gameweeks.map((gw) => (
                  <th key={gw} className="p-3">GW {gw}</th>
                ))}
              </tr>
            </thead>

            <tbody>
              {teams.map(([teamId, teamName]) => (
                <tr key={teamId}>
                  <td className="p-3 text-left font-semibold whitespace-nowrap">
                    {teamName}
                  </td>

                  {gameweeks.map((gw) => {
                    const matches = fixtures.filter(
                      (f) => f.team === teamId && f.round === gw
                    );

                    return (
                      <td key={gw} className="p-1 min-w-28">
                        {matches.length === 0 ? (
                          <div className="bg-slate-800 p-3 rounded-lg">-</div>
                        ) : (
                          <div className="flex flex-col gap-1">
                            {matches.map((match, index) => {
                              const difficulty = mode === "attack"
                                ? match.att_fdr
                                : match.def_fdr;

                              const raw = mode === "attack"
                                ? match.att_fdr_raw
                                : match.def_fdr_raw;

                              return (
                                <div
                                  key={index}
                                  title={`Dynamic FDR: ${Number(raw).toFixed(2)}`}
                                  className={`${getColor(difficulty)} p-3 rounded-lg`}
                                >
                                  <p className="font-semibold">
                                    {match.opponent_name} ({match.was_home === 1 ? "H" : "A"})
                                  </p>
                                  <p className="text-xs mt-1">
                                    FDR {Number(raw).toFixed(2)}
                                  </p>
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Fixed bottom switch */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50">
        <div className="flex gap-2 p-2 bg-slate-900 border border-slate-700 rounded-full shadow-xl">
          <button
            onClick={() => setMode("attack")}
            className={`px-7 py-3 rounded-full font-semibold ${
              mode === "attack"
                ? "bg-green-500 text-black"
                : "text-slate-400"
            }`}
          >
            Attack
          </button>

          <button
            onClick={() => setMode("defence")}
            className={`px-7 py-3 rounded-full font-semibold ${
              mode === "defence"
                ? "bg-green-500 text-black"
                : "text-slate-400"
            }`}
          >
            Defence
          </button>
        </div>
      </div>
    </div>
  );
}
