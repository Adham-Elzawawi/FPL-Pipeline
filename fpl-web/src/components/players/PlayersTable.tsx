
"use client";

import { useState } from "react";
import type { Player } from "@/lib/predictions";

type Props = {
  gameweeks: number[];
  playersByGw: Record<number, Player[]>;
};

export default function PlayersTable({ gameweeks, playersByGw }: Props) {
  const [selectedGw, setSelectedGw] = useState(gameweeks[0] ?? 0);
  const [position, setPosition] = useState("ALL");
  const [team, setTeam] = useState("ALL");
  const [minPrice, setMinPrice] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [search, setSearch] = useState("");

  // Players for the selected gameweek
  const players = playersByGw[selectedGw] ?? [];

  // Teams available across all loaded gameweeks
  const allPlayers = Object.values(playersByGw).flat();

  const teams = [
    ...new Set(allPlayers.map((p) => p.team_name)),
  ].sort();

  // Apply filters
  const filteredPlayers = players
    .filter((player) => {
      const price = Number(player.price_m);

      if (position !== "ALL" && player.position !== position) {
        return false;
      }

      if (team !== "ALL" && player.team_name !== team) {
        return false;
      }

      if (minPrice !== "" && price < Number(minPrice)) {
        return false;
      }

      if (maxPrice !== "" && price > Number(maxPrice)) {
        return false;
      }

      if (
        !player.web_name.toLowerCase().includes(search.toLowerCase())
      ) {
        return false;
      }

      return true;
    })
    .sort(
      (a, b) =>
        Number(b.expected_points) - Number(a.expected_points)
    );

  function resetFilters() {
    setPosition("ALL");
    setTeam("ALL");
    setMinPrice("");
    setMaxPrice("");
    setSearch("");
  }

  const inputStyle =
    "w-full bg-slate-800 border border-slate-700 rounded-lg p-2 mt-2 text-white";

  if (gameweeks.length === 0) {
    return <p>No predictions yet. Run main.py first.</p>;
  }

  return (
    <div>
      {/* Gameweek selector */}
      <div className="flex items-center gap-3 mb-6">
        <label className="font-semibold text-slate-300">
          Gameweek
        </label>

        <select
          value={selectedGw}
          onChange={(e) => setSelectedGw(Number(e.target.value))}
          className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white"
        >
          {gameweeks.map((gw) => (
            <option key={gw} value={gw}>
              GW {gw}
            </option>
          ))}
        </select>
      </div>

      {/* Filters */}
      <div className="bg-slate-900 rounded-xl p-5 mb-6">
        <h2 className="text-lg font-semibold mb-4">
          Filters
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {/* Player search */}
          <div>
            <label className="text-sm text-slate-400">
              Player
            </label>
            <input
              type="text"
              placeholder="Search player..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className={inputStyle}
            />
          </div>

          {/* Position */}
          <div>
            <label className="text-sm text-slate-400">
              Position
            </label>
            <select
              value={position}
              onChange={(e) => setPosition(e.target.value)}
              className={inputStyle}
            >
              <option value="ALL">All positions</option>
              <option value="GK">Goalkeeper</option>
              <option value="DEF">Defender</option>
              <option value="MID">Midfielder</option>
              <option value="FWD">Forward</option>
            </select>
          </div>

          {/* Team */}
          <div>
            <label className="text-sm text-slate-400">
              Team
            </label>
            <select
              value={team}
              onChange={(e) => setTeam(e.target.value)}
              className={inputStyle}
            >
              <option value="ALL">All teams</option>
              {teams.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>

          {/* Minimum price */}
          <div>
            <label className="text-sm text-slate-400">
              Min price (£m)
            </label>
            <input
              type="number"
              min="0"
              step="0.1"
              placeholder="Min"
              value={minPrice}
              onChange={(e) => setMinPrice(e.target.value)}
              className={inputStyle}
            />
          </div>

          {/* Maximum price */}
          <div>
            <label className="text-sm text-slate-400">
              Max price (£m)
            </label>
            <input
              type="number"
              min="0"
              step="0.1"
              placeholder="Max"
              value={maxPrice}
              onChange={(e) => setMaxPrice(e.target.value)}
              className={inputStyle}
            />
          </div>
        </div>

        <button
          onClick={resetFilters}
          className="mt-4 text-sm text-green-400 hover:text-green-300"
        >
          Reset filters
        </button>
      </div>

      {/* Player count */}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">
          Player Rankings
        </h2>

        <p className="text-slate-400 text-sm">
          {filteredPlayers.length} players
        </p>
      </div>

      {/* Players table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800">
        <table className="w-full text-left">
          <thead className="bg-slate-900 text-slate-400 text-sm">
            <tr>
              <th className="p-4">#</th>
              <th className="p-4">Player</th>
              <th className="p-4">Team</th>
              <th className="p-4">Position</th>
              <th className="p-4">Opponent</th>
              <th className="p-4">Price</th>
              <th className="p-4 text-right">xPts</th>
            </tr>
          </thead>

          <tbody>
            {filteredPlayers.map((player, index) => (
              <tr
                key={player.id}
                className="border-t border-slate-800 hover:bg-slate-900/70"
              >
                <td className="p-4 text-slate-500">
                  {index + 1}
                </td>

                <td className="p-4 font-semibold">
                  {player.web_name}
                </td>

                <td className="p-4 text-slate-400">
                  {player.team_name}
                </td>

                <td className="p-4">
                  {player.position}
                </td>

                <td className="p-4 text-slate-400">
                  {player.opponents}
                </td>

                <td className="p-4">
                  £{Number(player.price_m).toFixed(1)}m
                </td>

                <td className="p-4 text-right text-green-400 font-bold">
                  {Number(player.expected_points).toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* No results */}
        {filteredPlayers.length === 0 && (
          <p className="text-center text-slate-400 py-8">
            {players.length === 0
              ? `No predictions available for GW ${selectedGw}.`
              : "No players match your filters."}
          </p>
        )}
      </div>
    </div>
  );
}
