
"use client";

import { useState } from "react";
import Image from "next/image";

export default function FixturesPage() {
  const [mode, setMode] = useState<"attack" | "defence">("attack");

  const image =
    mode === "attack"
      ? "/fixtures/next_4_att_fdr.png"
      : "/fixtures/next_4_def_fdr.png";

  return (
    <div className="pb-28">
      <h1 className="text-3xl font-bold mb-2">
        Dynamic Fixture Difficulty
      </h1>

      <p className="text-slate-400 mb-6">
        Next 4 gameweeks — {mode === "attack" ? "Attacking" : "Defensive"} FDR
      </p>

      {/* Fixture heatmap */}
      <div className="bg-white rounded-xl p-3 overflow-x-auto">
        <Image
          src={image}
          alt={`${mode} fixture difficulty`}
          width={840}
          height={1000}
          unoptimized
          className="w-full min-w-[700px] h-auto"
        />
      </div>

      {/* Fixed bottom switch */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50">
        <div className="flex gap-2 p-2 bg-slate-900 border border-slate-700 rounded-full shadow-2xl">

          <button
            onClick={() => setMode("attack")}
            className={`px-8 py-3 rounded-full font-semibold transition ${
              mode === "attack"
                ? "bg-green-500 text-slate-950"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Attack
          </button>

          <button
            onClick={() => setMode("defence")}
            className={`px-8 py-3 rounded-full font-semibold transition ${
              mode === "defence"
                ? "bg-green-500 text-slate-950"
                : "text-slate-400 hover:text-white"
            }`}
          >
            Defence
          </button>

        </div>
      </div>
    </div>
  );
}
