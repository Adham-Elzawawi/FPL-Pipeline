"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

type Player = {
  id: number;
  name: string;
  position: string;
  club: string;
  price: number;
  expectedPoints?: number;
};

type Advice = {
  managerId: number;
  savedGw: number;
  gameweeks: number[];
  freeTransfers: number;
  transfersCount: number;
  currentPoints: number;
  recommendedPoints: number;
  gain: number;
  remainingBank: number;
  transfers: { sell: Player; buy: Player }[];
  lineups: {
    gameweek: number;
    expectedPoints: number;
    captainId: number;
    starters: Player[];
  }[];
  generatedAt: string;
};

type ApiResult = {
  state?: "idle" | "training" | "processing" | "ready" | "error";
  advice?: Advice;
  error?: string;
  message?: string;
};

export default function MyTeamPage() {
  const router = useRouter();
  const [managerId, setManagerId] = useState("");
  const [activeId, setActiveId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [advice, setAdvice] = useState<Advice | null>(null);
  const [pendingAdvice, setPendingAdvice] = useState<Advice | null>(null);
  const [preparingExplanations, setPreparingExplanations] = useState(false);
  const [selectedGw, setSelectedGw] = useState<number | null>(null);
  const [explanations, setExplanations] = useState<Record<string, string>>({});
  const [explanationErrors, setExplanationErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!loading || !activeId) return;

    let cancelled = false;
    const interval = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/manager-advice?managerId=${activeId}`, {
          cache: "no-store",
        });
        const data: ApiResult = await response.json();
        if (cancelled) return;
        if (data.state === "training") {
          router.push(`/model-loading?managerId=${activeId}`);
          return;
        }
        if (data.state === "ready" && data.advice) {
          setPendingAdvice(data.advice);
          setSelectedGw(data.advice.gameweeks[0] ?? null);
          setLoading(false);
        } else if (data.state === "error" || !response.ok) {
          setError(data.error || "Could not calculate the recommendations.");
          setLoading(false);
        }
      } catch {
        if (!cancelled) {
          setError("Connection problem. Try again in a moment.");
          setLoading(false);
        }
      }
    }, 3000);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeId, loading, router]);

  const analyzeManager = useCallback(async (id: string) => {
    setError("");
    setAdvice(null);
    setPendingAdvice(null);
    setPreparingExplanations(false);
    setExplanations({});
    setExplanationErrors({});
    setLoading(true);
    setActiveId(id);

    try {
      const response = await fetch("/api/manager-advice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ managerId: id }),
      });
      const data: ApiResult = await response.json();
      if (data.state === "training") {
        router.push(`/model-loading?managerId=${id}`);
        return;
      }
      if (!response.ok || data.state === "error") {
        throw new Error(data.error || "The optimizer could not start.");
      }
      if (data.state === "ready" && data.advice) {
        setPendingAdvice(data.advice);
        setSelectedGw(data.advice.gameweeks[0] ?? null);
        setLoading(false);
      }
      // Otherwise the manager optimizer is still running; poll for results.
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("managerId") || "";
    if (params.get("auto") === "1" && /^[1-9][0-9]{0,9}$/.test(id)) {
      window.history.replaceState(null, "", "/my-team");
      const timer = window.setTimeout(() => {
        setManagerId(id);
        void analyzeManager(id);
      }, 0);
      return () => window.clearTimeout(timer);
    }
  }, [analyzeManager]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const id = managerId.trim();
    if (!/^[1-9][0-9]{0,9}$/.test(id)) {
      setError("Please enter a valid numeric FPL manager ID.");
      return;
    }
    await analyzeManager(id);
  }

  // Prepare every transfer explanation before showing the recommendations.
  // This way the user sees the player, transfer and explanation together.
  useEffect(() => {
    if (!pendingAdvice) return;

    const controller = new AbortController();
    let cancelled = false;
    setPreparingExplanations(true);

    async function prepareRecommendations() {
      if (!pendingAdvice) return;

      const preparedExplanations: Record<string, string> = {};
      const preparedErrors: Record<string, string> = {};

      // Sequential calls are safer with Gemini's free-tier rate limits.
      for (const transfer of pendingAdvice.transfers) {
        if (controller.signal.aborted) return;
        const key = `transfer:${transfer.sell.id}:${transfer.buy.id}`;

        try {
          const response = await fetch("/api/player-explanation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              managerId: pendingAdvice.managerId,
              type: "transfer",
              sellId: transfer.sell.id,
              buyId: transfer.buy.id,
            }),
            signal: controller.signal,
          });
          const data: { explanation?: string; error?: string } = await response.json();
          if (!response.ok || !data.explanation) {
            throw new Error(data.error || "Could not generate this explanation.");
          }
          preparedExplanations[key] = data.explanation;
        } catch (err) {
          if (controller.signal.aborted) return;
          preparedErrors[key] = err instanceof Error ? err.message : "Connection problem.";
        }
      }

      if (cancelled) return;
      setExplanations(preparedExplanations);
      setExplanationErrors(preparedErrors);
      setAdvice(pendingAdvice);
      setPreparingExplanations(false);
      setPendingAdvice(null);
    }

    void prepareRecommendations();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [pendingAdvice]);

  const lineup = advice?.lineups.find((item) => item.gameweek === selectedGw);
  const positions = ["GK", "DEF", "MID", "FWD"];

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <header className="space-y-3">
        <p className="text-sm font-semibold uppercase tracking-widest text-green-400">Personalized FPL advice</p>
        <h1 className="text-4xl font-bold tracking-tight">Improve your squad</h1>
        <p className="max-w-2xl text-slate-400">
          Enter your FPL manager ID to see which players to sell and buy, based on
          predicted points over the next four gameweeks.
        </p>
      </header>

      <form onSubmit={submit} className="rounded-2xl border border-slate-800 bg-slate-900 p-5 sm:p-6">
        <label htmlFor="manager-id" className="mb-2 block text-sm font-medium text-slate-200">
          FPL Manager ID
        </label>
        <div className="flex flex-col gap-3 sm:flex-row">
          <input
            id="manager-id"
            type="text"
            inputMode="numeric"
            pattern="[0-9]*"
            value={managerId}
            onChange={(event) => setManagerId(event.target.value)}
            placeholder="e.g. 1234567"
            className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-green-500"
          />
          <button
            type="submit"
            disabled={loading}
            className="rounded-lg bg-green-500 px-6 py-3 font-bold text-slate-950 hover:bg-green-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Analyzing..." : "Analyze my team"}
          </button>
        </div>
        <p className="mt-3 text-xs text-slate-500">
          Your manager ID is the number in your FPL URL: fantasy.premierleague.com/entry/1234567/...
        </p>
      </form>

      {(loading || preparingExplanations || pendingAdvice) && (
        <div role="status" className="rounded-xl border border-slate-800 bg-slate-900 p-5 text-slate-300">
          {loading
            ? "Optimizing your squad using the latest saved predictions. This may take a few minutes."
            : "Preparing your transfer recommendations and explanations..."}
        </div>
      )}
      {error && (
        <div role="alert" className="rounded-xl border border-red-900 bg-red-950/40 p-5 text-red-300">
          {error}
        </div>
      )}

      {advice && (
        <div className="space-y-8">
          <div className="grid gap-3 sm:grid-cols-3">
            <Stat label="Free transfers (estimated)" value={String(advice.freeTransfers)} />
            <Stat label="Recommended transfers" value={String(advice.transfersCount)} />
            <Stat label="Expected gain over horizon" value={`${advice.gain > 0 ? "+" : ""}${advice.gain.toFixed(1)} xPts`} />
          </div>

          <section className="space-y-4">
            <div>
              <h2 className="text-2xl font-bold">Transfer recommendations</h2>
              <p className="mt-1 text-sm text-slate-400">
                Based on your last publicly saved squad (GW{advice.savedGw}).
              </p>
            </div>
            {advice.transfers.length === 0 ? (
              <div className="rounded-2xl border border-green-900 bg-green-950/30 p-6">
                <h3 className="text-xl font-semibold text-green-300">Keep your current squad</h3>
                <p className="mt-2 text-slate-300">
                  The optimizer did not find transfers that passed its expected-gain threshold.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {advice.transfers.map((transfer) => {
                  const key = `transfer:${transfer.sell.id}:${transfer.buy.id}`;
                  return (
                    <div key={key} className="rounded-xl border border-slate-800 bg-slate-900 p-4">
                      <div className="grid gap-3 sm:grid-cols-[1fr_auto_1fr] sm:items-center">
                        <TransferPlayer player={transfer.sell} action="SELL" />
                        <div aria-hidden="true" className="text-center text-2xl text-slate-500">→</div>
                        <TransferPlayer player={transfer.buy} action="BUY" />
                      </div>
                      <div className="mt-3 border-t border-slate-800 pt-3">
                        {explanations[key] && <p className="text-sm leading-6 text-slate-300">{explanations[key]}</p>}
                        {explanationErrors[key] && <p role="alert" className="text-sm text-red-400">{explanationErrors[key]}</p>}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
            <p className="text-xs leading-5 text-slate-500">
              FPL public data does not show unpublished squad changes, exact remaining free transfers,
              or every original purchase price. Check affordability and your current team in FPL
              before making any transfer. Recommendations are not applied automatically.
            </p>
          </section>

          <section className="space-y-4">
            <div>
              <h2 className="text-2xl font-bold">Recommended starting XI</h2>
              <p className="text-sm text-slate-400">Select a gameweek to see the lineup and captain.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {advice.gameweeks.map((gw) => (
                <button
                  key={gw}
                  type="button"
                  onClick={() => setSelectedGw(gw)}
                  className={`rounded-lg px-4 py-2 text-sm font-semibold ${selectedGw === gw ? "bg-green-500 text-slate-950" : "border border-slate-700 bg-slate-900 text-slate-300"}`}
                >
                  GW {gw}
                </button>
              ))}
            </div>
            {lineup && (
              <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
                <p className="mb-4 text-sm text-slate-400">
                  Predicted starting XI: <strong className="text-white">{lineup.expectedPoints.toFixed(1)} xPts</strong>
                  {" "}(includes captain bonus)
                </p>
                <div className="space-y-5">
                  {positions.map((position) => (
                    <div key={position}>
                      <p className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">{position}</p>
                      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                        {lineup.starters.filter((player) => player.position === position).map((player) => (
                          <div key={player.id} className="rounded-lg border border-slate-800 bg-slate-950 px-3 py-3">
                            <div className="flex items-center justify-between gap-2">
                              <div>
                                <p className="font-medium">
                                  {player.name} {player.id === lineup.captainId && <span className="text-green-400">(C)</span>}
                                </p>
                                <p className="text-xs text-slate-500">{player.club} · £{player.price.toFixed(1)}m</p>
                              </div>
                              <span className="text-sm font-semibold text-green-400">{player.expectedPoints?.toFixed(1)} xPts</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>

          <p className="text-xs text-slate-500">
            AI explanations summarize model predictions and SHAP contributions, not guaranteed results.
          </p>
          <p className="text-xs text-slate-500">
            Predicted GW{advice.gameweeks[0]}–GW{advice.gameweeks[advice.gameweeks.length - 1]} ·
            Last calculated {new Date(advice.generatedAt).toLocaleString()}
          </p>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
      <p className="text-xs text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-bold">{value}</p>
    </div>
  );
}

function TransferPlayer({ player, action }: { player: Player; action: "SELL" | "BUY" }) {
  return (
    <div>
      <p className={`text-xs font-bold ${action === "BUY" ? "text-green-400" : "text-rose-400"}`}>
        {action}
      </p>
      <p className="mt-1 text-lg font-semibold">{player.name}</p>
      <p className="text-sm text-slate-400">
        {player.position} · {player.club} · £{player.price.toFixed(1)}m
      </p>
    </div>
  );
}
