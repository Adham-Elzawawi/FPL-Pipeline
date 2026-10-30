type Props = {
  state: "checking" | "training" | "predicting" | "ready" | "error" | "idle";
  message: string;
};

const steps = [
  { title: "Checking gameweek", subtitle: "Looking for new completed matches" },
  { title: "Updating models", subtitle: "Refreshing strengths and training xP" },
  { title: "Generating predictions", subtitle: "Calculating points for the next fixtures" },
];

export default function ModelTrainingScreen({ state, message }: Props) {
  const currentStep = state === "predicting" || state === "ready" ? 2 : state === "training" ? 1 : 0;
  const hasError = state === "error";

  return (
    <div className="flex min-h-[70vh] items-center justify-center px-4 py-10">
      <div className="w-full max-w-xl rounded-3xl border border-slate-800 bg-slate-900 p-7 shadow-2xl sm:p-10">
        <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-green-500/10">
          {hasError ? (
            <span className="text-3xl text-rose-400" aria-hidden="true">!</span>
          ) : (
            <svg className="h-9 w-9 animate-spin text-green-400" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.18" strokeWidth="3" />
              <path d="M12 3a9 9 0 0 1 9 9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
            </svg>
          )}
        </div>

        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-green-400">
          FPL Predictor
        </p>
        <h1 className="text-3xl font-bold tracking-tight text-white">
          {hasError ? "The update failed" : "Preparing your predictions"}
        </h1>
        <p className="mt-3 text-sm leading-6 text-slate-400" role="status" aria-live="polite">
          {message}
        </p>

        {hasError && (
          <a href="/my-team" className="mt-7 inline-block rounded-lg bg-green-500 px-5 py-3 text-sm font-semibold text-slate-950 hover:bg-green-400">
            Back to My Team
          </a>
        )}

        {!hasError && (
          <>
            <div className="mt-8 h-1.5 overflow-hidden rounded-full bg-slate-800">
              <div className="h-full w-2/5 animate-pulse rounded-full bg-green-400" />
            </div>
            <div className="mt-8 space-y-5">
              {steps.map((step, index) => (
                <div key={step.title} className="flex items-start gap-4">
                  <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border text-sm font-semibold ${
                    index < currentStep
                      ? "border-green-500 bg-green-500 text-slate-950"
                      : index === currentStep
                        ? "border-green-400 text-green-400"
                        : "border-slate-700 text-slate-500"
                  }`}>
                    {index < currentStep ? "✓" : index + 1}
                  </div>
                  <div>
                    <p className={index <= currentStep ? "font-semibold text-white" : "font-semibold text-slate-500"}>
                      {step.title}
                    </p>
                    <p className="mt-0.5 text-sm text-slate-500">{step.subtitle}</p>
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-9 text-xs leading-5 text-slate-500">
              The results will load automatically when the backend is finished.
              Training happens only when a new gameweek has been finalized.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
