from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_matrix(fixtures, teams, start_gw, next_n, value_col):
    end_gw = start_gw + next_n - 1
    future = fixtures[(fixtures["round"] >= start_gw) & (fixtures["round"] <= end_gw)].copy()

    team_names = teams.set_index("id")["name"].to_dict()
    short_names = teams.set_index("id")["short_name"].to_dict()

    future["team_name"] = future["team"].map(team_names)
    future["opponent_name"] = future["opponent_team"].map(short_names)
    future["home_away"] = future["was_home"].map({1: "(H)", 0: "(A)"})
    future["cell_label"] = (
        future["opponent_name"]
        + " "
        + future["home_away"]
        + "\n"
        + future[value_col].round(2).astype(str)
    )

    # A double gameweek has two fixtures in the same cell.
    # Average their difficulty for the color, and show both names.
    values = future.pivot_table(index="team_name", columns="round",
                                values=value_col, aggfunc="mean")
    labels = (future.groupby(["team_name", "round"])["cell_label"]
              .agg(" / ".join).unstack())

    values = values.sort_index()
    labels = labels.reindex(values.index)

    return values, labels, future


def _team_axis_labels(future, metric):
    if metric == "att":
        cols = ["team", "team_name", "team_att_strength"]
        rating_col = "team_att_strength"
        short = "A"
    else:
        cols = ["team", "team_name", "team_def_strength"]
        rating_col = "team_def_strength"
        short = "D"

    table = future[cols].drop_duplicates("team").sort_values("team_name")
    table["axis_label"] = table["team_name"] + " (" + short + ":" + table[rating_col].round(2).astype(str) + ")"

    return dict(zip(table["team_name"], table["axis_label"]))


def _draw_heatmap(values, labels, axis_labels, title, save_path):
    if values.empty:
        print(f"No data to plot for {title}")
        return

    fig_width = max(8, len(values.columns) * 2.1)
    fig_height = max(6, len(values.index) * 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    matrix = values.to_numpy(dtype=float)

    valid = matrix[~np.isnan(matrix)]
    if len(valid) == 0:
        print(f"No valid values to plot for {title}")
        return

    # Dynamic scale so the differences look clearer
    vmin = float(valid.min())
    vmax = float(valid.max())
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    image = ax.imshow(matrix, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(np.arange(len(values.columns)))
    ax.set_xticklabels([f"GW{int(col)}" for col in values.columns])

    ax.set_yticks(np.arange(len(values.index)))
    ax.set_yticklabels([axis_labels.get(name, name) for name in values.index])

    # grid lines
    ax.set_xticks(np.arange(-0.5, len(values.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(values.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iloc[i, j]
            text = labels.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, str(text), ha="center", va="center", fontsize=8, color="black")

    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("Fixture Difficulty (raw)")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {save_path}")


def save_fixture_plots(fixtures, teams, start_gw, next_n=4, output_dir="plots"):
    output_dir = Path(output_dir)

    att_values, att_labels, att_future = _prepare_matrix(
        fixtures, teams, start_gw, next_n, "att_fdr_raw"
    )
    att_axis = _team_axis_labels(att_future, "att")
    _draw_heatmap(
        att_values,
        att_labels,
        att_axis,
        f"Next {next_n} Gameweeks — ATT_FDR (raw)",
        output_dir / f"next_{next_n}_att_fdr.png",
    )

    def_values, def_labels, def_future = _prepare_matrix(
        fixtures, teams, start_gw, next_n, "def_fdr_raw"
    )
    def_axis = _team_axis_labels(def_future, "def")
    _draw_heatmap(
        def_values,
        def_labels,
        def_axis,
        f"Next {next_n} Gameweeks — DEF_FDR (raw)",
        output_dir / f"next_{next_n}_def_fdr.png",
    )