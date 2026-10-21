"""Pick an 11-player starting team that follows FPL rules."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def choose_xi(ratings, budget=81.5):
    """Pick 11 players within budget and the position rules."""
    players = ratings.dropna(subset=["expected_points", "now_cost"]).copy()
    players = players[players["playing_chance"].fillna(100) >= 80]
    players = players.reset_index(drop=True)
    if len(players) < 11:
        raise ValueError("Not enough available players for an XI")

    count = len(players)
    costs = players["now_cost"].to_numpy(dtype=float)  # API costs: 10 = £1m
    points = players["expected_points"].to_numpy(dtype=float)

    rules = [np.ones(count), costs]
    min_values = [11, -np.inf]
    max_values = [11, budget * 10]

    # These are the allowed starting XI positions.
    for position, minimum, maximum in [(1, 1, 1), (2, 3, 5),
                                        (3, 3, 5), (4, 1, 3)]:
        rules.append((players["element_type"] == position).to_numpy(dtype=float))
        min_values.append(minimum)
        max_values.append(maximum)

    for club in players["team"].unique():
        rules.append((players["team"] == club).to_numpy(dtype=float))
        min_values.append(0)
        max_values.append(3)

    result = milp(
        c=-points,  # minimize negative points = maximize points
        integrality=np.ones(count),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.vstack(rules), min_values, max_values),
        options={"time_limit": 30},
    )
    if result.x is None or result.status != 0:
        raise ValueError("No legal starting XI found for this budget")

    xi = players.loc[result.x > 0.5].copy().sort_values("expected_points", ascending=False)
    xi["captain"] = ""
    xi.iloc[0, xi.columns.get_loc("captain")] = "C"
    xi.iloc[1, xi.columns.get_loc("captain")] = "VC"
    return xi
