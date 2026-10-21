"""Make the player ratings from the model predictions."""

import numpy as np
import pandas as pd

POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def make_player_ratings(predicted_matches, teams, gw):
    """Add predicted points across the GW and make the ratings."""
    games = predicted_matches.copy()

    # Add both fixture predictions if it is a double GW.
    sums = games.groupby("id")["predicted_points"].sum(min_count=1)
    counts = games.groupby("id")["fixture"].count()
    players = games.drop_duplicates("id").copy().set_index("id")
    players["expected_points"] = sums
    players["games"] = counts

    chance = pd.to_numeric(players["playing_chance"],
                           errors="coerce").fillna(100).clip(0, 100)
    minutes = pd.to_numeric(players["player_minutes_5"], errors="coerce").fillna(0).clip(lower=0)
    xg = pd.to_numeric(players["player_expected_goals_5"], errors="coerce").fillna(0)
    xa = pd.to_numeric(players["player_expected_assists_5"], errors="coerce").fillna(0)

    # Minutes security = usual minutes (as %) times FPL playing chance.
    players["playing_chance"] = chance
    base_security = (minutes / 90).clip(upper=1) * 100
    players["minutes_security"] = (base_security * chance / 100).round(1)

    # Predictions already use past minutes, so only apply chance to xPts.
    players["expected_points"] = players["expected_points"] * chance / 100
    players["xgi_per90"] = np.where(minutes > 0, (xg + xa) * 90 / minutes.clip(lower=1), 0)
    # Just a simple attacking score from 0 to 100.
    players["attacking_rating"] = (players["xgi_per90"].clip(0, 0.8) / 0.8 * 100).round(1)

    short_names = teams.set_index("id")["short_name"]
    players["team_name"] = players["team"].map(short_names)
    players["position"] = players["element_type"].map(POSITION_NAMES)
    players["price_m"] = players["now_cost"] / 10

    # Show opponents with H/A next to them.
    opponents = games.loc[games["fixture"].notna(),
                          ["id", "opponent_team", "was_home"]].copy()
    opponents["opponent"] = (opponents["opponent_team"].map(short_names)
                             + opponents["was_home"].map({1: " (H)", 0: " (A)"}))
    labels = opponents.groupby("id")["opponent"].agg(" / ".join)
    players["opponents"] = labels
    players["opponents"] = players["opponents"].fillna("-")
    players["gameweek"] = gw

    columns = ["gameweek", "web_name", "team_name", "position", "opponents",
               "price_m", "expected_points", "playing_chance", "minutes_security",
               "attacking_rating", "xgi_per90", "games", "now_cost", "team",
               "element_type", "chance_of_playing_this_round", "chance_of_playing_next_round"]
    result = players.reset_index()[["id"] + columns]
    result["expected_points"] = result["expected_points"].round(2)
    result["xgi_per90"] = result["xgi_per90"].round(3)
    return result.sort_values("expected_points", ascending=False, na_position="last")
