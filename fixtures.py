"""Find the next gameweek and calculate simple fixture difficulty scores.

Difficulty is between 1 (easy) and 5 (hard). To avoid looking into the
future, each gameweek uses team results from earlier fully completed GWs.
"""

from pathlib import Path

import pandas as pd


def next_gameweek(events):
    """Choose the next FPL gameweek from bootstrap data."""
    for event in events:
        if event.get("is_next"):
            return int(event["id"])
    for event in events:
        if event.get("is_current"):
            return int(event["id"]) + 1
    for event in events:
        if not event.get("finished", False):
            return int(event["id"])
    raise ValueError("No upcoming gameweek found. Pass --gw manually.")


def completed_gameweeks(fixtures, target_gw):
    """Return gameweeks before target_gw where every fixture is finished."""
    games = fixtures.dropna(subset=["event"]).copy()
    games = games[games["event"] < target_gw]
    finished = (games["finished"].fillna(False)
                & games["team_h_score"].notna()
                & games["team_a_score"].notna())
    return set(games.assign(ok=finished).groupby("event")["ok"].all()
               .loc[lambda results: results].index.astype(int))

