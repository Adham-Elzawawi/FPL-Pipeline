"""Make team and opponent stats from earlier matches."""
from collections import defaultdict

import numpy as np
import pandas as pd

from paper_features import TEAM_METRICS, WINDOW


def build_team_history(raw_fixtures, understat_teams):
    """Get the stats for both teams in each finished match."""
    rows = []
    for match in raw_fixtures.itertuples():
        if not bool(match.finished) or pd.isna(match.event):
            continue
        if pd.isna(match.team_h_score) or pd.isna(match.team_a_score):
            continue
        rows.append({'fixture': match.id, 'round': int(match.event),
                     'team': int(match.team_h),
                     'kickoff_time': getattr(match, 'kickoff_time', None),
                     'goals_scored': match.team_h_score,
                     'goals_conceded': match.team_a_score})
        rows.append({'fixture': match.id, 'round': int(match.event),
                     'team': int(match.team_a),
                     'kickoff_time': getattr(match, 'kickoff_time', None),
                     'goals_scored': match.team_a_score,
                     'goals_conceded': match.team_h_score})
    if not rows:
        raise ValueError('No completed fixtures for team features')
    history = pd.DataFrame(rows)
    if not understat_teams.empty:
        history = history.merge(understat_teams, on=['fixture', 'team'],
                                how='left', validate='one_to_one')
    for stat in TEAM_METRICS:
        if stat not in history:
            history[stat] = np.nan
        history[stat] = pd.to_numeric(history[stat], errors='coerce')
    return history


def add_team_features(fixtures, team_history, completed_gws):
    """Average the last 5 matches for each team and opponent."""
    history = team_history.sort_values(['round', 'kickoff_time', 'fixture'])
    played = defaultdict(list)
    results = []
    for gw in sorted(fixtures['round'].dropna().unique()):
        week = fixtures[fixtures['round'] == gw].copy()
        for side, column in [('team', 'team'), ('opponent', 'opponent_team')]:
            # Get each club's numbers before this gameweek.
            snapshots = {}
            for club in week[column].dropna().unique():
                previous = played[int(club)]
                stats = {}
                for name in TEAM_METRICS:
                    values = [item[name] for item in previous[-WINDOW:]]
                    valid = [v for v in values if pd.notna(v)]
                    stats[f'{side}_{name}_{WINDOW}'] = (
                        float(np.mean(valid)) if valid else np.nan
                    )
                snapshots[int(club)] = stats
            if snapshots:
                means = pd.DataFrame.from_dict(snapshots, orient='index')
                means.index.name = column
                week = week.join(means, on=column)
        results.append(week)
        # Add this GW at the end, so doubles do not use each other.
        if int(gw) in completed_gws:
            for item in history[history['round'] == gw].to_dict('records'):
                played[int(item['team'])].append(item)
    return pd.concat(results, ignore_index=True)
