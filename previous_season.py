"""Prepare last season's matches for training."""
import numpy as np
import pandas as pd

from fixtures import completed_gameweeks
from historical_data import load_previous_season
from paper_features import ALL_PLAYER_METRICS
from players import add_past_averages
from strengths import make_fixture_table, starting_strengths
from team_features import build_team_history, add_team_features
from understat_data import get_understat_matches, attach_player_stats


def previous_season_training(start_year, refresh=False):
    """Get the old season and calculate all our features."""
    raw_fixtures, teams, games = load_previous_season(start_year, refresh)
    completed = completed_gameweeks(raw_fixtures, 99)
    if not completed:
        raise ValueError('No fully completed gameweeks in previous-season data')

    fixture_table = make_fixture_table(
        raw_fixtures, teams, target_gw=99,
        initial_strengths=starting_strengths(teams, csv_file=None),
    )
    old_understat, old_rosters = get_understat_matches(
        raw_fixtures, teams, start_year, refresh
    )
    team_history = build_team_history(raw_fixtures, old_understat)
    fixture_table = add_team_features(fixture_table, team_history, completed)

    # Match using the team AND fixture, not just one of them.
    history = games.merge(
        fixture_table,
        on=['fixture', 'round', 'team', 'opponent_team', 'was_home'],
        how='left', validate='many_to_one',
    )
    if history['att_fdr'].isna().all():
        raise ValueError('No previous-season games matched dynamic fixture data')

    # Use names from last season when matching Understat players.
    people = games[['id', 'name']].drop_duplicates('id').copy()
    people['first_name'] = people['name'].str.rsplit(' ', n=1).str[0]
    people['second_name'] = people['name'].str.rsplit(' ', n=1).str[-1]
    people['web_name'] = people['second_name']
    history = attach_player_stats(history, people, old_rosters)

    for stat in ALL_PLAYER_METRICS:
        if stat not in history:
            history[stat] = np.nan
        history[stat] = pd.to_numeric(history[stat], errors='coerce')

    history = history[history['round'].isin(completed)].copy()
    training = add_past_averages(history)
    print(f'Previous-season player-fixture rows: {len(training)}')
    return training, raw_fixtures, teams
