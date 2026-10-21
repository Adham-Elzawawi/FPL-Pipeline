"""Get player stats and average the last 5 fixtures."""
import numpy as np
import pandas as pd

from api import get_player_history
from fixtures import next_gameweek
from pathlib import Path
from paper_features import ALL_PLAYER_METRICS, WINDOW, FEATURES_MAP
from understat_data import attach_player_stats

# Keep this for older code that imports FEATURES.
FEATURES = FEATURES_MAP[3]


def get_players(bootstrap, gw=None):
    """Get each player's playing chance for the gameweek we predict."""
    data = pd.DataFrame(bootstrap['elements'])
    chance_columns = ['chance_of_playing_this_round', 'chance_of_playing_next_round']
    for column in chance_columns:
        if column not in data:
            data[column] = np.nan

    columns = ['id', 'web_name', 'first_name', 'second_name', 'team',
               'element_type', 'now_cost'] + chance_columns
    players = data[columns].copy()

    events = bootstrap.get('events', [])
    current_gw = next((int(e['id']) for e in events if e.get('is_current')), None)
    next_gw = next_gameweek(events) if events else None
    if gw is None:
        gw = next_gw

    if gw == current_gw and current_gw is not None:
        column = 'chance_of_playing_this_round'
    elif gw == next_gw:
        column = 'chance_of_playing_next_round'
    else:
        column = None  # FPL does not provide a chance for distant gameweeks.

    if column is None:
        players['playing_chance'] = 100.0
    else:
        # A missing percentage usually means no doubt has been reported.
        players['playing_chance'] = pd.to_numeric(
            players[column], errors='coerce'
        ).fillna(100).clip(0, 100)
    return players


def load_history(players, fixtures, refresh=False, understat_players=None):
    """Download player games and add our fixture difficulty."""
    histories = []
    for number, player_id in enumerate(players['id'], 1):
        games = get_player_history(int(player_id), refresh)
        if games:
            part = pd.DataFrame(games)
            part['id'] = int(player_id)
            histories.append(part)
        if number % 100 == 0:
            print(f'Downloaded {number}/{len(players)} player histories')

    if not histories:
        raise ValueError('No player histories downloaded from FPL')

    history = pd.concat(histories, ignore_index=True)
    history = history.rename(columns={'total_points': 'points', 'value': 'now_cost'})
    history['was_home'] = history['was_home'].astype(int)
    history['round'] = history['round'].astype(int)
    history = history.merge(
        fixtures, on=['fixture', 'round', 'opponent_team', 'was_home'],
        how='left', validate='many_to_one'
    )
    history = history.merge(players[['id', 'element_type']], on='id', how='left')
    if understat_players is not None:
        history = attach_player_stats(history, players, understat_players)

    # Leave any stats we could not find as missing.
    for name in ALL_PLAYER_METRICS:
        if name not in history.columns:
            history[name] = np.nan
        history[name] = pd.to_numeric(history[name], errors='coerce')

    # Some Understat player names will not match, so keep missing values.
    history['player_availability'] = np.nan
    snapshots = Path('data/historical_availability.csv')
    if snapshots.exists():
        # We can optionally use old availability data if we saved it.
        available = pd.read_csv(snapshots)
        columns = ['id', 'fixture', 'player_availability']
        if not set(columns).issubset(available.columns):
            raise ValueError('historical_availability.csv needs id, fixture, player_availability')
        history = history.drop(columns=['player_availability']).merge(
            available[columns], on=['id', 'fixture'], how='left',
            validate='many_to_one'
        )
    return history


def _sorted(history):
    order = ['id', 'round']
    if 'kickoff_time' in history.columns:
        order.append('kickoff_time')
    order.append('fixture')
    return history.sort_values(order).copy()


def add_past_averages(history):
    """Use the last 5 fixtures, without using the game we predict."""
    games = _sorted(history)
    base = [stat for stat in ALL_PLAYER_METRICS if stat != 'relevant_points']
    for stat in base:
        if stat not in games:
            games[stat] = np.nan
        games[stat] = pd.to_numeric(games[stat], errors='coerce')

    values_to_add = {}
    for stat in base:
        shifted = games.groupby('id')[stat].shift(1)
        name = f'player_{stat}_{WINDOW}'
        values_to_add[name] = (shifted.groupby(games['id'])
                               .rolling(WINDOW, min_periods=1).mean()
                               .reset_index(level=0, drop=True))

    # Look at points from previous home/away games separately.
    venue_keys = [games['id'], games['was_home']]
    shifted_points = games.groupby(['id', 'was_home'])['points'].shift(1)
    name = f'player_relevant_points_{WINDOW}'
    values_to_add[name] = (shifted_points.groupby(venue_keys)
                           .rolling(WINDOW, min_periods=1).mean()
                           .reset_index(level=[0, 1], drop=True))

    games = pd.concat([games, pd.DataFrame(values_to_add, index=games.index)], axis=1)
    columns = [key for key in values_to_add if not key.startswith('player_relevant_points_')]
    relevant = [key for key in values_to_add if key.startswith('player_relevant_points_')]

    # Double GWs should use stats from before the GW started.
    first = games.drop_duplicates(['id', 'round'])[['id', 'round'] + columns]
    first_venue = games.drop_duplicates(['id', 'round', 'was_home'])[
        ['id', 'round', 'was_home'] + relevant
    ]
    games = games.drop(columns=columns + relevant)
    games = games.merge(first, on=['id', 'round'], how='left', validate='many_to_one')
    games = games.merge(first_venue, on=['id', 'round', 'was_home'], how='left',
                        validate='many_to_one')
    return games


def upcoming_players(players, history, fixtures, gw):
    """Make player rows for the next GW."""
    future = players.merge(fixtures[fixtures['round'] == gw], on='team', how='left')
    future = future[future['fixture'].notna()].copy()
    future['round'] = gw
    future['player_availability'] = future['playing_chance']
    future['_upcoming'] = True

    past = history.copy()
    past['_upcoming'] = False
    combined = pd.concat([past, future], ignore_index=True)
    # The next GW has no results yet, so only use past games.
    combined = add_past_averages(combined)
    upcoming = combined[combined['_upcoming']].drop(columns=['_upcoming']).copy()
    return upcoming
