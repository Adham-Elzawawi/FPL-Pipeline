"""Get last season's CSV files from GitHub."""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from understat_data import team_name

BASE = ('https://raw.githubusercontent.com/vaastav/'
        'Fantasy-Premier-League/master/data')
POSITIONS = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}


def season_label(start_year):
    return f'{start_year}-{str(start_year + 1)[-2:]}'


def read_season_csv(start_year, filename, refresh=False):
    """Save CSVs locally so we don't download them every time."""
    season = season_label(start_year)
    local = Path('data/vaastav') / season / filename
    if local.exists() and not refresh:
        return pd.read_csv(local)

    url = f'{BASE}/{season}/{filename}'
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(response.content)
    return pd.read_csv(StringIO(response.text))


def load_previous_season(start_year, refresh=False):
    """Turn the CSVs into fixtures, teams and player match rows.

    The old CSVs use team names, so we map them to last season's IDs.
    """
    teams = read_season_csv(start_year, 'teams.csv', refresh)
    fixtures = read_season_csv(start_year, 'fixtures.csv', refresh)
    games = read_season_csv(start_year, 'gws/merged_gw.csv', refresh)

    needed = {'element', 'team', 'fixture', 'round', 'opponent_team',
              'position', 'total_points', 'was_home'}
    if not needed.issubset(games.columns):
        raise ValueError(f'Missing previous-season columns: {sorted(needed - set(games.columns))}')

    club_ids = {team_name(name): int(i) for i, name in zip(teams['id'], teams['name'])}
    games['team'] = games['team'].map(lambda name: club_ids.get(team_name(name)))
    if games['team'].isna().any():
        missing = games.loc[games['team'].isna(), 'name'].head(5).tolist()
        raise ValueError(f'Could not map previous-season teams: {missing}')

    games = games.rename(columns={
        'element': 'id', 'total_points': 'points', 'value': 'now_cost',
    })
    games['element_type'] = games['position'].map(POSITIONS)
    if games['element_type'].isna().any():
        raise ValueError('Unknown historical player position')
    games['element_type'] = games['element_type'].astype(int)
    games['id'] = games['id'].astype(int)
    games['team'] = games['team'].astype(int)
    games['was_home'] = games['was_home'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})
    games['round'] = pd.to_numeric(games['round']).astype(int)
    games['fixture'] = pd.to_numeric(games['fixture']).astype(int)

    # We do not have old injury/availability snapshots, so leave them empty.
    games['player_availability'] = float('nan')
    fixtures['event'] = pd.to_numeric(fixtures['event'], errors='coerce')
    return fixtures, teams, games
