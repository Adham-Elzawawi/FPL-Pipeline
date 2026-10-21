"""Small helper to get FPL data. I cache the API responses locally."""
import json
from pathlib import Path

import requests

CACHE = Path('data/fpl_cache')
BASE = 'https://fantasy.premierleague.com/api'


def _get(path, refresh=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    filename = CACHE / (path.strip('/').replace('/', '_') + '.json')
    if filename.exists() and not refresh:
        return json.loads(filename.read_text(encoding='utf-8'))
    response = requests.get(f'{BASE}/{path.lstrip("/")}', timeout=30)
    response.raise_for_status()
    data = response.json()
    filename.write_text(json.dumps(data), encoding='utf-8')
    return data


def get_bootstrap(refresh=False):
    return _get('bootstrap-static/', refresh)


def get_fixtures(refresh=False):
    return _get('fixtures/', refresh)


def get_player_history(player_id, refresh=False):
    return _get(f'element-summary/{int(player_id)}/', refresh).get('history', [])
