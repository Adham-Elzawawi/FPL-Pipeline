"""Get PPDA and other match stats from Understat.

FPL and Understat have different IDs, so I match teams and dates first.
Some names might not match and then the stats stay missing.
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests

CACHE = Path('data/understat_cache')
TEAM_STATS = ['xg', 'xga', 'deep', 'deep_allowed',
              'ppda_att', 'ppda_def', 'ppda_allowed_att', 'ppda_allowed_def']
PLAYER_STATS = ['shots', 'xg_chain', 'xg_buildup', 'key_passes']

ALIASES = {
    'man city': 'manchester city',
    'man utd': 'manchester united',
    'man united': 'manchester united',
    'manchester utd': 'manchester united',
    'wolves': 'wolverhampton wanderers',
    'wolverhampton': 'wolverhampton wanderers',
    'spurs': 'tottenham',
    'tottenham hotspur': 'tottenham',
    'newcastle': 'newcastle united',
    'west ham': 'west ham united',
    'brighton': 'brighton and hove albion',
    'brighton hove albion': 'brighton and hove albion',
    'leicester': 'leicester city',
    'norwich': 'norwich city',
    'ipswich': 'ipswich town',
    'nottingham forest': 'nottingham forest',
    'nott m forest': 'nottingham forest',
    'leeds': 'leeds united',
}


def clean_name(name):
    text = unicodedata.normalize('NFKD', str(name))
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'[^a-z0-9 ]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()


def team_name(name):
    name = clean_name(name)
    return ALIASES.get(name, name)


def get_json(path, refresh=False):
    """Get Understat data and save a local copy."""
    CACHE.mkdir(parents=True, exist_ok=True)
    filename = CACHE / (path.replace('/', '_') + '.json')
    if filename.exists() and not refresh:
        return json.loads(filename.read_text(encoding='utf-8'))
    url = 'https://understat.com/' + path
    response = requests.get(url, timeout=30, headers={
        'User-Agent': 'Mozilla/5.0 (FPL student project)',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': 'https://understat.com/',
    })
    response.raise_for_status()
    data = response.json()
    filename.write_text(json.dumps(data), encoding='utf-8')
    return data


def match_fixtures(fpl_fixtures, teams, understat_matches):
    """Match FPL games to Understat games using teams and dates."""
    team_names = dict(zip(teams['id'], teams['name']))
    matches = []
    for match in understat_matches:
        try:
            home = team_name(match['h']['title'])
            away = team_name(match['a']['title'])
            date = pd.to_datetime(match['datetime'], utc=True)
            matches.append((home, away, date, int(match['id'])))
        except (ValueError, KeyError, TypeError):
            continue

    mapping = {}
    for game in fpl_fixtures.itertuples():
        if pd.isna(game.kickoff_time):
            continue
        home = team_name(team_names.get(game.team_h, ''))
        away = team_name(team_names.get(game.team_a, ''))
        date = pd.to_datetime(game.kickoff_time, utc=True)
        candidates = [(abs((when - date).total_seconds()), mid)
                      for h, a, when, mid in matches if h == home and a == away]
        if candidates:
            delta, match_id = min(candidates)
            if delta <= 36 * 3600:
                mapping[int(game.id)] = match_id
    return mapping


def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


def get_understat_matches(fpl_fixtures, teams, season, refresh=False):
    """Get team and player stats for games that have been played.

    The season is its first year, like 2026 for 2026/27.
    """
    completed_count = int(fpl_fixtures['finished'].fillna(False).sum())
    if completed_count == 0:
        # At the start of the season we have no games yet.
        return (pd.DataFrame(columns=['fixture', 'team'] + TEAM_STATS),
                pd.DataFrame(columns=['fixture', 'team', 'understat_name'] + PLAYER_STATS))
    league = get_json(f'getLeagueData/EPL/{season}', refresh)
    fixture_ids = match_fixtures(fpl_fixtures, teams, league['dates'])
    print(f'Understat fixtures matched: {len(fixture_ids)}/{len(fpl_fixtures)} '
          f'({completed_count} completed)')
    team_rows = []
    player_rows = []

    # These are Understat team stats for each match.
    stats_lookup = {}
    for club in league['teams'].values():
        club_id = int(club['id'])
        for match in club['history']:
            date = pd.to_datetime(match['date'], utc=True)
            stats_lookup[(club_id, date.strftime('%Y-%m-%d'))] = match

    understat_ids = {int(m['id']): m for m in league['dates']}
    for i, (fid, mid) in enumerate(fixture_ids.items(), 1):
        fpl_game = fpl_fixtures.loc[fpl_fixtures['id'] == fid].iloc[0]
        if not bool(fpl_game.get('finished', False)):
            continue
        u_match = understat_ids[mid]
        date = pd.to_datetime(u_match['datetime'], utc=True).strftime('%Y-%m-%d')
        for side, team_id in [('h', int(fpl_game['team_h'])),
                              ('a', int(fpl_game['team_a']))]:
            u_team_id = int(u_match[side]['id'])
            value = stats_lookup.get((u_team_id, date))
            if value is None:
                continue
            ppda = value.get('ppda') or {}
            allowed = value.get('ppda_allowed') or {}
            team_rows.append({
                'fixture': fid, 'team': team_id,
                'xg': _number(value.get('xG')),
                'xga': _number(value.get('xGA')),
                'deep': _number(value.get('deep')),
                'deep_allowed': _number(value.get('deep_allowed')),
                'ppda_att': _number(ppda.get('att')),
                'ppda_def': _number(ppda.get('def')),
                'ppda_allowed_att': _number(allowed.get('att')),
                'ppda_allowed_def': _number(allowed.get('def')),
            })

        # Player stats need an extra request for each match.
        match_data = get_json(f'getMatchData/{mid}', refresh)
        for side, team_id in [('h', int(fpl_game['team_h'])),
                              ('a', int(fpl_game['team_a']))]:
            roster = match_data.get('rosters', {}).get(side, {})
            for player in roster.values():
                player_rows.append({
                    'fixture': fid, 'team': team_id,
                    'understat_name': clean_name(player.get('player', '')),
                    'shots': _number(player.get('shots')),
                    'xg_chain': _number(player.get('xGChain')),
                    'xg_buildup': _number(player.get('xGBuildup')),
                    'key_passes': _number(player.get('key_passes')),
                })
        if i % 50 == 0:
            print(f'Understat: read {i}/{len(fixture_ids)} matched fixtures')

    print(f'Understat team stat rows: {len(team_rows)}; '
          f'player roster rows: {len(player_rows)}')
    if not team_rows:
        raise ValueError('No Understat team histories could be matched to completed FPL fixtures. '
                         'Check the Understat season and team names.')
    return pd.DataFrame(team_rows), pd.DataFrame(player_rows,
        columns=['fixture', 'team', 'understat_name'] + PLAYER_STATS)


def attach_player_stats(history, players, understat_players):
    """Match Understat player stats to FPL players.

    Only unambiguous matches are accepted. Unmatched rows remain NaN.
    """
    history = history.copy()
    if understat_players.empty:
        for stat in PLAYER_STATS:
            history[stat] = float('nan')
        return history

    names = players[['id', 'first_name', 'second_name', 'web_name']].copy()
    names['full_name'] = (names['first_name'] + ' ' + names['second_name']).map(clean_name)
    names['short_name'] = names['web_name'].map(clean_name)
    history = history.merge(names[['id', 'full_name', 'short_name']], on='id', how='left')

    # Try matching full name, team and match first.
    lookup = understat_players.drop_duplicates(['fixture', 'team', 'understat_name'], keep=False)
    full = lookup.rename(columns={'understat_name': 'full_name'})
    result = history.merge(full, on=['fixture', 'team', 'full_name'], how='left',
                           validate='many_to_one')

    # Short names are only safe if there is one matching player.
    short = lookup.copy()
    short['short_name'] = short['understat_name'].str.split().str[-1]
    short = short.drop_duplicates(['fixture', 'team', 'short_name'], keep=False)
    short = short.drop(columns=['understat_name']).rename(
        columns={stat: stat + '_short' for stat in PLAYER_STATS})
    result = result.merge(short, on=['fixture', 'team', 'short_name'], how='left',
                          validate='many_to_one')
    for stat in PLAYER_STATS:
        result[stat] = result[stat].fillna(result[stat + '_short'])
    return result.drop(columns=['full_name', 'short_name'] +
                       [stat + '_short' for stat in PLAYER_STATS])
