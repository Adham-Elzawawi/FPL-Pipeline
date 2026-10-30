"""Save manager optimizer results as JSON for the Next.js frontend.

This only formats the output from manager_advisor.run(). It never trains models.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from manager_advisor import POSITIONS, read_api
from shap_payload import explain_selected_players


OUTPUT_DIR = Path('outputs/manager_advice')


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temp, path)


def make_advice(manager_id, result):
    # Read player names and prices from the public FPL API.
    bootstrap = read_api('bootstrap-static/')
    players = {int(p['id']): p for p in bootstrap['elements']}
    clubs = {int(t['id']): t['short_name'] for t in bootstrap['teams']}

    current_squad = set(result['current']['squad'])
    new_squad = set(result['recommended']['squad'])
    gameweeks = [int(lineup['gw']) for lineup in result['recommended']['lineups']]
    manifest_path = Path('outputs/current_prediction.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    shap_by_player = explain_selected_players(
        current_squad | new_squad, gameweeks, manifest['strengths_last_gw']
    )

    def player_info(player_id, weeks=None):
        p = players[int(player_id)]
        if weeks is None:
            weeks = gameweeks
        return {
            'id': int(player_id),
            'name': p['web_name'],
            'position': POSITIONS[int(p['element_type'])],
            'club': clubs.get(int(p['team']), ''),
            'price': round(float(p['now_cost']) / 10, 1),
            'shap': {
                str(gw): shap_by_player[int(player_id)][str(gw)] for gw in weeks
            },
        }

    outgoing = sorted(current_squad - new_squad, key=lambda i: (
        players[int(i)]['element_type'], players[int(i)]['web_name']
    ))
    incoming = sorted(new_squad - current_squad, key=lambda i: (
        players[int(i)]['element_type'], players[int(i)]['web_name']
    ))

    transfers = []
    for position in POSITIONS:
        sells = [i for i in outgoing if int(players[int(i)]['element_type']) == position]
        buys = [i for i in incoming if int(players[int(i)]['element_type']) == position]
        for sell_id, buy_id in zip(sells, buys):
            transfers.append({
                'sell': player_info(sell_id),
                'buy': player_info(buy_id),
            })

    lineups = []
    for lineup in result['recommended']['lineups']:
        gw = int(lineup['gw'])
        starters = []
        for p in lineup['starters'].to_dict('records'):
            info = player_info(int(p['id']), weeks=[gw])
            info['expectedPoints'] = round(float(p[f'gw_{gw}_points']), 2)
            starters.append(info)
        lineups.append({
            'gameweek': gw,
            'expectedPoints': round(float(lineup['points']), 2),
            'captainId': int(lineup['captain']),
            'starters': starters,
        })

    current_points = float(result['current']['points'])
    recommended_points = float(result['recommended']['points'])

    return {
        'managerId': int(manager_id),
        'shapVersion': 1,
        'savedGw': int(result['saved_gw']),
        'gameweeks': [x['gameweek'] for x in lineups],
        'freeTransfers': int(result['free_transfers']),
        'transfersCount': int(result['transfers']),
        'currentPoints': round(current_points, 2),
        'recommendedPoints': round(recommended_points, 2),
        'gain': round(recommended_points - current_points, 2),
        'remainingBank': round(float(result['recommended']['bank']) / 10, 1),
        'transfers': transfers,
        'lineups': lineups,
        'predictionGeneratedAt': manifest['generated_at'],
        'generatedAt': datetime.now(timezone.utc).isoformat(),
    }


def save_manager_advice(manager_id, result):
    advice = make_advice(manager_id, result)
    destination = OUTPUT_DIR / f'manager_{int(manager_id)}.json'
    write_json(destination, advice)
    print(f'Saved manager advice in {destination}')
    return advice
