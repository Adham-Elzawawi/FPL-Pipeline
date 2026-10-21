"""Suggest FPL transfers using an existing four-gameweek predictions CSV.

No model training happens in this script. Run main.py first to create ratings.
"""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


# --- Change these settings to suit your account ---
MANAGER_ID = 1234567           # Your FPL ID (the number in /entry/1234567/)
MIN_GAIN = 4.0           # Minimum improvement over four GWs to recommend a move
TRANSFER_PENALTY = 1.0   # Extra improvement needed for each transfer (not FPL hits)
CANDIDATES_PER_POSITION = 60   # Set to 0 to consider every available player

API = 'https://fantasy.premierleague.com/api'
POSITIONS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
SQUAD_SIZES = {1: 2, 2: 5, 3: 5, 4: 3}


def read_api(path):
    response = requests.get(f'{API}/{path}', timeout=30,
                            headers={'User-Agent': 'FPL Manager Advisor (personal project)'})
    if response.status_code == 404:
        raise ValueError(f'No FPL data at {path} (404). Check the manager ID and GW.')
    response.raise_for_status()
    return response.json()


def read_predictions(path):
    ratings = pd.read_csv(path)
    weeks = []
    for name in ratings.columns:
        match = re.fullmatch(r'gw_(\d+)_points', name)
        if match:
            weeks.append(int(match.group(1)))
    weeks.sort()
    if not weeks or weeks != list(range(weeks[0], weeks[-1] + 1)):
        raise ValueError('Prediction CSV needs consecutive gw_N_points columns')
    if ratings['id'].duplicated().any():
        raise ValueError('Prediction CSV has duplicate player IDs')
    return ratings, weeks


def get_latest_public_team(manager_id, history, first_gw):
    """Read the newest saved squad (and skip temporary Free Hit squads)."""
    saved_weeks = sorted(int(row['event']) for row in history['current'])
    if not saved_weeks:
        raise ValueError('This manager has no saved gameweeks yet.')
    latest_gw = saved_weeks[-1]
    if latest_gw != first_gw - 1:
        raise ValueError(
            f'Latest saved squad is GW{latest_gw}, but predictions start GW{first_gw}. '
            'Generate new predictions for the next gameweek first.'
        )
    gw = latest_gw
    while gw >= 1:
        picks = read_api(f'entry/{manager_id}/event/{gw}/picks/')
        if picks.get('active_chip') != 'freehit':
            ids = [int(p['element']) for p in picks['picks']]
            if len(ids) != 15 or len(set(ids)) != 15:
                raise ValueError(f'GW{gw} did not have a normal 15-player squad')
            return ids, int(picks['entry_history']['bank']), gw
        print(f'GW{gw} was a Free Hit; using the prior permanent squad.')
        gw -= 1
    raise ValueError('No permanent saved squad found for this manager.')


def estimate_free_transfers(history, next_gw):
    """Rebuild rolled-over free transfers from publicly saved GW history.

    This is the allowance at the start of the upcoming GW, before any
    transfers the manager might already have made since the last deadline.
    """
    rows = {int(row['event']): row for row in history['current']}
    if not rows or max(rows) != next_gw - 1:
        raise ValueError('Cannot calculate transfers: saved GW history is incomplete.')
    first_gw = min(rows)
    if first_gw == next_gw:
        raise ValueError('The first GW has unlimited transfers, not a fixed allowance.')

    chip_weeks = {int(item['event']) for item in history.get('chips', [])
                  if item.get('name') in ('wildcard', 'freehit')}

    # After a manager's first saved GW, they start with one free transfer.
    free = 1
    for gw in range(first_gw + 1, next_gw):
        if gw not in rows:
            raise ValueError(f'Missing GW{gw} in manager history.')
        if gw in chip_weeks:
            # Wildcard/Free Hit preserve the existing bank of free transfers.
            continue
        used = int(rows[gw]['event_transfers'])
        free = min(5, max(0, free - used) + 1)

    return free


def selling_prices(squad, all_players, manager_id, source_gw, overrides=None):
    """Estimate FPL sale prices using known transfers and current market prices.

    For an initial player, the original purchase price is not public. Their
    estimated selling price is current price unless overridden.
    """
    bought = {}
    try:
        transfers = read_api(f'entry/{manager_id}/transfers/')
        transfers = sorted(transfers, key=lambda item: item.get('time', ''))
        for move in transfers:
            if int(move.get('event', 0)) <= source_gw:
                bought[int(move['element_in'])] = int(move['element_in_cost'])
    except (requests.RequestException, ValueError, KeyError):
        print('Could not read public transfer history; estimating all sale prices.')

    prices = all_players.set_index('id')['now_cost'].to_dict()
    result = {}
    estimated = []
    overrides = overrides or {}
    for player_id in squad:
        current = int(prices[player_id])
        if player_id in overrides:
            result[player_id] = int(overrides[player_id])
        elif player_id in bought:
            paid = bought[player_id]
            result[player_id] = (paid + (current - paid) // 2
                                 if current > paid else current)
        else:
            result[player_id] = current
            estimated.append(player_id)
    return result, estimated


def load_candidates(bootstrap, ratings, weeks, owned_ids, limit):
    """Join predictions with the latest prices, teams and player status."""
    columns = ['id', 'web_name', 'team', 'element_type', 'now_cost', 'status',
               'chance_of_playing_next_round']
    available_columns = [col for col in columns if col in bootstrap['elements'][0]]
    current = pd.DataFrame(bootstrap['elements'])[available_columns].copy()
    point_columns = [f'gw_{gw}_points' for gw in weeks]
    players = current.merge(ratings[['id'] + point_columns], on='id', how='left',
                            validate='one_to_one')
    players[point_columns] = players[point_columns].fillna(0).clip(lower=0)
    players['expected_points'] = players[point_columns].sum(axis=1)
    players['owned'] = players['id'].isin(owned_ids)
    if 'status' in players:
        # Existing injured players can be sold or kept, but don't suggest buying them.
        can_buy = players['status'].eq('a') | players['status'].isna()
        if 'chance_of_playing_next_round' in players:
            chance = pd.to_numeric(players['chance_of_playing_next_round'], errors='coerce')
            can_buy = can_buy | (players['status'].eq('d') & chance.ge(75))
        players = players[players['owned'] | can_buy].copy()

    if limit > 0:
        # Include good scorers, good value picks and cheaper options.
        selected = set(owned_ids)
        for position in POSITIONS:
            group = players[players['element_type'] == position]
            selected.update(group.nlargest(limit, 'expected_points')['id'])
            group = group.copy()
            group['value_score'] = group['expected_points'] / group['now_cost'].clip(lower=1)
            selected.update(group.nlargest(15, 'value_score')['id'])
            selected.update(group.nsmallest(10, 'now_cost')['id'])
        players = players[players['id'].isin(selected)].copy()

    if not set(owned_ids).issubset(set(players['id'])):
        raise ValueError('Some players in the squad were not found in the FPL data')
    return players.reset_index(drop=True)


def optimize(players, weeks, owned_ids, bank, sale_prices, number_transfers,
             time_limit=60):
    """MILP: choose a legal 15-player squad, XI and captain for each GW.

    Exactly number_transfers new players may enter. The solver maximizes
    expected starting XI + captain points, not bench points.
    """
    n = len(players)
    w_count = len(weeks)
    total_variables = n * (1 + 2 * w_count)
    ids = players['id'].astype(int).tolist()
    owned = np.array([player in owned_ids for player in ids], dtype=bool)
    types = players['element_type'].to_numpy()
    clubs = players['team'].to_numpy()

    # Cost of keeping an owned player is their forgone selling value.
    # A newly purchased player costs the current market price.
    costs = np.array([sale_prices[player] if is_owned else price
                      for player, is_owned, price in
                      zip(ids, owned, players['now_cost'])], dtype=float)
    funds = float(bank + sum(sale_prices[player] for player in owned_ids))

    objective = np.zeros(total_variables)
    weekly_points = np.array([players[f'gw_{gw}_points'].to_numpy(dtype=float)
                              for gw in weeks])
    for w in range(w_count):
        objective[n + w * n:n + (w + 1) * n] = -weekly_points[w]
        objective[n * (1 + w_count + w):n * (2 + w_count + w)] = -weekly_points[w]

    # Collect sparse constraints, so the model stays reasonably small.
    row_ids, col_ids, values, lower, upper = [], [], [], [], []

    def rule(terms, minimum, maximum):
        row = len(lower)
        for index, value in terms:
            row_ids.append(row)
            col_ids.append(index)
            values.append(value)
        lower.append(minimum)
        upper.append(maximum)

    for position, required in SQUAD_SIZES.items():
        rule([(i, 1) for i in range(n) if types[i] == position], required, required)
    for club in np.unique(clubs):
        rule([(i, 1) for i in range(n) if clubs[i] == club], 0, 3)
    rule([(i, costs[i]) for i in range(n)], -np.inf, funds)
    rule([(i, 1) for i in range(n) if not owned[i]], number_transfers, number_transfers)

    for w in range(w_count):
        start = n + w * n
        captain = n * (1 + w_count + w)
        rule([(start + i, 1) for i in range(n)], 11, 11)
        for position, low, high in [(1, 1, 1), (2, 3, 5), (3, 2, 5), (4, 1, 3)]:
            rule([(start + i, 1) for i in range(n) if types[i] == position], low, high)
        rule([(captain + i, 1) for i in range(n)], 1, 1)
        for i in range(n):
            rule([(start + i, 1), (i, -1)], -np.inf, 0)
            rule([(captain + i, 1), (start + i, -1)], -np.inf, 0)

    matrix = coo_matrix((values, (row_ids, col_ids)),
                        shape=(len(lower), total_variables)).tocsr()
    solution = milp(c=objective, integrality=np.ones(total_variables),
                    bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix, lower, upper),
                    options={'time_limit': time_limit, 'mip_rel_gap': 0.002})
    if solution.x is None:
        return None
    if solution.status != 0:
        print('Warning: solver stopped early; this is the best found solution, '
              'not a proven optimum.')

    chosen = solution.x[:n] > 0.5
    selected_ids = set(np.array(ids)[chosen].tolist())
    lineups = []
    total_points = 0.0
    for w, gw in enumerate(weeks):
        start = n + w * n
        captain_offset = n * (1 + w_count + w)
        starters = players.loc[solution.x[start:start + n] > 0.5].copy()
        captain_id = ids[int(np.argmax(solution.x[captain_offset:captain_offset + n]))]
        points = float(starters[f'gw_{gw}_points'].sum() +
                       starters.loc[starters['id'] == captain_id, f'gw_{gw}_points'].sum())
        total_points += points
        lineups.append({'gw': gw, 'starters': starters, 'captain': captain_id,
                        'points': points})
    return {'squad': selected_ids, 'lineups': lineups, 'points': total_points,
            'bank': int(round(funds - costs[chosen].sum())),
            'optimal': solution.status == 0}


def print_lineups(result, weeks):
    for lineup in result['lineups']:
        starter_names = lineup['starters'].set_index('id')['web_name'].to_dict()
        captain = starter_names[lineup['captain']]
        grouped = []
        for position in POSITIONS:
            names = lineup['starters'].loc[
                lineup['starters']['element_type'] == position, 'web_name'].tolist()
            grouped.append(f'{POSITIONS[position]}: {", ".join(names)}')
        print(f'\nGW{lineup["gw"]}: {lineup["points"]:.2f} xPts | Captain: {captain}')
        for group in grouped:
            print('  ' + group)


def save_results(manager_id, weeks, baseline, recommendation, players):
    path = Path('outputs')
    path.mkdir(exist_ok=True)
    prefix = f'manager_{manager_id}_gw_{weeks[0]}_to_{weeks[-1]}'
    rows = []
    for result_type, result in [('current', baseline), ('recommended', recommendation)]:
        for lineup in result['lineups']:
            for player in lineup['starters'].to_dict('records'):
                rows.append({'plan': result_type, 'gameweek': lineup['gw'],
                             'id': player['id'], 'player': player['web_name'],
                             'position': POSITIONS[int(player['element_type'])],
                             'captain': player['id'] == lineup['captain'],
                             'expected_points': player[f'gw_{lineup["gw"]}_points']})
    pd.DataFrame(rows).to_csv(path / f'{prefix}_lineups.csv', index=False)
    players[players['id'].isin(recommendation['squad'])].to_csv(
        path / f'{prefix}_squad.csv', index=False)
    print(f'\nSaved lineups and squad in {path.resolve()}')


def run(manager_id, min_gain=MIN_GAIN, penalty=TRANSFER_PENALTY, predictions=None,
        candidates=CANDIDATES_PER_POSITION, selling_file=None,
        my_team_json=None, bank_override=None):
    if manager_id <= 0 or min_gain < 0 or penalty < 0:
        raise ValueError('Use a valid manager ID and nonnegative gain settings')

    bootstrap = read_api('bootstrap-static/')
    next_gw = next((event['id'] for event in bootstrap['events'] if event.get('is_next')), None)
    if predictions is None:
        if next_gw is None:
            raise ValueError('No upcoming GW in FPL. Provide --predictions explicitly.')
        last_gw = min(next_gw + 3, 38)
        predictions = Path(f'outputs/gw_{next_gw}_to_{last_gw}_player_ratings.csv')
    predictions = Path(predictions)
    if not predictions.exists():
        raise FileNotFoundError(f'Missing {predictions}. First run python main.py --weeks 4')

    ratings, weeks = read_predictions(predictions)
    if next_gw is not None and weeks[0] != next_gw:
        print(f'WARNING: predictions start GW{weeks[0]}, but FPL says GW{next_gw} is next.')
    print(f'Reading saved predictions from {predictions} (GW{weeks[0]}-GW{weeks[-1]})')

    # Both the squad and transfer allowance come from the FPL API.
    history = read_api(f'entry/{manager_id}/history/')
    squad, bank, source_gw = get_latest_public_team(manager_id, history, weeks[0])
    free_transfers = estimate_free_transfers(history, weeks[0])
    exact_sales = {}
    print(f'Using the last saved permanent squad (GW{source_gw}).')
    print(f'Estimated free transfers for GW{weeks[0]}: {free_transfers} '
          '(calculated from public transfer and chip history).')
    print('NOTE: Moves made since the most recent deadline are not public.')

    if my_team_json:
        # An optional authenticated export provides an exact transfer balance.
        # Still use the latest SAVED squad, not the editable one.
        import json
        data = json.loads(Path(my_team_json).read_text(encoding='utf-8'))
        transfer_data = data['transfers']
        if transfer_data.get('limit') is not None:
            free_transfers = max(0, int(transfer_data['limit']) -
                                 int(transfer_data.get('made', 0)))
            print(f'Authenticated free transfers remaining: {free_transfers}')
        live_squad = {int(p['element']) for p in data['picks']}
        if live_squad == set(squad):
            bank = int(transfer_data['bank'])
            exact_sales = {int(p['element']): int(p['selling_price'])
                           for p in data['picks']}
            print('Authenticated bank and selling prices match the saved squad.')
        else:
            print('WARNING: Current editable squad differs from saved GW picks. '
                  'Keeping saved squad and ignoring private prices/bank.')
    if len(squad) != 15 or len(set(squad)) != 15:
        raise ValueError('Squad must contain 15 distinct players')
    if bank_override is not None:
        bank = int(round(bank_override * 10))
    all_players = pd.DataFrame(bootstrap['elements'])
    if selling_file:
        overrides = pd.read_csv(selling_file)
        if not {'id', 'selling_price'}.issubset(overrides.columns):
            raise ValueError('Selling CSV must have columns: id,selling_price (FPL 0.1m units)')
        exact_sales.update(dict(zip(overrides['id'].astype(int),
                                    overrides['selling_price'].astype(int))))
    sale_prices, estimated = selling_prices(squad, all_players, manager_id,
                                            source_gw, overrides=exact_sales)
    if estimated:
        print(f'WARNING: {len(estimated)} selling prices are estimates (original '
              'purchase prices unavailable). Check transfer affordability in FPL.')
    print(f'Bank: £{bank / 10:.1f}m | Free transfers to consider: {free_transfers}')

    players = load_candidates(bootstrap, ratings, weeks, set(squad), candidates)
    baseline = optimize(players, weeks, set(squad), bank, sale_prices, 0)
    if baseline is None:
        raise ValueError('Current squad could not be optimized. Check budget / player data.')
    print(f'\nCURRENT TEAM (best weekly lineups): {baseline["points"]:.2f} xPts')
    for lineup in baseline['lineups']:
        print(f'  GW{lineup["gw"]}: {lineup["points"]:.2f} xPts')

    best = baseline
    best_value = baseline['points']
    best_transfers = 0
    for transfer_count in range(1, min(free_transfers, 15) + 1):
        attempt = optimize(players, weeks, set(squad), bank, sale_prices, transfer_count)
        if attempt is None:
            print(f'{transfer_count} transfers: no legal, affordable plan found')
            continue
        gain = attempt['points'] - baseline['points']
        print(f'{transfer_count} transfer(s): {attempt["points"]:.2f} xPts '
              f'({gain:+.2f} xPts)')
        adjusted = attempt['points'] - penalty * transfer_count
        if adjusted > best_value + 1e-6:
            best, best_value, best_transfers = attempt, adjusted, transfer_count

    raw_gain = best['points'] - baseline['points']
    adjusted_gain = raw_gain - penalty * best_transfers
    if adjusted_gain < min_gain or best_transfers == 0:
        print(f'\nDECISION: KEEP YOUR TEAM. Best adjusted gain = {adjusted_gain:.2f} '
              f'xPts (required: {min_gain:.2f}).')
        chosen = baseline
    else:
        print(f'\nDECISION: MAKE {best_transfers} TRANSFER(S)! '
              f'Gain: {raw_gain:+.2f} xPts over {len(weeks)} GWs.')
        original = set(squad)
        outgoing = players[players['id'].isin(original - best['squad'])]
        incoming = players[players['id'].isin(best['squad'] - original)]
        for position in POSITIONS:
            out_players = outgoing[outgoing['element_type'] == position].sort_values('web_name')
            in_players = incoming[incoming['element_type'] == position].sort_values('web_name')
            for outgoing_row, incoming_row in zip(out_players.itertuples(),
                                                 in_players.itertuples()):
                print(f'  {POSITIONS[position]}: SELL {outgoing_row.web_name} '
                      f'(£{sale_prices[outgoing_row.id] / 10:.1f}m) -> '
                      f'BUY {incoming_row.web_name} (£{incoming_row.now_cost / 10:.1f}m)')
        print(f'Estimated remaining bank: £{best["bank"] / 10:.1f}m')
        chosen = best

    print('\nRECOMMENDED WEEKLY STARTING XIs (captain bonus included):')
    print_lineups(chosen, weeks)
    save_results(manager_id, weeks, baseline, chosen, players)
    return {'current': baseline, 'recommended': chosen, 'transfers':
            best_transfers if chosen is best and best_transfers else 0,
            'free_transfers': free_transfers, 'saved_gw': source_gw}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FPL manager transfer advisor (no training)')
    parser.add_argument('--id', type=int, default=MANAGER_ID, help='FPL entry / manager ID')
    parser.add_argument('--min-gain', type=float, default=MIN_GAIN)
    parser.add_argument('--transfer-penalty', type=float, default=TRANSFER_PENALTY)
    parser.add_argument('--predictions', help='Saved multi-GW ratings CSV')
    parser.add_argument('--candidates', type=int, default=CANDIDATES_PER_POSITION,
                        help='Top xPts players per position (0 = all)')
    parser.add_argument('--selling-prices', help='Optional CSV: id,selling_price (0.1m units)')
    parser.add_argument('--my-team-json', help='Optional authenticated JSON for exact transfer allowance/prices')
    parser.add_argument('--bank', type=float, help='Override bank in £m')
    args = parser.parse_args()
    if args.id <= 0:
        parser.error('Set MANAGER_ID in manager_advisor.py or pass --id YOUR_FPL_ID')
    run(args.id, args.min_gain, args.transfer_penalty,
        args.predictions, args.candidates, args.selling_prices,
        args.my_team_json, args.bank)
