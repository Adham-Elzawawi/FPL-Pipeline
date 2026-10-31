"""Compare the old 5-match features with the new 3/5/8/10 features.

Run from the FPL Pipeline folder:
    python test/backtest.py --from-gw 4 --to-gw 8

The backtest makes each prediction before training on that gameweek.
It also picks a NEW 15-player squad every GW. That squad simulation is
NOT the same as playing a real FPL season with limited transfers.
"""

import argparse
import hashlib
import json
import sys
import pickle
import gzip
from pathlib import Path

# This script lives in test/, while the pipeline modules live one folder above.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"

import numpy as np
import pandas as pd
import scipy
import xgboost
from scipy.optimize import Bounds, LinearConstraint, milp
from xgboost import XGBRegressor

from api import get_bootstrap, get_fixtures
from fixtures import completed_gameweeks
from paper_features import FEATURES_MAP
from players import add_past_averages, get_players, load_history, upcoming_players
from previous_season import previous_season_training
from strengths import last_season_strengths, make_fixture_table
from team_features import add_team_features, build_team_history
from understat_data import get_understat_matches


MODELS = ['5', '3-5-8-10']
POSITIONS = {1: 2, 2: 5, 3: 5, 4: 3}


def get_feature_sets():
    """Both models use the same columns, except for their rolling windows."""
    all_windows = {p: list(FEATURES_MAP[p]) for p in POSITIONS}
    only_five = {}
    for position, columns in all_windows.items():
        only_five[position] = [
            name for name in columns
            if name.endswith('_5') or name in ('player_availability', 'att_fdr', 'def_fdr')
        ]
    return {'5': only_five, '3-5-8-10': all_windows}


def inputs(rows, columns, medians=None):
    values = rows.reindex(columns=columns).apply(pd.to_numeric, errors='coerce')
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).fillna(0), medians


def new_model(trees, learning_rate, jobs):
    return XGBRegressor(
        n_estimators=trees, max_depth=5, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
        tree_method='hist', random_state=42, n_jobs=jobs,
    )


def add_week(models, rows, feature_map, trees, jobs):
    """Update a model only AFTER its gameweek has been evaluated."""
    for position in POSITIONS:
        if position not in models:
            continue
        part = rows[rows['element_type'] == position]
        part = part[pd.to_numeric(part['points'], errors='coerce').notna()]
        if part.empty:
            continue
        columns = feature_map[position]
        bundle = models[position]
        x, _ = inputs(part, columns, bundle['medians'])
        updated = new_model(trees, 0.03, jobs)
        updated.fit(x, part['points'].astype(float), xgb_model=bundle['model'].get_booster())
        bundle['model'] = updated


def train_previous(training, feature_map, start_trees, week_trees, jobs):
    """Mirror the project's previous-season walk-forward tree training."""
    result = {}
    valid = training[pd.to_numeric(training['points'], errors='coerce').notna()].copy()
    for gw in sorted(valid['round'].dropna().unique()):
        week = valid[valid['round'] == gw]
        for position in POSITIONS:
            if position not in result:
                earlier = valid[(valid['round'] <= gw) & (valid['element_type'] == position)]
                if len(earlier) < 10:
                    continue
                x, medians = inputs(earlier, feature_map[position])
                model = new_model(start_trees, 0.05, jobs)
                model.fit(x, earlier['points'].astype(float))
                result[position] = {'model': model, 'medians': medians}
            else:
                add_week({position: result[position]}, week, feature_map, week_trees, jobs)

    if set(result) != set(POSITIONS):
        raise ValueError('Previous-season data did not cover all four positions.')

    # Previous-season data is completely known before the current season begins.
    for position in POSITIONS:
        old = valid[valid['element_type'] == position]
        _, result[position]['medians'] = inputs(old, feature_map[position])
    return result


def predict(rows, models, feature_map):
    result = rows.copy()
    result['predicted_points'] = np.nan
    for position in POSITIONS:
        mask = result['element_type'] == position
        if not mask.any():
            continue
        bundle = models[position]
        x, _ = inputs(result.loc[mask], feature_map[position], bundle['medians'])
        result.loc[mask, 'predicted_points'] = np.clip(bundle['model'].predict(x), 0, None)
    return result


def make_candidates(gw, history, fixture_table, bootstrap_players):
    """Use only pre-GW history for each player's team, price, and form.

    A historical squad snapshot would be more accurate, but is not currently
    available. A player without previous match history cannot enter this pool.
    """
    past = history[history['round'] < gw].copy()
    if past.empty:
        return pd.DataFrame()

    previous = (past.sort_values(['round', 'fixture'])
                .drop_duplicates('id', keep='last'))
    names = bootstrap_players[['id', 'web_name', 'first_name', 'second_name']]
    people = previous[['id', 'team', 'element_type', 'now_cost']].merge(
        names, on='id', how='left', validate='one_to_one'
    )
    people['playing_chance'] = 100.0  # Historical injury reports are unavailable.
    people['chance_of_playing_this_round'] = np.nan
    people['chance_of_playing_next_round'] = np.nan

    matches = upcoming_players(people, past, fixture_table, gw)
    if matches.empty:
        return pd.DataFrame()
    matches['player_availability'] = np.nan  # Do not use today's injury news for past GWs.

    # Keep one candidate row PER fixture. The caller predicts every fixture
    # and then sums expected points by player for double gameweeks.
    matches['fixture_count'] = matches.groupby('id')['fixture'].transform('count')
    return matches


def pick_squad(candidates, budget, time_limit=45, mip_gap=0.0):
    """Optimize a legal 15-player squad and a valid starting XI for one GW."""
    data = candidates.dropna(subset=['expected_points', 'now_cost']).copy()
    # Stable tie-breaking when two players have identical predictions.
    data = data.sort_values('id')
    data = data[(data['now_cost'] > 0) & (data['element_type'].isin(POSITIONS))]
    # Keep strong starters and inexpensive bench options to make MILP faster.
    selected = []
    for position in POSITIONS:
        subset = data[data['element_type'] == position]
        high = subset.nlargest(28, 'expected_points')
        cheap = subset.nsmallest(12, 'now_cost')
        selected.append(pd.concat([high, cheap]).drop_duplicates('id'))
    data = pd.concat(selected, ignore_index=True).drop_duplicates('id')
    data = data.sort_values('id').reset_index(drop=True)

    count = len(data)
    if any((data['element_type'] == p).sum() < number for p, number in POSITIONS.items()):
        raise ValueError('Not enough eligible players for a legal FPL squad.')

    points = data['expected_points'].to_numpy(dtype=float)
    price = data['now_cost'].to_numpy(dtype=float)
    # Three binary flags per player: squad, starting XI, captain.
    # Captain receives an extra copy of their predicted points.
    costs = np.zeros(count * 3)
    costs[:count] = -0.0001 * points   # Prefer useful substitutes on ties.
    costs[count:2 * count] = -points  # Starting XI scores.
    costs[2 * count:] = -points       # Captain bonus scores.
    rules, lower, upper = [], [], []

    def rule(squad=None, xi=None, low=-np.inf, high=np.inf):
        row = np.zeros(count * 3)
        if squad is not None:
            row[:count] = squad
        if xi is not None:
            row[count:2 * count] = xi
        rules.append(row)
        lower.append(low)
        upper.append(high)

    ones = np.ones(count)
    rule(squad=ones, low=15, high=15)
    rule(xi=ones, low=11, high=11)
    rule(squad=price, high=budget * 10)
    captain_row = np.zeros(count * 3)
    captain_row[2 * count:] = 1
    rules.append(captain_row)
    lower.append(1)
    upper.append(1)

    for position, squad_count in POSITIONS.items():
        mask = (data['element_type'] == position).to_numpy(dtype=float)
        rule(squad=mask, low=squad_count, high=squad_count)
        if position == 1:
            minimum, maximum = 1, 1
        elif position == 2:
            minimum, maximum = 3, 5
        elif position == 3:
            minimum, maximum = 3, 5
        else:
            minimum, maximum = 1, 3
        rule(xi=mask, low=minimum, high=maximum)

    for team in data['team'].dropna().unique():
        mask = (data['team'] == team).to_numpy(dtype=float)
        rule(squad=mask, high=3)

    for i in range(count):
        row = np.zeros(count * 3)
        row[i] = -1
        row[count + i] = 1
        rules.append(row)
        lower.append(-np.inf)
        upper.append(0)

        # The captain must be in the starting XI.
        captain_rule = np.zeros(count * 3)
        captain_rule[count + i] = -1
        captain_rule[2 * count + i] = 1
        rules.append(captain_rule)
        lower.append(-np.inf)
        upper.append(0)

    solution = milp(
        c=costs, integrality=np.ones(count * 3),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.array(rules), lower, upper),
        options={'time_limit': time_limit, 'mip_rel_gap': mip_gap},
    )
    # Do not silently accept a timed-out, approximate solution. It could
    # change across machines and produce misleading model comparisons.
    if solution.status != 0 or solution.x is None:
        raise ValueError(
            f'The squad optimizer did not finish optimally: {solution.message}. '
            'Try increasing --optimizer-time-limit.'
        )

    data['in_squad'] = solution.x[:count] > 0.5
    data['starting'] = solution.x[count:2 * count] > 0.5
    data['captain'] = solution.x[2 * count:] > 0.5
    result = data[data['in_squad']].copy()
    starters = result[result['starting']].sort_values(
        ['expected_points', 'id'], ascending=[False, True]
    )
    vice = starters[~starters['captain']].iloc[0]
    result['vice_captain'] = result['id'] == vice['id']
    result['bench_order'] = 0
    bench = result[~result['starting']]
    bench_gk = bench[bench['element_type'] == 1]
    bench_outfield = bench[bench['element_type'] != 1].sort_values(
        'expected_points', ascending=False
    )
    result.loc[bench_gk.index, 'bench_order'] = 4
    for order, idx in enumerate(bench_outfield.index, 1):
        result.loc[idx, 'bench_order'] = order
    return result


def score_squad(squad, actual):
    """Score the XI including ordinary automatic subs and vice captain.

    This deliberately excludes chips and transfer hits.
    """
    team = squad.copy()
    team['actual_points'] = team['id'].map(actual['points']).fillna(0).astype(float)
    team['actual_minutes'] = team['id'].map(actual['minutes']).fillna(0).astype(float)

    field = list(team[team['starting']].index)
    bench = team[~team['starting']].sort_values('bench_order')
    for idx, player in bench.iterrows():
        if player['actual_minutes'] <= 0:
            continue
        if player['element_type'] == 1:
            missing = [i for i in field if team.loc[i, 'element_type'] == 1
                       and team.loc[i, 'actual_minutes'] == 0]
        else:
            missing = [i for i in field if team.loc[i, 'element_type'] != 1
                       and team.loc[i, 'actual_minutes'] == 0]
        for old in missing:
            next_field = [i for i in field if i != old] + [idx]
            types = team.loc[next_field, 'element_type'].value_counts()
            if (types.get(1, 0) == 1 and 3 <= types.get(2, 0) <= 5
                    and 3 <= types.get(3, 0) <= 5 and 1 <= types.get(4, 0) <= 3):
                field = next_field
                break

    total = team.loc[field, 'actual_points'].sum()
    captain = team[team['captain']].iloc[0]
    vice = team[team['vice_captain']].iloc[0]
    if captain['actual_minutes'] > 0:
        total += captain['actual_points']
    elif vice['actual_minutes'] > 0:
        total += vice['actual_points']
    return float(total)


def score_audit(squad, actual):
    """Explain the actual player scores used by score_squad.

    Replays the automatic-sub rules and captain fallback exactly once.
    The final audit sum must match the reported squad score.
    """
    team = squad.copy()
    team['actual_points'] = team['id'].map(actual['points']).fillna(0).astype(float)
    team['actual_minutes'] = team['id'].map(actual['minutes']).fillna(0).astype(float)
    field = list(team[team['starting']].index)
    for idx, player in team[~team['starting']].sort_values('bench_order').iterrows():
        if player['actual_minutes'] <= 0:
            continue
        if player['element_type'] == 1:
            missing = [i for i in field if team.loc[i, 'element_type'] == 1
                       and team.loc[i, 'actual_minutes'] == 0]
        else:
            missing = [i for i in field if team.loc[i, 'element_type'] != 1
                       and team.loc[i, 'actual_minutes'] == 0]
        for old in missing:
            next_field = [i for i in field if i != old] + [idx]
            types = team.loc[next_field, 'element_type'].value_counts()
            if (types.get(1, 0) == 1 and 3 <= types.get(2, 0) <= 5
                    and 3 <= types.get(3, 0) <= 5 and 1 <= types.get(4, 0) <= 3):
                field = next_field
                break
    team['counted_in_xi'] = team.index.isin(field)
    team['captain_bonus'] = 0.0
    captain = team[team['captain']].iloc[0]
    vice = team[team['vice_captain']].iloc[0]
    if captain['actual_minutes'] > 0:
        team.loc[team['captain'], 'captain_bonus'] = captain['actual_points']
    elif vice['actual_minutes'] > 0:
        team.loc[team['vice_captain'], 'captain_bonus'] = vice['actual_points']
    team['counted_points'] = (team['actual_points'] * team['counted_in_xi']
                               + team['captain_bonus'])
    assert abs(team['counted_points'].sum() - score_squad(squad, actual)) < 1e-8
    return team[['id', 'web_name', 'actual_points', 'actual_minutes',
                 'starting', 'counted_in_xi', 'captain', 'vice_captain',
                 'captain_bonus', 'counted_points', 'expected_points']].copy()


def metrics(all_predictions):
    """Accuracy per GW for each model, using actual player-GW totals."""
    result = []
    for (model, gw), group in all_predictions.groupby(['model', 'gw']):
        actual = group['actual'].astype(float)
        predicted = group['predicted'].astype(float)
        squared_errors = (actual - predicted) ** 2
        result.append({
            'model': model,
            'gw': int(gw),
            'players': len(group),
            'mae': float(np.abs(actual - predicted).mean()),
            'mse': float(squared_errors.mean()),
            'sse': float(squared_errors.sum()),
            'rmse': float(np.sqrt(squared_errors.mean())),
            'spearman': actual.corr(predicted, method='spearman'),
        })
    return pd.DataFrame(result).sort_values(['gw', 'model']).reset_index(drop=True)


def position_metrics(predictions):
    """Report accuracy separately for goalkeepers, defenders, etc."""
    rows = []
    for (name, position), group in predictions.groupby(['model', 'element_type']):
        actual = group['actual'].astype(float)
        pred = group['predicted'].astype(float)
        err = actual - pred
        rows.append({
            'model': name, 'position': int(position), 'player_gameweeks': len(group),
            'mae': float(np.abs(err).mean()),
            'rmse': float(np.sqrt(np.mean(err ** 2))),
            'spearman': actual.corr(pred, method='spearman'),
        })
    return pd.DataFrame(rows)


def make_plot(week_scores, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    averages = week_scores.drop_duplicates('gw').sort_values('gw')
    ax.plot(averages['gw'], averages['fpl_manager_average'], marker='o',
            label='FPL manager average')
    for model in MODELS:
        part = week_scores[week_scores['model'] == model].sort_values('gw')
        ax.plot(part['gw'], part['actual_squad_points'], marker='o',
                label=f'Fresh squad: {model}')
    ax.set_xlabel('Gameweek')
    ax.set_ylabel('Actual FPL points')
    ax.set_title('Fresh weekly squads vs FPL manager average (not a like-for-like test)')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / 'backtest_vs_fpl_average.png', dpi=150)
    plt.close(fig)



def make_mse_plot(by_gw, output):
    """Plot mean squared prediction errors for both feature sets by GW."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for model in MODELS:
        week = by_gw[by_gw['model'] == model].sort_values('gw')
        ax.plot(week['gw'], week['mse'], marker='o', label=f'Model: {model}')
    ax.set_xlabel('Gameweek')
    ax.set_ylabel('MSE (squared FPL points per player)')
    ax.set_title('Player prediction MSE by gameweek (lower is better)')
    ax.set_xticks(sorted(by_gw['gw'].unique()))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / 'backtest_mse_by_gw.png', dpi=150)
    plt.close(fig)


def save_mse_results(predictions, output):
    """Write per-GW accuracy and an MSE/SSE chart from saved predictions."""
    by_gw = metrics(predictions)
    by_gw.to_csv(output / 'backtest_prediction_metrics.csv', index=False)
    by_gw[['gw', 'model', 'players', 'mse', 'sse']].to_csv(
        output / 'backtest_mse_by_gw.csv', index=False
    )
    make_mse_plot(by_gw, output)
    print('\nPer-gameweek MSE and total squared error (SSE):')
    print(by_gw[['gw', 'model', 'players', 'mse', 'sse']].to_string(
        index=False, float_format=lambda x: f'{x:.3f}'
    ))
    print('\nMSE = SSE / number of player-GW predictions.')
    print('SSE is the total squared error; MSE is the mean squared error.')
    for model in MODELS:
        sample = predictions[predictions['model'] == model]
        if not sample.empty:
            diff = sample['actual'].astype(float) - sample['predicted'].astype(float)
            print(f'{model} overall MSE: {(diff ** 2).mean():.3f} '
                  f'(total SSE: {(diff ** 2).sum():.3f})')
    return by_gw


def mse_from_saved_predictions(output):
    """Refresh just MSE outputs without re-training the models or picking squads."""
    path = output / 'backtest_player_predictions.csv'
    if not path.exists():
        raise FileNotFoundError(f'Run the backtest once first: {path}')
    predictions = pd.read_csv(path)
    required = {'model', 'gw', 'actual', 'predicted'}
    if not required.issubset(predictions.columns):
        raise ValueError('Player predictions CSV is missing required columns.')
    if predictions.empty:
        raise ValueError('Player predictions CSV is empty.')
    output.mkdir(parents=True, exist_ok=True)
    save_mse_results(predictions, output)
    print(f'\nMSE files saved in {output.resolve()}')


def hash_file(path):
    """SHA256 of a cached input or code file."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(args):
    """Record data and versions so a backtest can be reproduced.

    Only cache/input directories are included, not trained production models.
    """
    files = []
    data_root = PROJECT_ROOT / 'data'
    for folder in ('fpl_cache', 'understat_cache', 'vaastav'):
        target = data_root / folder
        if target.exists():
            files.extend(p for p in target.rglob('*') if p.is_file())
    for name in ('historical_availability.csv', 'team_strengths.csv'):
        target = data_root / name
        if target.exists():
            files.append(target)
    for name in ('api.py', 'players.py', 'paper_features.py',
                 'team_features.py', 'strengths.py', 'fixtures.py',
                 'previous_season.py', 'historical_data.py',
                 'understat_data.py'):
        target = PROJECT_ROOT / name
        if target.exists():
            files.append(target)
    files.append(Path(__file__).resolve())

    file_hashes = {str(p.relative_to(PROJECT_ROOT)): hash_file(p)
                   for p in sorted(set(files))}
    full_hash = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True).encode('utf-8')
    ).hexdigest()
    return {
        'input_sha256': full_hash,
        'source_file_count': len(file_hashes),
        'source_hashes': file_hashes,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scipy': scipy.__version__,
        'xgboost': xgboost.__version__,
        'from_gw': args.from_gw,
        'to_gw': args.to_gw,
        'start_trees': args.start_trees,
        'old_week_trees': args.old_week_trees,
        'new_week_trees': args.new_week_trees,
        'jobs': args.jobs,
        'budget': args.budget,
        'optimizer_gap': args.optimizer_gap,
        'optimizer_time_limit': args.optimizer_time_limit,
    }


def compare_manifest(output, current):
    old_path = output / 'backtest_run_manifest.json'
    if not old_path.exists():
        return
    previous = json.loads(old_path.read_text(encoding='utf-8'))
    keys = ('input_sha256', 'python', 'numpy', 'pandas', 'scipy', 'xgboost',
            'from_gw', 'to_gw', 'start_trees', 'old_week_trees',
            'new_week_trees', 'jobs', 'budget', 'optimizer_gap',
            'optimizer_time_limit')
    changed = [key for key in keys if previous.get(key) != current.get(key)]
    if changed:
        print('WARNING: This run is not directly reproducible from the previous one.')
        print('Changed:', ', '.join(changed))
        if 'input_sha256' in changed:
            old_files = previous.get('source_hashes', {})
            new_files = current.get('source_hashes', {})
            differences = [p for p in sorted(set(old_files) | set(new_files))
                           if old_files.get(p) != new_files.get(p)]
            print('Changed data/code files (first 12):', ', '.join(differences[:12]))
    else:
        print('Input files, libraries and settings match the previous run.')


def save_data_snapshot(path, old, history, training, fixture_table, players,
                       finished, official, weeks):
    """Save exactly the prepared input tables used by the backtest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = (old, history, training, fixture_table, players,
              finished, official, weeks)
    with gzip.open(path, 'wb', compresslevel=3) as output:
        pickle.dump(tables, output, protocol=pickle.HIGHEST_PROTOCOL)


def load_data_snapshot(path):
    """Only load snapshots that YOU generated locally (pickle is not safe for strangers)."""
    if not path.is_file():
        raise FileNotFoundError(f'Frozen input data not found: {path}')
    with gzip.open(path, 'rb') as input_file:
        return pickle.load(input_file)


def frozen_fingerprint(args, snapshot_path):
    """A frozen-data run must report the same hash when replayed."""
    info = input_fingerprint(args)
    snap_hash = hash_file(snapshot_path)
    code_hash = hash_file(Path(__file__).resolve())
    info['input_sha256'] = hashlib.sha256(
        f'{snap_hash}:{code_hash}'.encode('utf-8')
    ).hexdigest()
    info['source_hashes'] = {
        str(snapshot_path): snap_hash,
        'test/backtest.py': code_hash,
    }
    info['source_file_count'] = 2
    return info


def main(args):
    feature_sets = get_feature_sets()
    print('Feature counts:')
    for name, features in feature_sets.items():
        print(name, {position: len(columns) for position, columns in features.items()})

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / 'backtest_data_snapshot.pkl.gz'

    if args.use_frozen_data:
        print('Loading FROZEN prepared data, not live caches...')
        (old, history, training, fixture_table, players,
         finished, official, weeks) = load_data_snapshot(snapshot_path)
        # The snapshot is the source of truth for this run.
        run_info = frozen_fingerprint(args, snapshot_path)
    else:
        print('\nLoading season data...')
        bootstrap = get_bootstrap(refresh=args.refresh)
        players = get_players(bootstrap)
        teams = pd.DataFrame(bootstrap['teams'])
        raw = pd.DataFrame(get_fixtures(args.refresh))
        season = pd.to_datetime(bootstrap['events'][0]['deadline_time']).year
        previous_year = season - 1
        finished = completed_gameweeks(raw, 39)
        official = {
            int(event['id']): float(event['average_entry_score'])
            for event in bootstrap['events']
            if event.get('finished') and event.get('average_entry_score') is not None
        }
        completed = sorted(finished.intersection(official))
        if not completed:
            raise ValueError('No completed gameweeks with official averages are available.')
        end_gw = args.to_gw or max(completed)
        weeks = [gw for gw in completed if args.from_gw <= gw <= end_gw]
        if not weeks:
            raise ValueError('No completed GWs in the requested range.')

        old, old_fixtures, old_teams = previous_season_training(previous_year, args.refresh)
        base_strengths = last_season_strengths(old_fixtures, old_teams, teams)
        fixture_table = make_fixture_table(raw, teams, max(completed) + 1,
                                           initial_strengths=base_strengths)
        understat_team, understat_player = get_understat_matches(
            raw, teams, season, args.refresh
        )
        team_hist = build_team_history(raw, understat_team)
        fixture_table = add_team_features(fixture_table, team_hist, finished)
        history = load_history(players, fixture_table, args.refresh, understat_player)
        history = history[history['round'].isin(finished)].copy()
        training = add_past_averages(history)

        run_info = input_fingerprint(args)
        if args.freeze_data:
            save_data_snapshot(snapshot_path, old, history, training, fixture_table,
                               players, finished, official, weeks)
            print('Saved frozen data snapshot:', snapshot_path)
            run_info = frozen_fingerprint(args, snapshot_path)

    if not weeks:
        raise ValueError('Frozen snapshot contains no test gameweeks.')
    if (args.from_gw != min(weeks) or
            (args.to_gw is not None and args.to_gw != max(weeks))):
        raise ValueError('The frozen data gameweek range differs from the CLI range.')
    print(f'Testing GW {weeks[0]} through GW {weeks[-1]}.')
    compare_manifest(output, run_info)
    print('Input SHA256:', run_info['input_sha256'][:16])

    # Never let an interrupted run leave an OLD plot masquerading as new results.
    results = ('backtest_vs_fpl_average.png', 'backtest_vs_fpl_average.csv',
               'backtest_prediction_metrics.csv', 'backtest_position_metrics.csv',
               'backtest_player_predictions.csv', 'backtest_squads.csv',
               'backtest_scoring_audit.csv',
               'backtest_mse_by_gw.csv', 'backtest_mse_by_gw.png')
    for name in results:
        (output / name).unlink(missing_ok=True)
    run_info['complete'] = False
    (output / 'backtest_run_manifest.json').write_text(
        json.dumps(run_info, indent=2, sort_keys=True), encoding='utf-8'
    )

    print('\nTraining both variants on previous-season historical data...')
    models = {}
    for name in MODELS:
        print('Training:', name)
        models[name] = train_previous(
            old, feature_sets[name], args.start_trees, args.old_week_trees, args.jobs
        )

    all_predictions, weeks_scores, picks, audits = [], [], [], []
    # Train on GWs 1..N in order. Every tested GW is predicted BEFORE updating.
    for gw in range(1, max(weeks) + 1):
        week_games = training[training['round'] == gw].copy()
        if gw in weeks:
            print(f'Predicting GW {gw}...')
            current_actual = history[history['round'] == gw].groupby('id').agg(
                points=('points', 'sum'), minutes=('minutes', 'sum')
            )
            candidates = make_candidates(gw, history, fixture_table, players)
            for name in MODELS:
                predicted_games = predict(week_games, models[name], feature_sets[name])
                grouped = predicted_games.groupby(['id', 'element_type']).agg(
                    actual=('points', 'sum'), predicted=('predicted_points', 'sum')
                ).reset_index()
                grouped['gw'] = gw
                grouped['model'] = name
                all_predictions.append(grouped)

                if candidates.empty:
                    raise RuntimeError(f'No historical candidate players for GW {gw}')
                candidate_pred = predict(candidates, models[name], feature_sets[name])
                candidate_points = candidate_pred.groupby('id')['predicted_points'].sum()
                candidate_players = candidate_pred.drop_duplicates('id').copy()
                candidate_players['expected_points'] = candidate_players['id'].map(
                    candidate_points
                )
                try:
                    squad = pick_squad(candidate_players, args.budget,
                                      args.optimizer_time_limit, args.optimizer_gap)
                except ValueError as error:
                    raise RuntimeError(
                        f'{name} GW{gw} could not produce an optimal squad: {error}'
                    ) from error
                points = score_squad(squad, current_actual)
                audit = score_audit(squad, current_actual)
                audit['model'] = name
                audit['gw'] = gw
                audits.append(audit)
                weeks_scores.append({
                    'model': name, 'gw': gw,
                    'actual_squad_points': points,
                    'fpl_manager_average': official[gw],
                    'difference_vs_average': points - official[gw],
                    'predicted_squad_xpts': round(
                        squad.loc[squad['starting'], 'expected_points'].sum()
                        + squad.loc[squad['captain'], 'expected_points'].sum(), 2
                    ),
                    'squad_cost': round(squad['now_cost'].sum() / 10, 2),
                })
                squad = squad[['id', 'web_name', 'element_type', 'team', 'now_cost',
                               'expected_points', 'starting', 'bench_order',
                               'captain', 'vice_captain']].copy()
                squad['model'] = name
                squad['gw'] = gw
                picks.append(squad)

        if gw in finished and not week_games.empty:
            for name in MODELS:
                add_week(models[name], week_games, feature_sets[name],
                         args.new_week_trees, args.jobs)

    if len(weeks_scores) != len(weeks) * len(MODELS):
        raise RuntimeError('Some weekly squads are missing; results are incomplete.')
    individual = pd.concat(all_predictions, ignore_index=True)
    individual.to_csv(output / 'backtest_player_predictions.csv', index=False)
    save_mse_results(individual, output)
    position_metrics(individual).to_csv(output / 'backtest_position_metrics.csv', index=False)
    week_scores = pd.DataFrame(weeks_scores)
    if not week_scores.empty:
        week_scores.to_csv(output / 'backtest_vs_fpl_average.csv', index=False)
        pd.concat(picks, ignore_index=True).to_csv(output / 'backtest_squads.csv', index=False)
        pd.concat(audits, ignore_index=True).to_csv(
            output / 'backtest_scoring_audit.csv', index=False
        )
        make_plot(week_scores, output)

    run_info['complete'] = True
    (output / 'backtest_run_manifest.json').write_text(
        json.dumps(run_info, indent=2, sort_keys=True), encoding='utf-8'
    )

    print('\nOut-of-sample player-GW prediction accuracy:')
    for name in MODELS:
        sample = individual[individual['model'] == name]
        err = sample['actual'] - sample['predicted']
        mae = np.abs(err).mean()
        rmse = np.sqrt((err ** 2).mean())
        corr = sample['actual'].corr(sample['predicted'], method='spearman')
        print(f'{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, Spearman={corr:.3f}, rows={len(sample)}')
    if not week_scores.empty:
        print('\nActual weekly squad scores (new optimized squad every GW):')
        print(week_scores[['model', 'gw', 'actual_squad_points',
                           'fpl_manager_average', 'difference_vs_average']].to_string(index=False))
    print(f'\nResults saved in {output.resolve()}')
    print('NOTE: Weekly squad rebuilding assumes unlimited free transfers.')
    print('FPL averages include real managers, chips, transfers and auto-subs.')
    print('Historical injuries/transfers/precise prices are only approximated.')
    print('For point-by-point reconciliation see backtest_scoring_audit.csv.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare 5 vs 3/5/8/10 XGBoost windows')
    parser.add_argument('--from-gw', type=int, default=4)
    parser.add_argument('--to-gw', type=int, default=None)
    parser.add_argument('--budget', type=float, default=100.0,
                        help='Budget for the 15-player squad in millions')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--mse-only', action='store_true',
                        help='Recalculate MSE and its plot from saved player predictions')
    parser.add_argument('--freeze-data', action='store_true',
                        help='Write a frozen prepared-data snapshot in output/')
    parser.add_argument('--use-frozen-data', action='store_true',
                        help='Reuse frozen prepared-data instead of live/cache inputs')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Single-threaded training improves reproducibility')
    parser.add_argument('--optimizer-time-limit', type=float, default=45,
                        help='Seconds to find an optimal weekly squad')
    parser.add_argument('--optimizer-gap', type=float, default=0.0,
                        help='MILP optimality gap; 0 requests exact optimization')
    parser.add_argument('--start-trees', type=int, default=30)
    parser.add_argument('--old-week-trees', type=int, default=10)
    parser.add_argument('--new-week-trees', type=int, default=10)
    args = parser.parse_args()
    if args.freeze_data and args.use_frozen_data:
        parser.error('Choose --freeze-data OR --use-frozen-data, not both.')
    if args.refresh and args.use_frozen_data:
        parser.error('--refresh is incompatible with --use-frozen-data.')
    if args.budget <= 0 or args.jobs < 1 or args.optimizer_time_limit <= 0:
        parser.error('Budget, jobs and optimizer time limit must be positive.')
    if not 0 <= args.optimizer_gap < 1:
        parser.error('Optimizer gap must be in [0, 1).')
    if args.mse_only:
        mse_from_saved_predictions(Path(args.output))
    else:
        main(args)
