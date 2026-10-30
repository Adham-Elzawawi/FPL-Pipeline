"""The two separate jobs used by the scheduler.

GW job: update strengths and add XGBoost trees once a gameweek is official.
CP job: use saved models, refresh live inputs, and solve teams (no training).
Run scripts from the project root so existing data/ and outputs/ paths work.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from api import get_bootstrap, get_fixtures
from fixtures import next_gameweek
from model import predict_matches
from model import load_model_checkpoint
from model_cache import (starting_from_previous, train_current_incrementally,
                         train_previous_once)
from players import get_players, load_history, upcoming_players, add_past_averages
from previous_season import previous_season_training
from ratings import make_player_ratings
from strengths import make_fixture_table, _updated_strengths
from team_features import add_team_features, build_team_history
from team_selection import choose_xi
from understat_data import get_understat_matches
from plot_fixtures import save_fixture_plots
from shap_payload import save_shap_inputs


SNAPSHOT_DIR = Path('data/runtime')
MODELS_DIR = Path('data/models')
OUTPUT_DIR = Path('outputs')


def season_year(bootstrap):
    """FPL season is named after the calendar year when GW1 starts."""
    return datetime.fromisoformat(
        bootstrap['events'][0]['deadline_time'].replace('Z', '+00:00')
    ).year


def confirmed_gameweeks(bootstrap, raw_fixtures):
    """Only accept consecutive gameweeks FPL has finalized.

    The fixtures also need final scores. A partially played double GW is not
    complete, even if some of its matches already have results.
    """
    games = raw_fixtures.dropna(subset=['event'])
    confirmed = []
    for event in sorted(bootstrap['events'], key=lambda e: int(e['id'])):
        gw = int(event['id'])
        if not event.get('finished', False):
            break
        if 'data_checked' in event and not event['data_checked']:
            break
        week_games = games[games['event'].astype(int) == gw]
        if week_games.empty:
            break
        complete = (week_games['finished'].fillna(False)
                    & week_games['team_h_score'].notna()
                    & week_games['team_a_score'].notna()).all()
        if not complete:
            break
        confirmed.append(gw)
    return confirmed


def _only_confirmed_results(raw_fixtures, completed):
    """Hide in-progress matches from strength and training calculations."""
    games = raw_fixtures.copy()
    in_completed = games['event'].isin(completed)
    games.loc[~in_completed, 'finished'] = False
    return games


def _save_pickle(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + '.tmp.pkl.gz')
    frame.to_pickle(temp)
    os.replace(temp, path)


def _save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.csv')
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temp, path)


def read_snapshot():
    filename = SNAPSHOT_DIR / 'snapshot.json'
    if not filename.exists():
        return None
    return json.loads(filename.read_text(encoding='utf-8'))


def update_gw_job(bootstrap=None, raw_fixtures=None, completed=None, on_progress=None):
    """Expensive work: rebuild the feature snapshot and update XGBoost.

    Call only at initialization or after another official GW was completed.
    """
    print('\n=== Updating trained models and team strengths ===', flush=True)
    progress = on_progress or (lambda message: None)
    if bootstrap is None:
        bootstrap = get_bootstrap(refresh=True)
    if raw_fixtures is None:
        raw_fixtures = pd.DataFrame(get_fixtures(refresh=True))
    if completed is None:
        completed = confirmed_gameweeks(bootstrap, raw_fixtures)

    season = season_year(bootstrap)
    try:
        gw = next_gameweek(bootstrap['events'])
    except ValueError:
        gw = 39  # The last GW can still need training after the season ends.
    gw = max(gw, max(completed, default=0) + 1)
    teams = pd.DataFrame(bootstrap['teams'])
    players = get_players(bootstrap, gw)
    games = _only_confirmed_results(raw_fixtures, completed)

    progress('Loading saved models and historical data')
    # Load previous-season models once; later GW updates use their checkpoint.
    previous_models, old_strengths = train_previous_once(
        season - 1, previous_season_training, refresh=False,
        model_folder=MODELS_DIR,
    )
    progress('Updating team strengths and fixture difficulty')
    base_strengths = starting_from_previous(old_strengths, teams)
    current_strengths = _updated_strengths(games, base_strengths, set(completed))

    fixtures = make_fixture_table(
        games, teams, gw, initial_strengths=base_strengths
    )
    understat_teams, understat_players = get_understat_matches(
        games, teams, season, refresh=bool(completed)
    )

    progress('Building player and team training features')
    if completed:
        team_history = build_team_history(games, understat_teams)
        fixtures = add_team_features(fixtures, team_history, set(completed))
        history = load_history(players, fixtures, refresh=True,
                               understat_players=understat_players)
        history = history[history['round'].isin(completed)].copy()
        current_training = add_past_averages(history)
    else:
        team_history = pd.DataFrame(columns=['round', 'fixture', 'team',
                                             'kickoff_time'])
        history = pd.DataFrame(columns=['id', 'round', 'fixture', 'team',
                                        'element_type', 'was_home', 'points'])
        current_training = pd.DataFrame(columns=['round', 'element_type',
                                                 'points'])

    progress('Training the updated xP models')
    # No other job calls train_current_incrementally.
    train_current_incrementally(
        previous_models, current_training, completed, season, season - 1,
        model_folder=MODELS_DIR,
    )
    progress('Saving models and strengths')
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    _save_pickle(current_strengths, SNAPSHOT_DIR / 'current_strengths.pkl.gz')
    _save_pickle(team_history, SNAPSHOT_DIR / 'team_history.pkl.gz')
    _save_pickle(history, SNAPSHOT_DIR / 'player_history.pkl.gz')
    _save_json({
        'season': season,
        'last_gw': max(completed, default=0),
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }, SNAPSHOT_DIR / 'snapshot.json')
    print(f'Updated snapshot through GW{max(completed, default=0)}')


def refresh_cp_job(bootstrap=None, raw_fixtures=None, budget=81.5):
    """Fresh predictions + CP/MILP, without touching model training or strengths.

    Load the last approved feature snapshot; refresh the fixture schedule and
    player availability before generating xPts for the next four gameweeks.
    """
    print('\n=== Refreshing predictions and optimizing teams ===', flush=True)
    snapshot = read_snapshot()
    if snapshot is None:
        raise RuntimeError('No model snapshot found. Run the GW update first.')

    if bootstrap is None:
        bootstrap = get_bootstrap(refresh=True)
    if raw_fixtures is None:
        raw_fixtures = pd.DataFrame(get_fixtures(refresh=True))
    season = season_year(bootstrap)
    if snapshot['season'] != season:
        raise RuntimeError('Snapshot is from another season. Run the GW update.')

    current_folder = MODELS_DIR / f'current_{season}'
    models, model_info = load_model_checkpoint(current_folder)
    if int(model_info['last_gw']) != int(snapshot['last_gw']):
        raise RuntimeError('Model and strengths snapshots do not match. Run GW update.')

    try:
        gw = next_gameweek(bootstrap['events'])
    except ValueError:
        print('No future deadline. No CP refresh needed.')
        return None
    if gw > 38:
        print('Season finished. No CP refresh needed.')
        return None

    teams = pd.DataFrame(bootstrap['teams'])
    players = get_players(bootstrap, gw)
    current_strengths = pd.read_pickle(SNAPSHOT_DIR / 'current_strengths.pkl.gz')
    team_history = pd.read_pickle(SNAPSHOT_DIR / 'team_history.pkl.gz')
    history = pd.read_pickle(SNAPSHOT_DIR / 'player_history.pkl.gz')

    # Build the latest fixture matchups with FROZEN saved strengths.
    # Even if a fixture gets rescheduled, this job never updates strengths.
    completed = list(range(1, int(snapshot['last_gw']) + 1))
    games = raw_fixtures.copy()
    games['finished'] = False
    fixtures = make_fixture_table(games, teams, gw,
                                  initial_strengths=current_strengths)
    fixtures = add_team_features(fixtures, team_history, set(completed))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    four_fixtures = fixtures[fixtures['round'].between(gw, gw + 3)].copy()
    four_fixtures['team_name'] = four_fixtures['team'].map(
        teams.set_index('id')['name'])
    four_fixtures['opponent_name'] = four_fixtures['opponent_team'].map(
        teams.set_index('id')['short_name'])
    _save_csv(four_fixtures, OUTPUT_DIR / 'next_4_fixtures.csv')
    _save_csv(fixtures[fixtures['round'] == gw],
              OUTPUT_DIR / f'gw_{gw}_fixtures.csv')

    try:
        save_fixture_plots(fixtures, teams, start_gw=gw, next_n=4,
                           output_dir='fpl-web/public/fixtures')
    except (ValueError, KeyError) as error:
        print(f'Could not refresh fixture plot: {error}')

    ratings_by_gw = []
    for future_gw in range(gw, min(gw + 4, 39)):
        # FPL only reports a playing-chance percentage for the current/next
        # GW. Avoid applying this GW's injury percentage to all four weeks.
        future_players = get_players(bootstrap, future_gw)
        upcoming = upcoming_players(future_players, history, fixtures, future_gw)
        if upcoming.empty:
            # A blank GW still needs a zero-point column for the advisor.
            print(f'GW{future_gw} has no fixtures; xPts will be zero.')
            points = pd.DataFrame({'id': players['id'],
                                   f'gw_{future_gw}_points': 0.0})
            # Keep an empty SHAP input file for a blank gameweek.
            save_shap_inputs(upcoming, future_gw)
            ratings_by_gw.append((future_gw, None, points))
            continue
        predicted = predict_matches(upcoming, models)
        save_shap_inputs(predicted, future_gw)
        ratings = make_player_ratings(predicted, teams, future_gw)
        _save_csv(ratings, OUTPUT_DIR / f'gw_{future_gw}_player_ratings.csv')
        points = ratings[['id', 'expected_points']].rename(columns={
            'expected_points': f'gw_{future_gw}_points'
        })
        ratings_by_gw.append((future_gw, ratings, points))

    if not ratings_by_gw:
        raise ValueError('No gameweeks left to predict')

    # This is the wide CSV that manager_advisor.py already expects.
    combined = players[['id']].drop_duplicates().copy()
    for _, _, points in ratings_by_gw:
        combined = combined.merge(points, on='id', how='left', validate='one_to_one')
    point_columns = [column for column in combined if column.endswith('_points')]
    combined[point_columns] = combined[point_columns].fillna(0)
    end_gw = ratings_by_gw[-1][0]
    prediction_path = OUTPUT_DIR / f'gw_{gw}_to_{end_gw}_player_ratings.csv'
    _save_csv(combined, prediction_path)

    # Keep the existing frontend's optimal_xi_1gw.csv, etc. files.
    valid = [(week, rating) for week, rating, _ in ratings_by_gw
             if rating is not None]
    for horizon in range(1, len(ratings_by_gw) + 1):
        period = [(week, rating) for week, rating in valid if week < gw + horizon]
        if not period:
            continue
        all_ratings = pd.concat([rating for _, rating in period], ignore_index=True)
        candidates = all_ratings.sort_values('gameweek').drop_duplicates('id').copy()
        totals = all_ratings.groupby('id')['expected_points'].sum(min_count=1)
        candidates['expected_points'] = candidates['id'].map(totals)
        try:
            xi = choose_xi(candidates, budget)
        except ValueError as error:
            print(f'Optimal XI for {horizon} GWs failed: {error}')
            continue
        _save_csv(xi, OUTPUT_DIR / f'optimal_xi_{horizon}gw.csv')
        if horizon == 1:
            _save_csv(xi, OUTPUT_DIR / f'gw_{gw}_starting_xi.csv')

    # The frontend uses this pointer so it never displays last month's GW.
    _save_json({
        'gameweek': gw,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'strengths_last_gw': snapshot['last_gw'],
    }, OUTPUT_DIR / 'current_prediction.json')
    print(f'New CP input: {prediction_path}')
    return prediction_path
