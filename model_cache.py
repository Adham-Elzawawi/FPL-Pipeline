"""Store last season's trained models so we only build them once."""
import json
from pathlib import Path

import pandas as pd

from paper_features import FEATURES_MAP
from model import (train_previous_walk_forward, add_gameweek_trees,
                   save_model_checkpoint, load_model_checkpoint)
from strengths import last_season_strengths, starting_strengths
from understat_data import team_name


MODEL_FOLDER = Path('data/models')


def checkpoint_matches_features(folder):
    """Check whether a saved model uses our current feature columns."""
    file = Path(folder) / 'metadata.json'
    if not file.exists():
        return False
    try:
        metadata = json.loads(file.read_text())
    except (OSError, ValueError):
        return False
    expected = {str(position): features
                for position, features in FEATURES_MAP.items()}
    return metadata.get('version') == 1 and metadata.get('features') == expected


def train_previous_once(year, load_previous_data, refresh=False,
                        rebuild=False, model_folder=MODEL_FOLDER):
    """Run previous season once. Next calls load the models from disk."""
    folder = Path(model_folder) / f'previous_{year}'
    stats_path = folder / 'team_strengths.csv'

    saved_ok = checkpoint_matches_features(folder)
    if saved_ok and stats_path.exists() and not rebuild:
        print(f'Loading previous-season model from {folder}')
        models, _ = load_model_checkpoint(folder)
        strengths = pd.read_csv(stats_path)
        return models, strengths

    if (folder / 'metadata.json').exists() and not saved_ok:
        print('Old previous-season model uses different features. Rebuilding.')

    print(f'Preparing and training {year}/{year + 1} (one-time step)...')
    training, fixtures, teams = load_previous_data(year, refresh)
    models = train_previous_walk_forward(training)

    # These are the ratings after the final completed GW of the old season.
    final_ratings = last_season_strengths(fixtures, teams, teams)
    strengths = teams[['id', 'name']].copy()
    strengths = strengths.merge(final_ratings.reset_index().rename(
        columns={'team_id': 'id'}), on='id', how='left')

    save_model_checkpoint(models, folder, year,
                          int(training['round'].max()))
    strengths.to_csv(stats_path, index=False)
    print(f'Saved previous-season model to {folder}')
    return models, strengths


def starting_from_previous(saved_strengths, current_teams):
    """Move old strengths by team NAME; promoted clubs start at 3/3."""
    base = starting_strengths(current_teams, csv_file=None)
    old = {team_name(row['name']): row
           for row in saved_strengths.to_dict('records')}
    for team in current_teams.to_dict('records'):
        previous = old.get(team_name(team['name']))
        if previous is not None:
            base.loc[int(team['id']), ['att_score', 'def_score']] = [
                previous['att_score'], previous['def_score']
            ]
    return base


def train_current_incrementally(old_models, current_training, completed_gws,
                                season, previous_year, rebuild=False,
                                model_folder=MODEL_FOLDER):
    """Only train on completed GWs that have not been processed yet."""
    folder = Path(model_folder) / f'current_{season}'
    checkpoint = folder / 'metadata.json'

    saved_ok = checkpoint_matches_features(folder)
    if checkpoint.exists() and saved_ok and not rebuild:
        models, info = load_model_checkpoint(folder)
        if info['season'] != season or info['previous_season'] != previous_year:
            raise ValueError('Current-model checkpoint belongs to another season')
        last_gw = int(info['last_gw'])
        if last_gw > max(completed_gws, default=0):
            raise ValueError('Saved model is ahead of the completed GWs. '
                             'Run with --rebuild-models.')
        print(f'Loaded current-season checkpoint after GW{last_gw}')
    else:
        if checkpoint.exists() and not saved_ok:
            print('Old current-season model uses different features. Rebuilding.')
        models = old_models
        last_gw = 0
        rebuild = True

    for gw in sorted(completed_gws):
        if gw <= last_gw:
            continue
        rows = current_training[current_training['round'] == gw]
        add_gameweek_trees(models, rows)
        print(f'Updated current-season models with GW{gw} ({len(rows)} rows)')
        last_gw = int(gw)

    if not checkpoint.exists() or rebuild or last_gw != int(info['last_gw']):
        save_model_checkpoint(models, folder, season, last_gw,
                              previous_season=previous_year)
    return models
