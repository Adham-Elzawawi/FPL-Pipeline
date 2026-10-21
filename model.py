"""Train the FPL models with XGBoost, using the last season."""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from paper_features import FEATURES_MAP

# Start with last season, then add a few trees using this season.
BASE_TREES = 400
CURRENT_TREES = 100


def _inputs(rows, features, medians=None):
    """Get the feature columns in the right order and fill missing values."""
    x = rows.reindex(columns=features).apply(pd.to_numeric, errors='coerce')
    if medians is None:
        medians = x.median().fillna(0)
    return x.fillna(medians), medians


def _make_model(trees, learning_rate):
    """Use the same settings for each position."""
    return XGBRegressor(
        n_estimators=trees,
        max_depth=5,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
    )


def train_models(last_season, current_season=None):
    """Train on last season, then continue with this season's matches."""
    models = {}
    if current_season is None:
        current_season = pd.DataFrame(columns=last_season.columns)

    for position in [1, 2, 3, 4]:
        old = last_season[last_season['element_type'] == position].copy()
        new = current_season[current_season['element_type'] == position].copy()
        old = old[pd.to_numeric(old['points'], errors='coerce').notna()]
        new = new[pd.to_numeric(new['points'], errors='coerce').notna()]

        if len(old) < 10 and len(new) < 10:
            print(f'Not enough matches for position {position}, skipping')
            continue

        features = FEATURES_MAP[position]
        base = old if len(old) >= 10 else new
        x_base, medians = _inputs(base, features)
        y_base = pd.to_numeric(base['points'])

        # First train on last season's matches.
        model = _make_model(BASE_TREES, 0.05)
        model.fit(x_base, y_base)

        if len(old) >= 10 and len(new) >= 10:
            # xgb_model keeps the old trees and adds more using this season.
            x_new, _ = _inputs(new, features, medians)
            updated = _make_model(CURRENT_TREES, 0.03)
            updated.fit(x_new, pd.to_numeric(new['points']),
                        xgb_model=model.get_booster())
            model = updated
            print(f'Position {position}: {len(old)} old matches and '
                  f'{len(new)} current matches, '
                  f'{model.get_booster().num_boosted_rounds()} trees')
        else:
            print(f'Position {position}: only trained on {len(base)} matches')

        models[position] = {'model': model, 'medians': medians}

    if not models:
        raise ValueError('Not enough player matches to train any model')
    return models


def predict_matches(upcoming, models):
    """Predict each player's points for each scheduled match."""
    result = upcoming.copy()
    result['predicted_points'] = np.nan

    for position, bundle in models.items():
        mask = (result['element_type'] == position) & result['fixture'].notna()
        if mask.any():
            features = FEATURES_MAP[position]
            x, _ = _inputs(result.loc[mask], features, bundle['medians'])
            predictions = bundle['model'].predict(x)
            result.loc[mask, 'predicted_points'] = np.clip(predictions, 0, None)

    return result

# Walk-forward training used for the saved season models.
# Keep train_models() above for the existing backtest and tests.
OLD_START_TREES = 30
OLD_WEEK_TREES = 10
NEW_WEEK_TREES = 10
MIN_START_ROWS = 10


def add_gameweek_trees(models, week_rows, trees=NEW_WEEK_TREES):
    """Add trees using results from ONE completed gameweek."""
    for position, bundle in models.items():
        rows = week_rows[week_rows['element_type'] == position].copy()
        rows = rows[pd.to_numeric(rows['points'], errors='coerce').notna()]
        if rows.empty:
            continue

        features = FEATURES_MAP[position]
        x, _ = _inputs(rows, features, bundle['medians'])
        updated = _make_model(trees, 0.03)
        updated.fit(x, pd.to_numeric(rows['points']),
                    xgb_model=bundle['model'].get_booster())
        bundle['model'] = updated
    return models


def train_previous_walk_forward(training):
    """Process previous-season games in GW order, building trees over time."""
    models = {}
    valid = training[pd.to_numeric(training['points'], errors='coerce').notna()].copy()

    for gw in sorted(valid['round'].dropna().unique()):
        week = valid[valid['round'] == gw]
        for position in [1, 2, 3, 4]:
            if position not in models:
                # Wait for enough rows before starting a position's model.
                earlier = valid[(valid['round'] <= gw) &
                                (valid['element_type'] == position)]
                if len(earlier) < MIN_START_ROWS:
                    continue
                x, medians = _inputs(earlier, FEATURES_MAP[position])
                model = _make_model(OLD_START_TREES, 0.05)
                model.fit(x, pd.to_numeric(earlier['points']))
                models[position] = {'model': model, 'medians': medians}
            else:
                part = week[week['element_type'] == position]
                if not part.empty:
                    add_gameweek_trees({position: models[position]},
                                       part, OLD_WEEK_TREES)

        counts = ', '.join(
            f'{position}: {models[position]["model"].get_booster().num_boosted_rounds()}'
            for position in sorted(models)
        )
        print(f'Previous GW {int(gw)} trained - trees ({counts})')

    if set(models) != {1, 2, 3, 4}:
        raise ValueError('Not enough previous-season matches for all four positions')

    # For predictions NEXT season, use medians from the whole completed season.
    for position in models:
        rows = valid[valid['element_type'] == position]
        _, medians = _inputs(rows, FEATURES_MAP[position])
        models[position]['medians'] = medians
    return models


def save_model_checkpoint(models, folder, season, last_gw, previous_season=None):
    """Save XGBoost JSON models and the values needed to use them."""
    import json
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        'version': 1,
        'season': int(season),
        'last_gw': int(last_gw),
        'previous_season': previous_season,
        'features': {},
        'medians': {},
    }
    for position, bundle in models.items():
        features = FEATURES_MAP[position]
        metadata['features'][str(position)] = features
        medians = bundle['medians'].reindex(features).fillna(0)
        metadata['medians'][str(position)] = [float(v) for v in medians]
        bundle['model'].save_model(str(folder / f'position_{position}.json'))

    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2))


def load_model_checkpoint(folder):
    """Load the saved trees; refuse an incompatible feature list."""
    import json
    from pathlib import Path

    folder = Path(folder)
    metadata = json.loads((folder / 'metadata.json').read_text())
    if metadata.get('version') != 1:
        raise ValueError('Unknown saved-model version. Rebuild your models.')

    models = {}
    for key, features in metadata['features'].items():
        position = int(key)
        if features != FEATURES_MAP[position]:
            raise ValueError('Features have changed since saving the model. '
                             'Run main.py --rebuild-models to rebuild.')
        model = XGBRegressor()
        model.load_model(str(folder / f'position_{position}.json'))
        medians = pd.Series(metadata['medians'][key], index=features)
        models[position] = {'model': model, 'medians': medians}
    return models, metadata
