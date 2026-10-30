"""Create JSON-friendly SHAP values for players chosen by the optimizer.

These explanations describe the XGBoost points prediction, not the MILP
optimizer's transfer or budget decisions.
"""
import json
from pathlib import Path

import pandas as pd

from model import _inputs, load_model_checkpoint
from paper_features import FEATURES_MAP


SHAP_INPUTS_DIR = Path('outputs/shap_inputs')


def save_shap_inputs(predicted, gw):
    """Save model features during CP refresh so we can explain picks later."""
    feature_names = sorted(set(sum(FEATURES_MAP.values(), [])))
    columns = ['id', 'fixture', 'element_type', 'playing_chance',
               'predicted_points'] + feature_names
    data = predicted.reindex(columns=columns)
    SHAP_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SHAP_INPUTS_DIR / f'gw_{gw}.pkl.gz'
    temp = path.with_name(f'gw_{gw}.tmp.pkl.gz')
    data.to_pickle(temp)
    temp.replace(path)


def explain_selected_players(player_ids, gameweeks, expected_last_gw):
    """Return explanations indexed by player ID, then by gameweek.

    Only compute SHAP for requested players. Each fixture has its own
    baseline and contributions, including double-gameweek fixtures.
    """
    import shap

    player_ids = {int(player_id) for player_id in player_ids}
    gameweeks = [int(gw) for gw in gameweeks]
    if not player_ids or not gameweeks:
        return {}

    snapshot_path = Path('data/runtime/snapshot.json')
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    if int(snapshot['last_gw']) != int(expected_last_gw):
        raise ValueError('The saved SHAP models do not match these predictions.')

    model_dir = Path('data/models') / f"current_{snapshot['season']}"
    models, model_info = load_model_checkpoint(model_dir)
    if int(model_info['last_gw']) != int(expected_last_gw):
        raise ValueError('The SHAP models are from a different gameweek.')

    # One explainer per position, reused over the selected gameweeks.
    explainers = {}
    result = {player_id: {} for player_id in player_ids}
    for gw in gameweeks:
        input_path = SHAP_INPUTS_DIR / f'gw_{gw}.pkl.gz'
        if not input_path.exists():
            raise FileNotFoundError(
                f'{input_path} missing. Run python scheduler.py --force-cp.'
            )
        frames = pd.read_pickle(input_path)
        frames = frames[
            frames['id'].isin(player_ids)
            & frames['fixture'].notna()
            & frames['predicted_points'].notna()
        ]

        for player_id in player_ids:
            result[player_id][str(gw)] = {
                'gameweek': gw,
                'expectedPoints': 0.0,
                'fixtures': [],
            }

        for position, bundle in models.items():
            group = frames[frames['element_type'] == position]
            if group.empty:
                continue
            features = FEATURES_MAP[position]
            x, _ = _inputs(group, features, bundle['medians'])
            if position not in explainers:
                explainers[position] = shap.TreeExplainer(bundle['model'])
            explanations = explainers[position](x)

            for i, (_, player) in enumerate(group.iterrows()):
                player_id = int(player['id'])
                contribution_values = explanations[i].values
                base_points = float(explanations[i].base_values)
                model_points = base_points + float(contribution_values.sum())
                predicted_points = float(player['predicted_points'])
                chance = float(player['playing_chance']) if pd.notna(player['playing_chance']) else 100.0
                chance = max(0.0, min(100.0, chance))
                adjusted_points = predicted_points * chance / 100.0

                fixture_info = {
                    'fixtureId': int(player['fixture']),
                    'basePoints': round(base_points, 5),
                    'modelPoints': round(model_points, 5),
                    'predictedPoints': round(predicted_points, 5),
                    'adjustedPoints': round(adjusted_points, 5),
                    'playingChance': chance,
                    'features': [
                        {
                            'name': feature,
                            'value': round(float(value), 5),
                            'shapPoints': round(float(effect), 5),
                        }
                        for feature, value, effect in zip(
                            features, x.iloc[i], contribution_values
                        )
                    ],
                }
                result[player_id][str(gw)]['fixtures'].append(fixture_info)
                result[player_id][str(gw)]['expectedPoints'] += adjusted_points

        for player_id in player_ids:
            week = result[player_id][str(gw)]
            week['expectedPoints'] = round(week['expectedPoints'], 2)

    return result
