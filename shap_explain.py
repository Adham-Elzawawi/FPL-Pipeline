"""Save SHAP explanations for selected FPL players."""
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import shap

from model import _inputs
from paper_features import FEATURES_MAP


def save_player_shap(predicted, ratings, models, gw, top_n=10, player_name=None):
    """Make one waterfall plot per fixture and a CSV of feature effects."""
    folder = Path('outputs/shap') / f'gw_{gw}'
    folder.mkdir(parents=True, exist_ok=True)

    if player_name is None:
        player_ids = ratings.head(top_n)['id'].tolist()
    else:
        # Find the player by display name, full name, or FPL ID.
        players = predicted.drop_duplicates('id')[
            ['id', 'web_name', 'first_name', 'second_name']
        ].copy()
        players['full_name'] = (
            players['first_name'].fillna('') + ' ' + players['second_name'].fillna('')
        ).str.strip()

        query = player_name.strip().casefold()
        if not query:
            raise ValueError('Please give a player name or ID for --shap-player')

        if query.isdigit():
            matches = players[players['id'] == int(query)]
        else:
            short = players['web_name'].str.casefold()
            full = players['full_name'].str.casefold()
            matches = players[(short == query) | (full == query)]
            if matches.empty:
                matches = players[
                    short.str.contains(re.escape(query), na=False)
                    | full.str.contains(re.escape(query), na=False)
                ]

        if matches.empty:
            raise ValueError(f'No player found for "{player_name}"')
        if len(matches) > 1:
            names = ', '.join(
                f"{row.web_name} (ID {row.id})" for row in matches.itertuples()
            )
            raise ValueError(f'Multiple players found: {names}. Use the FPL ID instead.')

        player_ids = matches['id'].tolist()
        print(f"Creating SHAP plots for {matches.iloc[0]['web_name']}...")
    games = predicted[
        predicted['id'].isin(player_ids)
        & predicted['fixture'].notna()
        & predicted['predicted_points'].notna()
    ].copy()
    if games.empty:
        print('No player predictions to explain with SHAP')
        return

    all_features = []
    for position in [1, 2, 3, 4]:
        part = games[games['element_type'] == position]
        if part.empty or position not in models:
            continue

        features = FEATURES_MAP[position]
        x, _ = _inputs(part, features, models[position]['medians'])
        explainer = shap.TreeExplainer(models[position]['model'])
        explanations = explainer(x)

        for i, (_, player) in enumerate(part.iterrows()):
            name = str(player['web_name'])
            fixture = int(player['fixture'])
            chance = float(player.get('playing_chance', 100))
            model_points = float(explanations[i].base_values + explanations[i].values.sum())
            adjusted_points = float(player['predicted_points']) * chance / 100

            # The SHAP values explain the raw model output, before our
            # zero-clipping and playing-chance adjustment.
            for feature, value, effect in zip(features, x.iloc[i], explanations[i].values):
                all_features.append({
                    'player': name,
                    'id': int(player['id']),
                    'fixture': fixture,
                    'feature': feature,
                    'feature_value': value,
                    'shap_points': effect,
                    'base_points': float(explanations[i].base_values),
                    'model_points': model_points,
                    'playing_chance': chance,
                    'adjusted_points': adjusted_points,
                })

            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
            filename = folder / f'{safe_name}_{int(player["id"])}_fixture_{fixture}.png'
            shap.plots.waterfall(explanations[i], max_display=12, show=False)
            plt.title(f'{name} | Fixture {fixture} | Model: {model_points:.2f} pts\n'
                      f'Playing chance: {chance:.0f}% | Adjusted: {adjusted_points:.2f} pts')
            plt.savefig(filename, dpi=160, bbox_inches='tight')
            plt.close()

    if all_features:
        pd.DataFrame(all_features).to_csv(folder / 'feature_contributions.csv', index=False)
        print(f'SHAP waterfall plots and feature contributions saved in {folder}')
