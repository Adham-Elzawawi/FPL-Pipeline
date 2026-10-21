"""Offline tests for saved and incremental XGBoost training."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import train_previous_walk_forward, load_model_checkpoint
from model_cache import (train_previous_once, starting_from_previous,
                         train_current_incrementally)


def fake_training(weeks=4):
    """Small synthetic dataset: 12 matches/position/week."""
    rows = []
    for gw in range(1, weeks + 1):
        for position in range(1, 5):
            for player in range(1, 13):
                rows.append({
                    'round': gw, 'element_type': position,
                    'id': position * 100 + player,
                    'points': (gw + player + position) % 9,
                    'player_points_5': 3 + 0.1 * gw,
                    'att_fdr': 2 + gw % 3, 'def_fdr': 2,
                })
    return pd.DataFrame(rows)


class IncrementalModelTests(unittest.TestCase):
    def test_previous_adds_trees_gameweek_by_gameweek(self):
        models = train_previous_walk_forward(fake_training())
        self.assertEqual(set(models), {1, 2, 3, 4})
        for bundle in models.values():
            self.assertEqual(bundle['model'].get_booster().num_boosted_rounds(), 60)

    def test_previous_model_only_built_once_and_loaded(self):
        training = fake_training()
        fixtures = pd.DataFrame([
            {'id': gw, 'event': gw, 'finished': True,
             'team_h': 1, 'team_a': 2, 'team_h_score': 2,
             'team_a_score': 1} for gw in range(1, 5)
        ])
        teams = pd.DataFrame({'id': [1, 2], 'name': ['Arsenal', 'Chelsea']})
        calls = []

        def load_fake(year, refresh):
            calls.append(year)
            return training, fixtures, teams

        with tempfile.TemporaryDirectory() as tmp:
            old_models, strengths = train_previous_once(2025, load_fake,
                                                         model_folder=tmp)
            loaded_models, loaded_strengths = train_previous_once(
                2025, load_fake, model_folder=tmp
            )
            self.assertEqual(calls, [2025])
            self.assertEqual(len(loaded_models), 4)
            self.assertEqual(list(strengths['att_score']),
                             list(loaded_strengths['att_score']))
            model_file = Path(tmp) / 'previous_2025' / 'position_1.json'
            self.assertTrue(model_file.exists())
            self.assertTrue((model_file.parent / 'metadata.json').exists())
            self.assertEqual(old_models[1]['model'].get_booster().num_boosted_rounds(),
                             loaded_models[1]['model'].get_booster().num_boosted_rounds())

    def test_current_only_adds_each_completed_gw_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = train_previous_walk_forward(fake_training())
            current = fake_training(3)
            current.loc[current['round'] == 3, 'points'] = 100  # should not train yet
            first = train_current_incrementally(
                original, current, {1, 2}, 2026, 2025, model_folder=tmp
            )
            first_count = first[1]['model'].get_booster().num_boosted_rounds()
            self.assertEqual(first_count, 80)
            # No extra trees on rerun for the same completed gameweeks.
            second = train_current_incrementally(
                original, current, {1, 2}, 2026, 2025, model_folder=tmp
            )
            self.assertEqual(second[1]['model'].get_booster().num_boosted_rounds(),
                             first_count)
            last = train_current_incrementally(
                original, current, {1, 2, 3}, 2026, 2025, model_folder=tmp
            )
            self.assertEqual(last[1]['model'].get_booster().num_boosted_rounds(),
                             first_count + 10)
            _, info = load_model_checkpoint(Path(tmp) / 'current_2026')
            self.assertEqual(info['last_gw'], 3)

    def test_promoted_team_starts_neutral(self):
        old = pd.DataFrame([
            {'id': 5, 'name': 'Arsenal', 'att_score': 4.1, 'def_score': 4.0},
        ])
        new = pd.DataFrame([
            {'id': 14, 'name': 'Arsenal'},
            {'id': 21, 'name': 'Promoted FC'},
        ])
        rating = starting_from_previous(old, new)
        self.assertAlmostEqual(rating.loc[14, 'att_score'], 4.1)
        self.assertEqual(rating.loc[21, 'att_score'], 3.0)
        self.assertEqual(rating.loc[21, 'def_score'], 3.0)


if __name__ == '__main__':
    unittest.main()
