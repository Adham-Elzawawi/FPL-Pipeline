"""Small checks for the rolling-window backtest."""

import numpy as np
import pandas as pd

from backtest import get_feature_sets, pick_squad, score_squad
from players import add_past_averages


def test_feature_counts():
    features = get_feature_sets()
    assert len(features['5'][1]) == 43
    assert len(features['5'][2]) == 45
    assert len(features['3-5-8-10'][1]) == 163
    assert len(features['3-5-8-10'][3]) == 171
    assert 'player_minutes_8' in features['3-5-8-10'][3]
    assert 'player_minutes_8' not in features['5'][3]


def test_rolling_windows_use_only_previous_matches():
    games = pd.DataFrame({
        'id': [1] * 11, 'round': list(range(1, 12)),
        'fixture': list(range(101, 112)),
        'was_home': [1] * 11,
        'points': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000],
        'minutes': [90] * 11,
    })
    prepared = add_past_averages(games)
    last = prepared.loc[prepared['round'] == 11].iloc[0]
    assert last['player_points_3'] == 9
    assert last['player_points_5'] == 8
    assert last['player_points_8'] == 6.5
    assert last['player_points_10'] == 5.5
    assert last['player_relevant_points_5'] == 8


def test_optimizer_builds_legal_squad():
    rows = []
    player_id = 1
    for position, total in ((1, 5), (2, 12), (3, 14), (4, 8)):
        for n in range(total):
            rows.append({
                'id': player_id, 'element_type': position,
                'team': 1 + player_id % 20,
                'now_cost': 45 + (n % 4) * 5,
                'expected_points': 3 + (n % 5),
                'web_name': f'Player {player_id}',
            })
            player_id += 1
    selected = pick_squad(pd.DataFrame(rows), budget=100)
    assert len(selected) == 15
    assert selected['starting'].sum() == 11
    assert selected['captain'].sum() == 1
    assert selected['vice_captain'].sum() == 1
    assert selected['now_cost'].sum() <= 1000
    assert selected['team'].value_counts().max() <= 3
    assert selected.groupby('element_type').size().to_dict() == {1: 2, 2: 5, 3: 5, 4: 3}
    starters = selected[selected['starting']]['element_type'].value_counts()
    assert starters[1] == 1
    assert 3 <= starters[2] <= 5
    assert 3 <= starters[3] <= 5
    assert 1 <= starters[4] <= 3


def test_captain_is_doubled_when_playing():
    team = pd.DataFrame({
        'id': list(range(1, 16)),
        'element_type': [1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 1, 2, 2, 4],
        'starting': [True] * 11 + [False] * 4,
        'bench_order': [0] * 11 + [4, 1, 2, 3],
        'captain': [True] + [False] * 14,
        'vice_captain': [False, True] + [False] * 13,
    })
    actual = pd.DataFrame({'points': [2] * 15, 'minutes': [90] * 15},
                          index=list(range(1, 16)))
    assert score_squad(team, actual) == 24
    actual.loc[1, 'minutes'] = 0
    actual.loc[1, 'points'] = 0
    # Backup goalkeeper enters, and the vice captain is doubled.
    assert score_squad(team, actual) == 24


def test_score_audit_reconciles_points():
    from backtest import score_audit
    team = pd.DataFrame({
        'id': list(range(1, 16)),
        'web_name': [f'P{i}' for i in range(1, 16)],
        'element_type': [1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 1, 2, 2, 4],
        'starting': [True] * 11 + [False] * 4,
        'bench_order': [0] * 11 + [4, 1, 2, 3],
        'captain': [True] + [False] * 14,
        'vice_captain': [False, True] + [False] * 13,
        'expected_points': [3.0] * 15,
    })
    actual = pd.DataFrame({'points': [2.0] * 15, 'minutes': [90] * 15},
                          index=list(range(1, 16)))
    actual.loc[1, ['points', 'minutes']] = 0
    audit = score_audit(team, actual)
    assert audit['counted_points'].sum() == score_squad(team, actual)
    assert audit.loc[audit['id'] == 2, 'captain_bonus'].iloc[0] == 2


def test_candidate_rows_keep_double_gameweek(monkeypatch):
    import backtest
    history = pd.DataFrame({
        'id': [1], 'round': [3], 'fixture': [42],
        'team': [4], 'element_type': [3], 'now_cost': [75],
    })
    players = pd.DataFrame({'id': [1], 'web_name': ['Test'],
                            'first_name': ['First'], 'second_name': ['Last']})

    def fake_upcoming(_people, _past, _fixture_table, gw):
        return pd.DataFrame({'id': [1, 1], 'fixture': [101, 102],
                             'round': [gw, gw], 'team': [4, 4],
                             'element_type': [3, 3], 'now_cost': [75, 75]})

    monkeypatch.setattr(backtest, 'upcoming_players', fake_upcoming)
    candidates = backtest.make_candidates(4, history, pd.DataFrame(), players)
    assert len(candidates) == 2
    assert candidates['fixture_count'].tolist() == [2, 2]


def test_frozen_data_can_be_reloaded(tmp_path):
    from backtest import save_data_snapshot, load_data_snapshot
    path = tmp_path / 'snapshot.pkl.gz'
    small = pd.DataFrame({'id': [1, 2], 'points': [5, 10]})
    save_data_snapshot(path, small, small, small, small, small,
                       {1, 2, 3}, {4: 55}, [4])
    loaded = load_data_snapshot(path)
    pd.testing.assert_frame_equal(loaded[0], small)
    assert loaded[-1] == [4]
    assert loaded[-2] == {4: 55}


def test_optimizer_is_reproducible_with_same_inputs():
    rows = []
    player_id = 1
    for position, total in ((1, 3), (2, 7), (3, 7), (4, 5)):
        for n in range(total):
            rows.append({
                'id': player_id, 'element_type': position,
                'team': 1 + player_id % 20,
                'now_cost': 45 + (n % 4) * 5,
                'expected_points': 3.0 + n * 0.7 + position * 0.01,
                'web_name': f'P{player_id}',
            })
            player_id += 1
    players = pd.DataFrame(rows)
    a = pick_squad(players, budget=100)
    b = pick_squad(players, budget=100)
    assert a[a['starting']]['id'].tolist() == b[b['starting']]['id'].tolist()
    assert a[a['captain']]['id'].tolist() == b[b['captain']]['id'].tolist()


def test_mse_and_sse_for_each_gw():
    from backtest import metrics
    rows = pd.DataFrame({
        'model': ['5', '5', '5', '3-5-8-10', '3-5-8-10'],
        'gw': [4, 4, 5, 4, 4],
        'actual': [3, 5, 2, 3, 5],
        'predicted': [1, 4, 3, 2, 5],
    })
    result = metrics(rows)
    five_gw4 = result[(result['model'] == '5') & (result['gw'] == 4)].iloc[0]
    assert five_gw4['players'] == 2
    assert np.isclose(five_gw4['sse'], 5)
    assert np.isclose(five_gw4['mse'], 2.5)
    assert np.isclose(five_gw4['rmse'], np.sqrt(2.5))
    multi_gw4 = result[(result['model'] == '3-5-8-10') & (result['gw'] == 4)].iloc[0]
    assert np.isclose(multi_gw4['mse'], 0.5)
    assert np.isclose(multi_gw4['sse'], 1)
    five_gw5 = result[(result['model'] == '5') & (result['gw'] == 5)].iloc[0]
    assert np.isclose(five_gw5['mse'], 1)


def test_recalculate_mse_without_retraining(tmp_path):
    from backtest import mse_from_saved_predictions
    rows = pd.DataFrame({
        'model': ['5', '5', '3-5-8-10', '3-5-8-10'],
        'gw': [4, 4, 4, 4],
        'actual': [3, 5, 3, 5],
        'predicted': [1, 4, 2, 5],
    })
    rows.to_csv(tmp_path / 'backtest_player_predictions.csv', index=False)
    mse_from_saved_predictions(tmp_path)
    csv_path = tmp_path / 'backtest_mse_by_gw.csv'
    assert csv_path.exists()
    assert (tmp_path / 'backtest_mse_by_gw.png').exists()
    assert (tmp_path / 'backtest_prediction_metrics.csv').exists()
    result = pd.read_csv(csv_path)
    assert set(result['model']) == {'5', '3-5-8-10'}
    assert np.isclose(result.loc[result['model'] == '5', 'mse'].iloc[0], 2.5)
