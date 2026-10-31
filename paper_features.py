"""Feature names for the player and team prediction models."""

WINDOWS = (3, 5, 8, 10)
# Keep the old name in case another script still imports it.
WINDOW = 5

PLAYER_COMMON = [
    'points', 'relevant_points', 'minutes', 'influence', 'creativity',
    'threat', 'assists', 'goals_conceded', 'own_goals', 'yellow_cards',
    'red_cards', 'bps', 'bonus', 'defensive_contribution', 'xg_chain', 'xg_buildup', 'key_passes',
    'expected_assists',
]
GK_EXTRA = ['saves', 'penalties_saved']
OUTFIELD_EXTRA = ['goals_scored', 'penalties_missed', 'shots', 'expected_goals']

# Some stats come from FPL and some from Understat.
TEAM_METRICS = [
    'goals_scored', 'goals_conceded', 'xg', 'deep_allowed',
    'ppda_allowed_att', 'ppda_allowed_def', 'xga', 'deep',
    'ppda_att', 'ppda_def',
]


def player_features(position):
    names = PLAYER_COMMON + (GK_EXTRA if position == 1 else OUTFIELD_EXTRA)
    return [f'player_{name}_{window}' for window in WINDOWS for name in names]


def side_features(side):
    return [f'{side}_{name}_{window}' for window in WINDOWS for name in TEAM_METRICS]


def features_for_position(position):
    # Use our own FDR and previous points at the same venue.
    return (player_features(position) + side_features('team') +
            side_features('opponent') +
            ['player_availability', 'att_fdr', 'def_fdr'])


FEATURES_MAP = {position: features_for_position(position) for position in range(1, 5)}
ALL_PLAYER_METRICS = sorted(set(PLAYER_COMMON + GK_EXTRA + OUTFIELD_EXTRA))
