"""Run the FPL predictor. Save old-season models and update them each GW."""
import argparse
from pathlib import Path

import pandas as pd

from api import get_bootstrap, get_fixtures
from fixtures import completed_gameweeks, next_gameweek
from strengths import make_fixture_table
from team_features import build_team_history, add_team_features
from understat_data import get_understat_matches
from model import predict_matches
from model_cache import (train_previous_once, starting_from_previous,
                         train_current_incrementally)
from players import add_past_averages, get_players, load_history, upcoming_players
from previous_season import previous_season_training
from ratings import make_player_ratings
from team_selection import choose_xi
from plot_fixtures import save_fixture_plots


POSITION_NAMES = {
    'GK': 'GOALKEEPERS',
    'DEF': 'DEFENDERS',
    'MID': 'MIDFIELDERS',
    'FWD': 'FORWARDS',
}


def print_players_by_position(players, title):
    """Print the players in football position order."""
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)

    for position, label in POSITION_NAMES.items():
        group = players[players['position'] == position]
        if group.empty:
            continue

        group = group.sort_values('expected_points', ascending=False)
        print(f'\n{label} ({len(group)})')
        print('-' * 78)

        columns = ['web_name', 'team_name', 'opponents', 'price_m',
                   'playing_chance', 'minutes_security', 'expected_points']
        if 'captain' in group.columns:
            columns.append('captain')

        table = group[columns].rename(columns={
            'web_name': 'Player',
            'team_name': 'Team',
            'opponents': 'Fixtures',
            'price_m': 'Price (£m)',
            'playing_chance': 'Chance (%)',
            'minutes_security': 'Min Security (%)',
            'expected_points': 'xPts',
            'captain': 'C/VC',
        })
        print(table.to_string(index=False, formatters={
            'Price (£m)': lambda value: f'{value:.1f}',
            'Chance (%)': lambda value: f'{value:.0f}',
            'Min Security (%)': lambda value: f'{value:.1f}',
            'xPts': lambda value: f'{value:.2f}',
        }))

    if 'captain' in players.columns:
        captain_points = players.loc[
            players['captain'] == 'C', 'expected_points'
        ].sum()
        total_points = players['expected_points'].sum() + captain_points
        print('\n' + '-' * 78)
        print(f'Team cost: £{players["price_m"].sum():.1f}m  |  '
              f'Expected XI points (captain x2): {total_points:.2f}')


def run(gw=None, refresh=False, budget=81.5, pick_team=True,
        rebuild_models=False, model_folder="data/models", explain_shap=False,
        shap_player=None):
    print('1. Reading current FPL season...')
    # Always update injury news even if match histories stay cached.
    bootstrap = get_bootstrap(refresh=True)
    raw_fixtures = pd.DataFrame(get_fixtures(refresh))
    teams = pd.DataFrame(bootstrap['teams'])
    if gw is None:
        gw = next_gameweek(bootstrap['events'])
    players = get_players(bootstrap, gw)
    season = pd.to_datetime(bootstrap['events'][0]['deadline_time']).year
    previous_year = season - 1
    print(f'Previous season: {previous_year}; current season: {season}; GW {gw}')

    print('2. Loading saved previous-season models and strengths...')
    old_models, saved_strengths = train_previous_once(
        previous_year, previous_season_training, refresh=refresh,
        rebuild=rebuild_models, model_folder=model_folder
    )
    base_strengths = starting_from_previous(saved_strengths, teams)

    print('3. Recalculating current-season dynamic FDR for every GW...')
    fixtures = make_fixture_table(
        raw_fixtures, teams, gw, initial_strengths=base_strengths
    )
    if fixtures[fixtures['round'] == gw].empty:
        raise ValueError(f'No fixtures found for gameweek {gw}')

    print('3b. Saving next 4 gameweeks FDR plots...')
    save_fixture_plots(fixtures, teams, start_gw=gw, next_n=4, output_dir='fpl-web/public/fixtures')

    completed = completed_gameweeks(raw_fixtures, gw)

    print('4. Building current-season team and opponent features...')
    understat_teams, understat_players = get_understat_matches(
        raw_fixtures, teams, season, refresh
    )
    if completed:
        team_history = build_team_history(raw_fixtures, understat_teams)
    else:
        team_history = pd.DataFrame(columns=['round', 'fixture', 'team', 'kickoff_time'])
    fixtures = add_team_features(fixtures, team_history, completed)

    print('5. Building current-season player features...')
    if completed:
        history = load_history(players, fixtures, refresh, understat_players)
        history = history[history['round'].isin(completed)].copy()
        current_training = add_past_averages(history)
    else:
        history = pd.DataFrame(columns=['id', 'round', 'fixture', 'team',
                                        'element_type', 'was_home'])
        current_training = pd.DataFrame(columns=['round', 'element_type', 'points'])

    print('6. Adding trees only for newly completed gameweeks...')
    models = train_current_incrementally(
        old_models, current_training, completed, season, previous_year,
        rebuild=rebuild_models, model_folder=model_folder
    )
    upcoming = upcoming_players(players, history, fixtures, gw)
    predicted = predict_matches(upcoming, models)
    ratings = make_player_ratings(predicted, teams, gw)

    output = Path('outputs')
    output.mkdir(exist_ok=True)
    next_four = fixtures[fixtures["round"].between(gw, gw + 3)].copy()

    team_names = teams.set_index("id")["name"]
    short_names = teams.set_index("id")["short_name"]

    next_four["team_name"] = next_four["team"].map(team_names)
    next_four["opponent_name"] = next_four["opponent_team"].map(short_names)

    next_four.to_csv(output / "next_4_fixtures.csv", index=False)
    
    ratings.to_csv(output / f'gw_{gw}_player_ratings.csv', index=False)
    fixtures[fixtures['round'] == gw].to_csv(output / f'gw_{gw}_fixtures.csv', index=False)
    print_players_by_position(ratings.head(10), 'TOP 10 PREDICTED PLAYERS')

    if explain_shap or shap_player is not None:
        from shap_explain import save_player_shap
        save_player_shap(predicted, ratings, models, gw, player_name=shap_player)

    if pick_team:
        print('\n7. Selecting a starting XI...')
        try:
            xi = choose_xi(ratings, budget)
            xi.to_csv(output / f'gw_{gw}_starting_xi.csv', index=False)
            print_players_by_position(xi, 'SELECTED STARTING XI')
        except ValueError as error:
            print(f'Could not select a team: {error}')

    print(f'Files saved in: {output.resolve()}')
    return ratings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-season FPL XGBoost predictor')
    parser.add_argument('--gw', type=int, help='Gameweek (default next GW)')
    parser.add_argument('--refresh', action='store_true', help='Refresh cached FPL, GitHub and Understat data')
    parser.add_argument('--budget', type=float, default=81.5, help='XI budget (£m)')
    parser.add_argument('--no-team', action='store_true', help='Skip starting XI')
    parser.add_argument('--rebuild-models', action='store_true',
                        help='Re-train the saved previous and current season models')
    parser.add_argument('--shap', action='store_true',
                        help='Save SHAP plots for the top 10 predicted players')
    parser.add_argument('--shap-player', type=str,
                        help='Save SHAP plots for a player by name or FPL ID')
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error('--budget must be positive')
    run(args.gw, args.refresh, args.budget, not args.no_team,
        rebuild_models=args.rebuild_models, explain_shap=args.shap,
        shap_player=args.shap_player)
