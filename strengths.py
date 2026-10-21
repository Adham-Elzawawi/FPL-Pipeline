"""Calculate team strengths and our dynamic fixture difficulty."""

from pathlib import Path
import pandas as pd
import math

from fixtures import completed_gameweeks


LEAGUE_AVERAGE_GOALS = 1.4


def expected_goals(attack, opponent_defence, home_modifier=0):
    """Estimate goals using team strengths and home advantage."""

    goals = LEAGUE_AVERAGE_GOALS * attack / opponent_defence

    return max(0.1, goals + home_modifier)


def poisson_probability(goals_expected, goals_scored):
    """Probability of scoring exactly N goals."""

    return (
        math.exp(-goals_expected)
        * goals_expected ** goals_scored
        / math.factorial(goals_scored)
    )


def starting_strengths(teams, csv_file="data/team_strengths.csv"):
    """Start from the saved strengths, or 3/3 for each team."""

    ratings = pd.DataFrame({
        "team_id": teams["id"].astype(int),
        "att_score": 3.0,
        "def_score": 3.0,
        "home_modifier": 0.35,
    })

    if csv_file and Path(csv_file).exists():
        provided = pd.read_csv(csv_file)

        required = [
            "team_id", "att_score",
            "def_score", "home_modifier"
        ]

        missing = set(required) - set(provided.columns)

        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        if provided["team_id"].duplicated().any():
            raise ValueError("Duplicate team IDs")

        if not set(ratings["team_id"]).issubset(set(provided["team_id"])):
            raise ValueError("Missing teams in CSV")

        ratings = provided[required].copy()

    return ratings.set_index("team_id")


# TODO: Maybe improve how the starting strengths are normalized.
def _updated_strengths(fixtures, ratings, previous_gws):
    """Adjust team strengths using matches already finished."""

    played = fixtures[fixtures["event"].isin(previous_gws)].copy()

    played = played.dropna(
        subset=["team_h_score", "team_a_score"]
    )

    if played.empty:
        return ratings.copy()

    # Make home team scores.
    home = played[
        ["team_h", "team_h_score", "team_a_score"]
    ].copy()

    home.columns = ["team_id", "scored", "conceded"]

    # Make away team scores.
    away = played[
        ["team_a", "team_a_score", "team_h_score"]
    ].copy()

    away.columns = ["team_id", "scored", "conceded"]

    # Put all team results in one table.
    results = pd.concat([home, away], ignore_index=True)

    # Work out average goals scored and conceded.
    averages = results.groupby("team_id").agg(
        games=("scored", "size"),
        scored=("scored", "mean"),
        conceded=("conceded", "mean")
    )

    # Use a fixed league average to avoid early-season weird results.
    league_average = LEAGUE_AVERAGE_GOALS

    updated = ratings.copy()

    for team_id, row in averages.iterrows():

        if team_id not in updated.index:
            continue

        # Trust the results more when a team has played more games.
        weight = row["games"] / (row["games"] + 7)

        # Use actual goals to update our score.
        measured_attack = (
            3 + (row["scored"] - league_average) * 1.5
        )

        measured_defence = (
            3 + (league_average - row["conceded"]) * 1.5
        )

        # Mix the starting score with the new results.
        updated.loc[team_id, "att_score"] = (
            (1 - weight) * updated.loc[team_id, "att_score"]
            + weight * measured_attack
        )

        updated.loc[team_id, "def_score"] = (
            (1 - weight) * updated.loc[team_id, "def_score"]
            + weight * measured_defence
        )

    # Keep scores in the 1 to 5 range.
    updated[["att_score", "def_score"]] = (
        updated[["att_score", "def_score"]].clip(1, 5)
    )

    return updated


def make_fixture_table(fixtures, teams, target_gw, initial_strengths=None):
    """Calculate our FDR and goal probabilities for each fixture."""

    games = fixtures.dropna(subset=["event"]).copy()

    if games.empty:
        raise ValueError("No Premier League fixtures found")

    games["event"] = games["event"].astype(int)

    # Only completed GWs can update the strengths.
    finished_gws = completed_gameweeks(games, target_gw)

    # Starting ratings from last season, or 3/3.
    baseline = initial_strengths.copy() if initial_strengths is not None else starting_strengths(teams)

    rows = []

    for gw in sorted(games["event"].unique()):

        # Do not look at games from this GW or later.
        previous = {x for x in finished_gws if x < gw}

        strength = _updated_strengths(games, baseline, previous)

        week_games = games[games["event"] == gw]

        for game in week_games.to_dict("records"):

            home_team = int(game["team_h"])
            away_team = int(game["team_a"])

            # Get both teams' current attack/defence values.
            home_att = strength.loc[home_team, "att_score"]
            home_def = strength.loc[home_team, "def_score"]

            away_att = strength.loc[away_team, "att_score"]
            away_def = strength.loc[away_team, "def_score"]

            home_modifier = strength.loc[home_team, "home_modifier"]

            # Estimate goals using the strengths.

            home_xg = expected_goals(
                home_att,
                away_def,
                home_modifier
            )

            away_xg = expected_goals(
                away_att,
                home_def
            )

            # The table has one row for the home and away side.

            for is_home in [True, False]:

                team = home_team if is_home else away_team
                opponent = away_team if is_home else home_team

                team_att = strength.loc[team, "att_score"]
                team_def = strength.loc[team, "def_score"]

                opp_att = strength.loc[opponent, "att_score"]
                opp_def = strength.loc[opponent, "def_score"]

                # Add home advantage to the FDR calculation.
                home_adjustment = (
                    -strength.loc[team, "home_modifier"]
                    if is_home
                    else strength.loc[opponent, "home_modifier"]
                )

                # Calculate our attacking and defensive FDR.
    
                attack_fdr_raw = 3 + opp_def - team_att + home_adjustment
                defence_fdr_raw = 3 + opp_att - team_def + home_adjustment

                attack_fdr = round(attack_fdr_raw)
                defence_fdr = round(defence_fdr_raw)

                attack_fdr = max(1, min(5, attack_fdr))
                defence_fdr = max(1, min(5, defence_fdr))

                # Estimate goal probabilities.
    
                # Expected goals from this team's perspective
                team_xg = home_xg if is_home else away_xg
                opponent_xg = away_xg if is_home else home_xg

                # Probability of conceding 0 goals
                clean_sheet_prob = poisson_probability(
                    opponent_xg, 0
                )

                # Probability of scoring at least 1 goal
                score_1plus_prob = (
                    1 - poisson_probability(team_xg, 0)
                )

                # Probability of scoring at least 2 goals
                score_2plus_prob = 1 - (
                    poisson_probability(team_xg, 0)
                    + poisson_probability(team_xg, 1)
                )

                    # Save the numbers for this match.
    
                rows.append({
                    "fixture": int(game["id"]),
                    "round": int(gw),
                    "team": team,
                    "opponent_team": opponent,
                    "was_home": int(is_home),

                    "att_fdr": attack_fdr,
                    "def_fdr": defence_fdr,
                    "att_fdr_raw": round(attack_fdr_raw, 3),
                    "def_fdr_raw": round(defence_fdr_raw, 3),

                    "team_att_strength": round(team_att, 3),
                    "team_def_strength": round(team_def, 3),
                    "opp_att_strength": round(opp_att, 3),
                    "opp_def_strength": round(opp_def, 3),

                    "team_xg": round(team_xg, 3),
                    "opponent_xg": round(opponent_xg, 3),
                    "clean_sheet_prob": round(clean_sheet_prob, 3),
                    "score_1plus_prob": round(score_1plus_prob, 3),
                    "score_2plus_prob": round(score_2plus_prob, 3),
                })

    return pd.DataFrame(rows)


def last_season_strengths(previous_fixtures, previous_teams, current_teams):
    """Carry old ratings over. New promoted teams start at 3/3."""
    from understat_data import team_name

    previous = previous_fixtures.dropna(subset=['event']).copy()
    finished = completed_gameweeks(previous, 99)
    baseline = starting_strengths(previous_teams, csv_file=None)
    final = _updated_strengths(previous, baseline, finished)

    new = starting_strengths(current_teams, csv_file=None)
    old_ids = {team_name(t['name']): int(t['id'])
               for t in previous_teams.to_dict('records')}
    for team in current_teams.to_dict('records'):
        previous_id = old_ids.get(team_name(team['name']))
        if previous_id is None:
            # No Premier League rating last season, so leave it at 3/3.
            continue
        new.loc[int(team['id']), ['att_score', 'def_score']] = (final.loc[
            previous_id, ['att_score', 'def_score']
        ].to_numpy())
    return new
