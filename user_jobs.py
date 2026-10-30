"""Run the existing manager MILP optimizer for one user or all registered users."""
from pathlib import Path

from job_state import (add_user, last_success, manager_ids, mark_success,
                       open_db)
from manager_advisor import run as optimize_manager
from manager_advice import save_manager_advice


def latest_predictions():
    """Get the predictions for the CURRENT next GW, not an old output file."""
    from api import get_bootstrap
    from fixtures import next_gameweek

    events = get_bootstrap(refresh=True)['events']
    gw = next_gameweek(events)
    end_gw = min(gw + 3, 38)
    path = Path(f'outputs/gw_{gw}_to_{end_gw}_player_ratings.csv')
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Run the CP refresh first.')
    return path


def optimize_user(manager_id, predictions=None):
    """Never calls the XGBoost training pipeline."""
    if predictions is None:
        predictions = latest_predictions()
    print(f'\nOptimizing manager {manager_id} with {predictions}', flush=True)
    result = optimize_manager(int(manager_id), predictions=predictions)
    save_manager_advice(int(manager_id), result)
    with open_db() as db:
        mark_success(db, f'user:{manager_id}')
    return result


def register_and_optimize(manager_id):
    """Call this from your signup handler after you know their FPL ID."""
    if int(manager_id) <= 0:
        raise ValueError('Manager ID must be positive')
    result = optimize_user(int(manager_id))
    # Only register after a valid public manager squad was optimized.
    with open_db() as db:
        add_user(db, int(manager_id))
    return result


def optimize_all_users(predictions):
    """One failing or private manager must not block other users."""
    with open_db() as db:
        ids = manager_ids(db)
    print(f'Running personalized CP for {len(ids)} user(s).')
    for manager_id in ids:
        try:
            optimize_user(manager_id, predictions)
        except Exception as error:
            print(f'Manager {manager_id} failed: {error}')
