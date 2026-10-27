"""Run every hour via cron. Jobs run only when their conditions are met."""
import argparse
import fcntl
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# cron can call this script from anywhere.
PROJECT = Path(__file__).resolve().parent

from api import get_bootstrap, get_fixtures
from job_state import last_success, mark_success, open_db
from pipeline_jobs import (confirmed_gameweeks, read_snapshot, refresh_cp_job,
                           season_year, update_gw_job)
from user_jobs import optimize_all_users


def next_deadline(bootstrap, now):
    future = []
    for event in bootstrap['events']:
        deadline = datetime.fromisoformat(
            event['deadline_time'].replace('Z', '+00:00')
        )
        if deadline > now:
            future.append(deadline)
    return min(future) if future else None


def due_for_cp(db, bootstrap, now):
    deadline = next_deadline(bootstrap, now)
    if deadline is None:
        return False
    last = last_success(db, 'cp')
    if last is None:
        return True
    interval = timedelta(days=1)
    if deadline - now <= timedelta(hours=24):
        interval = timedelta(hours=1)
    return now - last >= interval


def run_scheduler(force_cp=False):
    # flock ensures a slow GW update cannot overlap with another cron run.
    lockfile = Path('data/scheduler.lock')
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    with lockfile.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another scheduler run is still in progress. Skipping.')
            return

        now = datetime.now(timezone.utc)
        bootstrap = get_bootstrap(refresh=True)
        raw_fixtures = pd.DataFrame(get_fixtures(refresh=True))
        completed = confirmed_gameweeks(bootstrap, raw_fixtures)
        season = season_year(bootstrap)
        snapshot = read_snapshot()
        last_gw = max(completed, default=0)
        must_update = (snapshot is None or snapshot['season'] != season
                       or int(snapshot['last_gw']) < last_gw)

        with open_db() as db:
            if must_update:
                print(f'GW update due. Official data through GW{last_gw}.')
                update_gw_job(bootstrap, raw_fixtures, completed)
            else:
                print(f'GW model is up to date through GW{last_gw}.')

            # A model update immediately refreshes predictions, even if the
            # most recent daily CP run was only an hour ago.
            # A newly entered GW should also get predictions straight away.
            from fixtures import next_gameweek
            if next_deadline(bootstrap, now) is None:
                cp_due = False
            else:
                current_gw = next_gameweek(bootstrap['events'])
                needed = Path(f'outputs/gw_{current_gw}_player_ratings.csv')
                cp_due = (must_update or force_cp or not needed.exists()
                          or due_for_cp(db, bootstrap, now))
            if cp_due:
                prediction_path = refresh_cp_job(bootstrap, raw_fixtures)
                if prediction_path is not None:
                    # Mark global CP as successful only after it has produced data.
                    mark_success(db, 'cp')
                    optimize_all_users(prediction_path)
            else:
                print('CP refresh not due yet.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-cp', action='store_true',
                        help='Refresh CP now, without retraining the models')
    args = parser.parse_args()
    try:
        import os
        os.chdir(PROJECT)
        run_scheduler(args.force_cp)
    except Exception as error:
        print(f'Scheduler failed: {error}', file=sys.stderr)
        raise
