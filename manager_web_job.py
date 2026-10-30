"""Run one user's existing optimizer in a separate process.

Launched by POST /api/manager-advice. Writes progress so the web page can poll.
"""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from manager_advice import OUTPUT_DIR, write_json
from user_jobs import register_and_optimize


def set_status(manager_id, state, message=None):
    data = {
        'state': state,
        'updatedAt': datetime.now(timezone.utc).isoformat(),
    }
    if message:
        data['message'] = str(message)
    write_json(OUTPUT_DIR / f'manager_{manager_id}.status.json', data)


def main(manager_id):
    os.chdir(Path(__file__).resolve().parent)
    set_status(manager_id, 'running')
    try:
        register_and_optimize(manager_id)
    except Exception as error:
        print(f'Manager {manager_id} failed: {error}', flush=True)
        set_status(manager_id, 'error', str(error))
        raise
    set_status(manager_id, 'ready')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True, type=int)
    args = parser.parse_args()
    if args.id <= 0:
        parser.error('Manager ID must be a positive number')
    main(args.id)
