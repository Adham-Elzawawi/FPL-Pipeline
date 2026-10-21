"""Train and save last season's model once, before using the current season."""
import argparse

from model_cache import train_previous_once
from previous_season import previous_season_training


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True,
                        help='Season start year, e.g. 2025 for 2025/26')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()

    train_previous_once(args.year, previous_season_training,
                        refresh=args.refresh, rebuild=args.rebuild)
