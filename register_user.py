"""Demo signup trigger; your future API can call register_and_optimize()."""
import argparse
import os
from pathlib import Path

from user_jobs import register_and_optimize

if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, required=True, help='FPL manager ID')
    args = parser.parse_args()
    register_and_optimize(args.id)
