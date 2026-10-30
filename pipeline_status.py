"""Write a small status file that Next.js can read while the pipeline runs."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

STATUS_FILE = Path('outputs/pipeline_status.json')


def set_pipeline_status(state, message):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'state': state,
        'message': message,
        'updatedAt': datetime.now(timezone.utc).isoformat(),
        # The web API uses this to detect jobs that stopped unexpectedly.
        'pid': os.getpid() if state in ('checking', 'training', 'predicting') else None,
    }
    temporary = STATUS_FILE.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temporary, STATUS_FILE)
