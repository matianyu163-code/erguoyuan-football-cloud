"""Initialize independent, empty Cloud Era append-only state."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erguoyuan_football.blind_test_r3.user_daily_runner import init_state

if __name__ == "__main__":
    init_state()
    print("CLOUD_STATE_INITIALIZED")
