from pathlib import Path
import sys

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from sim.tools.acmi2weapon_events import *  # noqa: F401,F403

if __name__ == "__main__":
    from sim.tools.acmi2weapon_events import main

    main()
