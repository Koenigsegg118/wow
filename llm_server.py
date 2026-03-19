from sim.llm_server import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).resolve().parent / "sim" / "llm_server.py"), run_name="__main__")
