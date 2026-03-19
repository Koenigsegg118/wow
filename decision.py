if __name__ == "__main__":
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).resolve().parent / "sim" / "decision.py"), run_name="__main__")
