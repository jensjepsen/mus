from . import run_file

if __name__ == "__main__":
    try:
        import typer
    except ImportError:
        raise ImportError("Please install the cli dependencies by running `pip install mus[cli]`")
    typer.run(run_file)