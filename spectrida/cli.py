"""spectrIDA CLI."""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Ghost through binaries — parallel IDA analysis + AI naming.")


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    demo: bool = typer.Option(False, "--demo", help="Run the TUI on canned data (no IDA/Ollama)."),
    no_onboard: bool = typer.Option(False, "--no-onboard", help="Skip the first-run wizard."),
):
    from spectrida import config
    if not config.onboarded() and not no_onboard:
        from spectrida.onboard import run_onboarding
        run_onboarding()
        if ctx.invoked_subcommand is None:
            demo = True  # first-run bare command → land in the demo
    if ctx.invoked_subcommand is None:
        from spectrida.tui.app import SpectrIDAApp
        SpectrIDAApp(demo=demo).run()


@app.command()
def analyze(
    binary: str = typer.Argument(..., help="Binary to analyze (DLL/EXE/NSO…)."),
    workers: int = typer.Option(None, "-w", "--workers"),
):
    """Run parallel analysis, then open the browser."""
    p = Path(binary).expanduser()
    if not p.exists():
        typer.echo(f"error: not found: {p}", err=True)
        raise typer.Exit(1)
    from spectrida.tui.app import SpectrIDAApp
    SpectrIDAApp(binary=str(p.resolve()), workers=workers).run()


@app.command("open")
def open_(i64: str = typer.Argument(..., help="Path to an .i64 database.")):
    """Open an existing .i64 in the browser."""
    p = Path(i64).expanduser()
    if not p.exists():
        typer.echo(f"error: not found: {p}", err=True)
        raise typer.Exit(1)
    from spectrida.tui.app import SpectrIDAApp
    SpectrIDAApp(i64=str(p.resolve())).run()


@app.command()
def onboard():
    """Re-run the setup wizard, then open the demo."""
    from spectrida.onboard import run_onboarding
    run_onboarding(force=True)
    from spectrida.tui.app import SpectrIDAApp
    SpectrIDAApp(demo=True).run()


@app.command()
def serve():
    """Check Ollama + the model are ready."""
    import asyncio

    from spectrida.config import ollama_model
    from spectrida.core.services import ensure_model_loaded, ensure_ollama, model_present

    async def _check():
        if not await ensure_ollama():
            typer.echo("✗ Ollama not reachable. Install: https://ollama.com/download", err=True)
            raise typer.Exit(1)
        typer.echo("● Ollama up")
        if await model_present():
            await ensure_model_loaded()
            typer.echo(f"● {ollama_model()} ready")
        else:
            typer.echo(f"✗ {ollama_model()} not pulled — ollama pull hf.co/gdfhhjk/spectrida-re-gguf", err=True)

    asyncio.run(_check())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
