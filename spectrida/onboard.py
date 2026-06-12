"""First-run setup — a text flow (rich console, no TUI). Has jokes. Skippable forever."""
from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel

from spectrida import config, voice
from spectrida.core import services

_GHOST = r"""[magenta]
        .-.
       (o o)    boo.
       | O |
       '~~~'[/]"""


def run_onboarding(force: bool = False) -> None:
    if config.onboarded() and not force:
        return
    c = Console()
    c.print(_GHOST)
    c.print(Panel.fit(
        "[b cyan]hey. i'm the ghost.[/]\n\n"
        "I name functions while you get coffee. I shard binaries so IDA doesn't take an\n"
        "eight-minute nap. I'm not Ghidra — never claimed to be — but the thing I do, I do\n"
        "[b]fast[/], and I'll be honest when I'm guessing.\n\n"
        "Quick setup. 20 seconds. [dim](you can ignore all of this — demo mode needs none of it.)[/]",
        border_style="cyan"))

    async def checks() -> list[str]:
        out = ["[green]✓[/]  Python — you're running me, so, yeah."]
        if config.idalib_dir() and services.idalib_ok():
            out.append("[green]✓[/]  IDA / idalib — found it. nice.")
        else:
            out.append("[yellow]•[/]  IDA / idalib — not set. Put your IDA path in "
                       "[b]~/.spectrida/config.toml[/] under [b][ida] idalib[/]. "
                       "[dim](demo works without it.)[/]")
        if not services.ollama_installed():
            out.append(f"[yellow]•[/]  Ollama — not installed:  [b]{services.ollama_install_hint()}[/]")
        elif not await services.ollama_running():
            out.append("[yellow]•[/]  Ollama — installed but napping. Run [b]ollama serve[/].")
        else:
            out.append("[green]✓[/]  Ollama — up and awake.")
            if await services.model_present():
                out.append("[green]✓[/]  the model — pulled and ready. you absolute professional.")
            else:
                out.append("[yellow]•[/]  the model:  [b]ollama pull hf.co/gdfhhjk/spectrida-re-gguf[/] "
                           "[dim](8.7 GB, worth it)[/]")
        return out

    c.print("\n[b]checking your setup…[/]")
    for line in asyncio.run(checks()):
        c.print("  " + line)

    config.write_default_config()
    config.set_onboarded()
    c.print("\n  [dim]wrote a starter config to ~/.spectrida/config.toml[/]")
    c.print("\n  [b]Keys inside:[/] [cyan]N[/] name · [cyan]C[/] chain · [cyan]D[/] decompile · "
            "[cyan]/[/] search · [cyan]?[/] help")
    c.print(f"  [dim]{voice.quip('welcome')}[/]\n")
    c.print("  [b]launching the demo — go ghost through some binaries.[/] 👻\n")
