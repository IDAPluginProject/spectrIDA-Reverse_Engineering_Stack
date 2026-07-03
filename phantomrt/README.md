<div align="center">

# 👻 phantomrt

**Ghosts don't just read the walls. They walk through them.**

The dynamic half of [spectrIDA](https://github.com/ggfuchsi-oss/spectrIDA-Reverse_Engineering_Stack).
spectrIDA reads the binary and names the functions. phantomrt **runs** them — emulates,
live-instruments, and fuzzes — then writes what actually happened back onto the same graph.

`pip install phantomrt`

</div>

```
   function: al::Bubble::exeSink(void)   @ 0x71000e3518
   static  : "probably a scene sink"     ← spectrIDA guessed
   runtime : candidate_crash             ← phantomrt ran it and watched
             read_unmapped @ wild addr, input-controlled → go look
```

---

## What it is

Static reverse engineering tells you what the code **is**. It never tells you what it **does**
when you actually pull the trigger. phantomrt pulls the trigger.

It takes a function spectrIDA already found and named, and does one of three haunted things to it:

- **Emulate** it (Unicorn, no OS, any arch) — works on binaries you *can't* even run, like a
  Switch `.nso` or an Android `.so`. Reports `candidate_crash` / `needs_state` / `clean`.
- **Live-instrument** it (Frida) — attach to the *running* process, hook the function, watch
  real arguments and return values fly by. For the functions emulation shrugs at with
  `needs_state`, because the real process actually has the state.
- **Fuzz** it — feed it mutated inputs (seeds you bring, or assets it *carves out of the binary
  itself*), catch the crashes, keep the reproducing input.

Then it stamps the verdict onto the Neo4j graph as `dyn_*` properties, so the next agent that
reads the function sees the name **and** the behavior in one place. That's the whole trick: the
map and the walk-through, on the same node.

```
spectrIDA (names it)  ─▶  phantomrt (runs it)  ─▶  graph (dyn_status/crash/live args)  ─▶  the agent reasons
```

There's also a self-improving world-model in here that learns to predict how functions react —
it's real, it's honest about what it hasn't learned yet, and it's the research corner. The
load-bearing part is the emulation + honest verdicts.

---

## It talks to Claude (or anything that speaks MCP)

Installed as the `spectrida[atlas]` extra, phantomrt lights up five tools on spectrIDA's MCP
server — so the reverse-engineer isn't you, it's the agent driving the graph:

| tool | what the ghost does |
|------|---------------------|
| `emulate_function` | runs one function, honest verdict onto the graph |
| `hunt_crashes` | fuzzes it (bring seeds, or it carves them from the binary), records repro inputs |
| `live_trace` | attaches Frida, captures real args/returns |
| `dynamic_overview` | "what have we actually run, and what crashed?" |
| `risk_functions` | ranked by functions that *actually faulted* — not a static hunch |

The reasoning corner is whatever LLM is driving. phantomrt just makes sure it has real runtime
facts to reason with instead of vibes.

---

## Install

```
pip install phantomrt              # emulation + fuzzing + the world model
pip install "phantomrt[live]"      # + Frida live instrumentation
```

Or, the way you probably want it — as spectrIDA's dynamic extra:

```
pip install "spectrida[atlas]"
```

Heads up: it drags in `torch` and `unicorn`. It's the heavy extra on purpose — spectrIDA's base
install stays light and pulls none of this unless you ask for it.

---

## Honest ghost disclaimer

Because pretending is worse than admitting:

- **It is not a magic bug oracle.** It's a solid dynamic layer. On a short run it found the two
  planted bugs in a toy target instantly; on a real hardened target it'll tell you honestly when
  it's stuck (`needs_state`), when a crash is only a *candidate* (verify the pointer's actually
  input-controlled), and when it just needs more time and better seeds — same as every real
  fuzzer.
- **Switch binaries don't run live.** Frida needs something it can actually launch. Emulation is
  your only door there, and it'll say `needs_state` a lot. That's honest, not broken.
- **It's `0.1.0`.** Alpha. Genuinely works, genuinely early. The machinery is real; the polish is
  in progress.

It doesn't do everything. It does the annoying part — *seeing what the code does when it runs* —
and it doesn't lie to you about the rest.

**No cloud. No telemetry. Runs entirely on your machine.**

---

*The static ghost names your functions. This one makes them confess.* 👻
