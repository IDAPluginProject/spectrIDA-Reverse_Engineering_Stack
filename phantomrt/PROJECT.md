# Project Atlas — Dynamic Reverse-Engineering Brain

> Static tools tell you what code **is**. Atlas tells you what happens when it **runs**.

Atlas is the *dynamic* corner of a reverse-engineering system. It takes a function
that a static tool (spectrIDA/IDA) has found and named, actually **executes** it —
by CPU emulation or by live instrumentation — and reports what really happens:
does it run, what does it do to its inputs, does it crash, what state does it need.
Those runtime facts get written back onto the shared function graph so an agent
reasoning over the code sees behavior, not just structure.

---

## The "RE Triforce"

The goal is a reverse-engineering system where three corners each cover the
others' blind spots, connected through one shared function graph (Neo4j):

| Corner | Who | Job | Blind spot it has |
|--------|-----|-----|-------------------|
| **Static** | spectrIDA + `spectrida-re` model | Find & **name** functions, build the call graph, decompile | Never runs anything — can't tell what actually happens |
| **Dynamic** | **Atlas** (this project) | **Run** functions, report behavior/crashes/coverage, annotate the graph | Can't reason about intent; stalls at hard gates |
| **Reasoning** | The agent driving the MCP server (Claude / MiMo / etc.) | Read names + runtime facts, reason, decide, crack gates | Ungrounded without the other two feeding it real facts |

It's a **triangle, not a line**: the reasoning agent reads Atlas's honest
`needs_state` / `crash` / `coverage` facts *and* spectrIDA's names, reasons about
them, and can feed Atlas back a key (a seed, a stub, a valid input) to get past
where it stalled. No corner does the job alone.

> Correction baked in from experience: the reasoning corner is **the agent itself**
> (whatever LLM drives the MCP server), **not** a bundled model. `spectrida-re` is a
> *naming* model and belongs to the static corner — it names, it does not reason.

---

## What Atlas actually does

Atlas has **two execution modes**, because "run the function" means different
things depending on whether the target can run on this machine.

### 1. Emulation mode (`atlas/analysis/unicorn_harness.py`, `spectrida_bridge.py`)
CPU emulation with **Unicorn** — no OS, any architecture. Works on binaries you
**cannot** run natively (ARM64 Switch NSO, Android `.so`) as well as ones you can.

- Loads real function bytes + arch via spectrIDA's `FormatHandler` plugins (PE / NSO
  / ELF / .so) — no re-implementing loaders.
- **Chain emulation:** maps the whole image so internal calls land on real code;
  **stubs** out-of-chain calls (return 0) and skips syscalls, so only real data
  faults count.
- **Honest classification** (`EmuResult.status`):
  - `crash` — fault on a *wild* address → candidate bug (verify it's input-controlled).
  - `needs_state` — fault on a *near-null* address → uninitialized global/`this`;
    the function needs live engine state we don't have. **Not a fake crash.**
  - `clean` / `inconclusive`.

**Strength:** the only option for non-runnable targets (Switch). **Weakness:** state-
entangled functions (game engine, Bun runtime) mostly come back `needs_state` — honest,
but low-value. Best on self-contained, input-facing functions (parsers, decoders, crypto).

### 2. Live mode (`atlas/analysis/frida_live.py`) — *the useful one for runnable targets*
**Frida** instrumentation of the **running** process. The live process *has* all the
real state emulation lacks, so `needs_state` functions actually run.

- Attach/spawn the target, resolve functions by **RVA** (`module.base + offset`) —
  the same addresses the spectrIDA graph stores.
- **Trace** what really executes: real args, return values, call counts. *(validated)*
- **In-process fuzz**: call a function with mutated inputs using the live process
  state; **spawn-per-crash** — run fast while the process survives, on a crash record
  the reproducing input and respawn to continue. *(validated)*
- Annotate the graph with `dyn_live_*` facts.

**Proven concretely (`tests/test_frida_live.py`):** a test function gated on a global
(`if (g_secret != 0x1337) return -1`) returns `-1` / `needs_state` under emulation but
runs for real **live** (`check("hello") → 5`), reaching the vulnerable code past the
gate — and the live fuzzer detects the crash + reproduces the input. That gap *is* why
live mode exists.

**Honest limits:**
- Target must actually run on this machine → **not** Switch NSO (Frida isn't magic;
  emulation stays the only option there).
- **Crash detection + reproducing input are solid; the exact fault *address* is
  best-effort on Windows** — an unhandled access-violation triggers Windows Error
  Reporting and leaves the JS runtime in a bad state mid-fault, so the address often
  comes back `detached`. The reproducing input is the real artifact (re-run under a
  debugger for the address, same as AFL).
- Crash-restart is spawn-per-crash (no `fork()` on Windows); the throughput upgrade if
  ever needed is snapshot-restore (WinAFL-style), not a fork-server.
- Packed runtimes (Bun): native functions are the *runtime*, not the app's JS logic.

### 3. The self-improving world-model (`atlas/`, `atlas/training/`)
The original Atlas: a genuine world-model (VAE encoder → Neural-ODE/Euler dynamics →
decoder), curiosity-driven exploration, need-driven growth, and continual learning
(EWC + replay). It **predicts** how a target reacts to an input, **learns** from the
real outcome (surprise = prediction error), and **grows itself** only when it hits a
learnable-but-underfit wall — with a noisy-TV guard so it doesn't chase unlearnable
randomness.

The headline honesty metric is **held-out prediction error**: it's tested on
inputs/functions it never trained on. Rising held-out accuracy = it *understood*
(generalized); flat held-out while training error falls = it *memorized* — and the
metric exposes that instead of hiding it.

> Honest status: the world-model proved itself on **toy fuzzing** (found 2 planted
> bugs in seconds, held-out error 0.003). In the RE context so far, the *emulation +
> honest classification* is what's load-bearing; the learning finds real signal where
> execution is cheap, deterministic, and repeatable (live mode, self-contained parsers).

---

## Repository map

```
atlas/
  core/            world-model: encoder, dynamics (Neural-ODE), decoder, surprise
  training/        self_train (closed learning loop), growth (need-driven), continual (EWC/replay)
  environments/    base, grid_world, physics_2d, vm_world (isolated-WSL learner)
  agents/          command_space (unrestricted proposer + compositional embedding)
  analysis/
    unicorn_harness.py   Unicorn emulation: single-fn + chain mode + stubbing
    spectrida_bridge.py  load real binary bytes+arch via spectrIDA FormatHandler
    binary_fuzz.py       coverage-guided input fuzzer (WSL-compiled targets)
    graph_annotator.py   write dyn_/atlas_ runtime facts onto graph Function nodes
    frida_live.py        LIVE instrumentation backend (Frida)
    re_triangle.py       orchestrator: spectrIDA fn -> Atlas verdict -> graph
  vm/              WslVM: isolated throwaway WSL distro (for the VM learner)
run_atlas.py       self-directed VM learner
run_fuzz.py        coverage-guided crash hunt on a binary
run_learn.py       learn/improve on a target's functions   (planned)
tests/             65+ tests (emulation on real arm64+x86-64, growth, annotator, ...)
archive/           the original FAKE "self-training" scripts, kept as a cautionary tale
```

## Status at a glance
- ✅ Emulation core (Unicorn, chain+stub, honest `needs_state`) — tested on real
  ARM64 (`main.nso`) and x86-64 (`droid.exe`).
- ✅ spectrIDA bridge (real bytes+arch via FormatHandler), graph annotation (live-tested).
- ✅ Triforce wired end-to-end on a real Odyssey function.
- ✅ Live mode (Frida) — validated: live state passes gates emulation can't; trace +
  in-process fuzz + spawn-per-crash all work (fault address best-effort on Windows).
- ✅ World-model self-training — real closed loop, honest metrics; proven on toys.
- 🔜 Atlas Live backend hardening (crash-restart), pip packaging, spectrIDA MCP tools.

## Guiding principle
**No theater.** The archived scripts once produced impressive logs from disconnected
learning loops. Everything here reports what actually happened — including "this
faulted because it needs live state," "this is a candidate, verify it," and "held-out
accuracy is flat, so it's memorizing." Honest and useful beats impressive and fake.
