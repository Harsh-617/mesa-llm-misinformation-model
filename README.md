# Misinformation Spread Simulation — Mesa-LLM vs Rule-Based Agents

A [mesa-llm](https://github.com/projectmesa/mesa-llm) agent-based model simulating the spread of misinformation through a small community. LLM-powered agents with unique personas reason about whether to spread or challenge a rumor about water contamination. Includes a rule-based comparison model to demonstrate the value of LLM-driven agent behavior.

Built as part of my contribution to [Project Mesa](https://github.com/projectmesa) for **GSoC 2026**.

## Features

- **12 LLM-powered agents** with unique personas (teacher, shopkeeper, journalist, doctor, etc.)
- **ReAct reasoning** for step-by-step decision making via `mesa-llm`
- **Custom tools**: `check_neighbors`, `spread_rumor`, `challenge_rumor`, `update_belief`
- **Spatial grid** (5×5 Moore neighborhood) where agents interact with nearby neighbors
- **Memory system** tracking inter-agent communication via `send_message`
- **Data collection** and matplotlib visualization of belief dynamics over time
- **Rule-based comparison model** for benchmarking LLM vs traditional ABM approaches

## Project Structure

```
mesa-llm-misinformation-model/
├── misinformation_model/
│   ├── __init__.py
│   ├── agents.py            # CitizenAgent — LLM-powered agent with ReAct reasoning
│   ├── model.py              # MisinformationModel — main simulation with 12 agents
│   ├── tools.py              # Custom tools: check_neighbors, spread/challenge_rumor, update_belief
│   └── rule_based.py         # RuleBasedAgent & RuleBasedModel — non-LLM comparison
├── run.py                    # Run full LLM simulation (12 agents, 5 steps)
├── run_small.py              # Run small LLM simulation (4 agents, 3 steps)
├── run_comparison.py         # Run rule-based simulation (12 agents, 10 steps, no LLM)
├── requirements.txt          # Python dependencies
├── LICENSE                   # Apache 2.0
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Harsh-617/mesa-llm-misinformation-model.git
cd mesa-llm-misinformation-model
```

### 2. Create a conda environment

```bash
conda create -n mesa-misinfo python=3.12 -y
conda activate mesa-misinfo
```

### 3. Install dependencies

```bash
pip install mesa mesa-llm[ollama] matplotlib python-dotenv
```

### 4. Set up Ollama (for LLM simulation only)

Install [Ollama](https://ollama.com/) and pull a model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running before starting the LLM simulation:

```bash
ollama serve
```

> The rule-based simulation (`run_comparison.py`) does **not** require Ollama or any LLM.

## Usage

### LLM Simulation (full — 12 agents, 5 steps)

```bash
python run.py
```

Runs all 12 LLM-powered agents for 5 steps. Each agent reasons through the ReAct loop, calling tools to check neighbors, spread or challenge the rumor, and update their belief. Saves plot to `results.png`.

> Expect ~25–30 minutes with a 3B model on CPU.

### LLM Simulation (small — 4 agents, 3 steps)

```bash
python run_small.py
```

A faster variant with 4 agents and 3 steps for quick iteration. Saves plot to `results_small.png`.

### Rule-Based Comparison (no LLM)

```bash
python run_comparison.py
```

Runs 12 rule-based agents for 10 steps with deterministic influence rules. Completes in seconds. Saves plot to `results_rulebased.png` and prints a comparison summary.

## How It Works

### The Rumor Scenario

A rumor spreads through the community:

> *"The town's water supply has been contaminated with dangerous chemicals from the nearby factory."*

Each agent starts with an **initial stance** (believer, skeptic, or neutral) and a **belief score** (0.0–1.0). Over time, agents interact with neighbors and their beliefs evolve.

### LLM Agent Reasoning (ReAct Loop)

Each step, every `CitizenAgent` receives a structured prompt and reasons through a ReAct loop:

1. **Check neighbors** — call `check_neighbors` to see nearby agents and their stances
2. **Communicate** — call `spread_rumor` (if belief > 0.5) or `challenge_rumor` (if belief ≤ 0.5) targeting a neighbor
3. **Update belief** — adjust belief score based on the action taken

The LLM decides *which* neighbor to target and *how* to reason about the rumor based on its persona.

### Tool-Based Agent–Environment Interaction

Agents interact with the simulation through four custom tools built with `mesa-llm`'s `@tool` decorator:

| Tool | Description |
|------|-------------|
| `check_neighbors` | Scans the Moore neighborhood for nearby agents and their stances |
| `spread_rumor` | Sends the rumor to a target agent via `send_message` |
| `challenge_rumor` | Sends a counter-argument to a target agent via `send_message` |
| `update_belief` | Updates the agent's belief score and recalculates stance |

### Belief Dynamics

- **Belief score** is a float in [0.0, 1.0] representing confidence in the rumor
- **Stance thresholds**: > 0.7 → believer, < 0.3 → skeptic, otherwise → neutral
- A programmatic fallback ensures belief updates happen even when the LLM skips the `update_belief` tool call

### Rule-Based vs LLM Approach

| Aspect | Rule-Based | LLM-Based |
|--------|-----------|-----------|
| Decision logic | Fixed mathematical rules | Natural language reasoning |
| Neighbor influence | Believer: +0.1×(1−score), Skeptic: −0.1×score | LLM chooses to spread or challenge |
| Personality | None (persona param unused) | Unique persona shapes reasoning |
| Determinism | Mostly deterministic | Non-deterministic |
| Speed | Seconds | 25+ minutes |
| Use case | Parameter sweeps, baselines | Emergent behavior, rich dynamics |

## Key Findings

Working with `mesa-llm` and small local LLMs revealed several practical insights:

- **Small LLMs (3B) struggle with multi-step tool calling** — agents frequently skip tools or hallucinate agent IDs, requiring programmatic fallbacks
- **Built-in mesa-llm tools can confuse agents** — movement tools (`move_one_step`, `teleport_to_location`, `speak_to`) had to be manually removed to prevent the LLM from calling them instead of custom tools
- **ReAct loop executes only one tool per step** — multi-tool plans are generated but only the first tool is reliably called
- **Tool argument type coercion is missing** — `update_belief(new_score="0.85")` passes a string instead of float, requiring manual `float()` conversion in the tool
- **Rule-based runs in seconds vs 25+ minutes for LLM** — critical for iteration and debugging
- **Despite limitations, LLM agents show personality-driven reasoning** — a skeptical doctor reasons differently than an anxious shopkeeper, even with a 3B model

## Contributions to mesa-llm

As part of this GSoC effort, I contributed 5 pull requests to [mesa-llm](https://github.com/projectmesa/mesa-llm) (4 merged, 1 open):

| PR | Description | Status |
|----|-------------|--------|
| [#89](https://github.com/projectmesa/mesa-llm/pull/89) | Validate `llm_model` format to catch misconfigured model strings early | Merged |
| [#130](https://github.com/projectmesa/mesa-llm/pull/130) | Fix `ignore_agent` handling in docstring validation for tool parameters | Merged |
| [#157](https://github.com/projectmesa/mesa-llm/pull/157) | Fix JSON serialization in `send_message` for communication history | Merged |
| [#160](https://github.com/projectmesa/mesa-llm/pull/160) | Fix ReWOO replay mutation bug where plans were modified in place | Merged |
| [#194](https://github.com/projectmesa/mesa-llm/pull/194) | Fix `get_communication_history` rendering for agent memory | Open |

## Built With

- [Mesa 3.x](https://github.com/projectmesa/mesa) — Agent-based modeling framework
- [mesa-llm 0.3.0](https://github.com/projectmesa/mesa-llm) — LLM integration for Mesa agents
- [Ollama](https://ollama.com/) with `llama3.2:3b` — Local LLM inference
- [Python 3.12](https://www.python.org/)
- [matplotlib](https://matplotlib.org/) — Visualization

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
