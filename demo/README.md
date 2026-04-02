# GCS-MMS Demo

This demo solves motion planning as a single integrated one-phase MIOCP relaxation:

- graph flow variables `y_uv`, `p_v`
- multiple-shooting trajectory variables `s_v^-`, `s_v^+`, `w_v`, `Delta_v`
- interface variables `z_uv`
- local cost epigraphs `rho_v`

The runtime solver is:

- `IPOPT` on the integrated continuous relaxation
- optional fixed-path NLP polish when the extracted relaxed path is fractional or discontinuous

## Quick Start

From the repository root:

```bash
python3 -m venv demo/.venv
source demo/.venv/bin/activate
pip install -r requirements.txt
python3 demo/main_demo.py --scenario simple
```

If you only want to inspect the available configs without activating the demo environment:

```bash
python3 demo/main_demo.py --list-scenarios
python3 demo/main_demo.py --list-presets
```

## Structure

The code is organized so geometry, scenarios, and solver assembly are separate:

- `problem_data.py`: geometry presets for each environment/problem
- `config.yaml`: scenario list and parameter overrides
- `app_config.py`: config loading and scenario resolution
- `scenario_builder.py`: builds environment, regions, graph, dynamics, optimizer
- `optimizer.py`: integrated solver and fixed-path polish
- `experiments.py`: scenario runner and result saving
- `main_demo.py`: CLI entry point

## Run

```bash
cd demo
python main_demo.py
python main_demo.py --scenario simple
python main_demo.py --list-scenarios
python main_demo.py --list-presets
python main_demo.py --visualize-only --scenario medium
python main_demo.py --quick-test
```

## Configure Parameters

Global defaults live in `config.yaml`:

```yaml
cost:
  a: 1.0
  w_L: 1.0
  w_E: 1.0

shooting:
  n_integration_steps: 20
  n_mesh_points: 10
  safety_margin: 0.02

optimizer:
  ipopt:
    max_iter: 3000
    tol: 1.0e-6
    print_level: 0
```

Scenario-specific changes should go under `scenarios.<name>.overrides`:

```yaml
scenarios:
  high_resolution:
    name: "High Resolution"
    description: "All regions with increased mesh points"
    active_regions: "all"
    overrides:
      shooting:
        n_mesh_points: 15
      optimizer:
        ipopt:
          max_iter: 5000
```

You can also override `start_state`, `goal_state`, and `problem_preset` per scenario.

## Add a New Environment

1. Add a new preset entry in `problem_data.py`.
2. Define:
   - `workspace_vertices`
   - `obstacle_vertices`
   - `region_vertices`
   - `default_start_state`
   - `default_goal_state`
   - `region_buffer`
3. Reference that preset from `config.yaml`:

```yaml
problem:
  default_preset: "my_new_problem"
```

or per scenario:

```yaml
scenarios:
  my_case:
    problem_preset: "my_new_problem"
    active_regions: "all"
```

That is enough for the CLI and experiment runner to pick it up.

## Outputs

Results are saved under `results/`:

- `{scenario}_result.png`
- `{scenario}_animation.gif`
- `{scenario}_summary.json`
- `benchmark_summary.json`
