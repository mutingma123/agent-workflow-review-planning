# Agent workflow review planning

Research code for *Fewer Task Errors, More Systemic Failures in AI Agent Workflows* by Muting Ma and Jiaqi Gong. The two-stage model assigns a model and a human review decision to each active task. On each sampled workflow, the experiment follows errors across handoffs and measures how often at least a quarter of the reference workflow is released wrong.

## Run the experiments

Run the commands from the repository root in a Python 3.10 or later environment on Linux or macOS. Full optimization requires a Gurobi license suitable for the model sizes. The campaign reported in the paper used Gurobi 13.0.3, 11 worker processes and two solver threads per worker.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s Verification -p test_code_only.py
python Implementation/kp_campaign.py plan --workers 11 --threads 2
python Implementation/kp_campaign.py preflight
python Implementation/kp_campaign.py launch --workers 11 --threads 2
python Implementation/kp_analysis.py
```

The plan command lists the jobs and computes the time allocation. Preflight solves small instances. Launch generates each workflow from its seed, estimates available worker capacity, and records jobs locally under `runs/`. Repeating the launch command resumes an interrupted campaign within its original deadline. Use `python Implementation/kp_campaign.py status` for progress. The analysis writes tables, figures and protocol checks from the local job records.

## Source

| File | Purpose |
| --- | --- |
| `Implementation/kp_model.py` | Workflow generation, scenario sampling, error propagation and scoring |
| `Implementation/kp_cuts.py` | Closed-form scenario cuts and feasible local search |
| `Implementation/kp_solve.py` | Routing, mean-value, extensive-form and Benders solvers |
| `Implementation/kp_blocks.py` | Experiment settings, seeds, benchmarks and warm starts |
| `Implementation/kp_campaign.py` | Parallel scheduling, resource limits and restart |
| `Implementation/kp_analysis.py` | Paired comparisons, equivalence tests, solver diagnostics and plots |
| `Notebooks/formulation_runs.ipynb` | Regenerates the implementation and runs checks. Enable its solver flags to launch optimization. |
| `Verification/test_code_only.py` | Checks model logic, cuts, budgets and scheduler behavior. The optional historical pilot replay reads a user-supplied JSONL file named by the PILOT_RESULTS environment variable. |

The [MIT license](LICENSE) covers this repository's source. Gurobi is distributed separately under its own license.
