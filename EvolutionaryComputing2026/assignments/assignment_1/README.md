# Evolutionary Computing — Assignment 1

**Team:** Group 51
**Members:** Kaj Westendorp, Luka Vidovic, Sardar Sardarov, Justin Dotzlaw

This project compares fixed, linear, logarithmic, and exponential mutation-rate schedules for evolving robot morphologies. It also includes a random-search baseline.

## Setup

Run all commands from the `EvolutionaryComputing2026` directory. Install the project and its dependencies with:

```bash
uv sync
```

## Run all experiments

```bash
uv run python assignments/assignment_1/our_answer.py --run-all
```

This runs all four mutation schedules and the random-search baseline, then creates the comparison plots and runs the significance test. The full experiment may take some time. You may also adjust the number of runs in experiment_settings.py to run for less runs/generations. However, running the code again will override the data and plots from previous runs, so make sure to save/inspect those first.

Results are saved in `assignments/assignment_1/outputs/`.

## Run one condition

```bash
# Fixed mutation rate
uv run python assignments/assignment_1/our_answer.py

# Decreasing mutation rates
uv run python assignments/assignment_1/our_answer.py --adaptive --linear
uv run python assignments/assignment_1/our_answer.py --adaptive --logarithmic
uv run python assignments/assignment_1/our_answer.py --adaptive --exponential

# Random-search baseline
uv run python assignments/assignment_1/our_answer.py --random
```

## Analyze existing results

```bash
# Create the averaged, zoomed, and threshold-crossing plots
uv run python assignments/assignment_1/data_analysis.py --plot-results

# Plot the mutation-rate schedules
uv run python assignments/assignment_1/data_analysis.py --plot-mutation-rates

# Compare selected conditions (choose two or more condition flags)
uv run python assignments/assignment_1/data_analysis.py --compare-results --linear --fixed

# Run the multivariate log-rank test
uv run python assignments/assignment_1/data_analysis.py --run-significance-test
```
