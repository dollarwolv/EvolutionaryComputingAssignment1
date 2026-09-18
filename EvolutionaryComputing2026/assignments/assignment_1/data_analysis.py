import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import math
import re
from experiment_settings import (
    NUM_GENERATIONS,
    FIXED_MUTATION_RATE,
    START_MUTATION_RATE,
    END_MUTATION_RATE,
    THRESHOLD_MULTIPLIER,
    ZOOM_START_GENERATION,
    ZOOM_FITNESS_MARGIN,
)

HERE = Path(__file__).parent


def main():

    parser = argparse.ArgumentParser(
        description="Plot type",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot individual runs for the selected mutation schedule",
    )

    parser.add_argument(
        "--plot-mutation-rates",
        action="store_true",
        help="Plot different mutation rates",
    )

    parser.add_argument(
        "--plot-results",
        action="store_true",
        help="Plot the average of all runs for each mutation schedule",
    )

    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Plot fixed mutation rate",
    )

    parser.add_argument(
        "--linear",
        action="store_true",
        help="Decreasing mutation rate: Linearly decreasing.",
    )

    parser.add_argument(
        "--logarithmic",
        action="store_true",
        help="Decreasing mutation rate: Logarithmically decreasing.",
    )

    parser.add_argument(
        "--exponential",
        action="store_true",
        help="Decreasing mutation rate: Exponentially decreasing.",
    )

    parser.add_argument(
        "--compare-results",
        action="store_true",
        help="Plot the average of all runs for each mutation schedule",
    )

    args = parser.parse_args()

    if args.plot:
        plot_type = None

        if args.fixed:
            plot_type = "fixed"
        elif args.linear:
            plot_type = "linear"
        elif args.logarithmic:
            plot_type = "logarithmic"
        elif args.exponential:
            plot_type = "exponential"

        create_plot(plot_type)

    elif args.plot_mutation_rates:
        visualize_decrease_schedules(NUM_GENERATIONS)

    elif args.plot_results:
        min_value = get_best_run()
        value_to_compare = min_value * THRESHOLD_MULTIPLIER

        create_averaged_plot(value_to_compare)

    elif args.compare_results:
        min_value = get_best_run()
        value_to_compare = min_value * THRESHOLD_MULTIPLIER

        enabled_args = [name for name, value in vars(args).items() if value is True]
        enabled_args.remove("compare_results")

        create_averaged_plot(value_to_compare, enabled_args)


def pick_mutation_rate(
    x,
    n_generations,
    adaptive=False,
    schedule="linear",
    start=START_MUTATION_RATE,
    end=END_MUTATION_RATE,
):
    # Without adaptation, the rate stays constant.
    if not adaptive:
        return start

    # With adaptation, choose how the rate decreases.
    if schedule == "linear":
        progress = x / n_generations
    elif schedule == "logarithmic":
        progress = math.log1p(x) / math.log1p(n_generations)
    elif schedule == "exponential":
        # Exponential decay requires both endpoints to be positive.
        return start * (end / start) ** (x / n_generations)

    else:
        raise ValueError("schedule must be 'linear' or 'logarithmic'.")

    return start + (end - start) * progress


def create_plot(plot_type):
    if not plot_type:
        raise ValueError("no plot type given")

    df = pd.read_csv(HERE / "outputs" / f"dataset_{plot_type}.csv")

    for run in df["run"].unique():
        run_data = df[df["run"] == run]
        plt.plot(run_data["generation"], run_data["best_so_far"], label=f"Run {run}")

    plt.xlabel("Generation")
    plt.ylabel("Best-so-far fitness")
    plt.title(f"{plot_type.capitalize()} Mutation Rate")
    plt.savefig(HERE / "outputs" / f"{plot_type}.png", dpi=300, bbox_inches="tight")


def get_best_run():
    files = [
        "dataset_exponential.csv",
        "dataset_logarithmic.csv",
        "dataset_linear.csv",
        "dataset_fixed.csv",
    ]

    dataframes = []

    for file in files:
        path = HERE / "outputs" / file
        df = pd.read_csv(path)
        dataframes.append(df)

    big_df = pd.concat(dataframes, ignore_index=True)

    min_value = big_df["best_so_far"].min()

    return min_value


def create_averaged_plot(threshold: float, comparison: None | list[str] = None):

    if not comparison:
        files = [
            "dataset_exponential.csv",
            "dataset_logarithmic.csv",
            "dataset_linear.csv",
            "dataset_fixed.csv",
        ]

    else:
        files = [f"dataset_{s.strip().lower()}.csv" for s in comparison]

    last_generation = 0
    crossings = {}
    for file in files:
        path = HERE / "outputs" / file
        df = pd.read_csv(path)

        stats = df.groupby("generation")["best_so_far"].agg(["mean", "std"])
        last_generation = max(last_generation, stats.index.max())

        plot_type = re.search(r"(?<=_)[^.]+(?=\.)", file)
        # Find where the average curve first reaches the shared threshold.
        reached = stats[stats["mean"] <= threshold]
        crossings[plot_type.group().capitalize()] = (
            int(reached.index[0]) if not reached.empty else None
        )

        plt.plot(stats.index, stats["mean"], label=f"{plot_type.group().capitalize()}")

        if not comparison:
            plt.title("Fittest individual per generation, averaged by run")

        else:
            plt.title(" vs. ".join(s.strip().capitalize() for s in comparison))

        plt.fill_between(
            stats.index,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.2,
            label=f"{plot_type.group().capitalize()} ±1 standard deviation",
        )

    plt.axhline(
        y=threshold,
        color="red",
        linestyle=":",
        label=f"Target: {threshold:.2f} ({(THRESHOLD_MULTIPLIER - 1) * 100:.0f}% above the best value ever reached)",
    )

    plt.legend()

    # Create the output folder if it does not exist.
    output_dir = HERE / "outputs" / "averages"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        (
            output_dir / "average_best_runs.png"
            if not comparison
            else output_dir
            / ("comparison_" + "_".join(s.strip().lower() for s in comparison) + ".png")
        ),
        dpi=300,
        bbox_inches="tight",
    )

    # Save the same curves again, zoomed around the threshold in later generations.
    # Use the plotted data's endpoint, so older datasets also display correctly.
    plt.xlim(min(ZOOM_START_GENERATION, last_generation / 2), last_generation)
    plt.ylim(threshold - ZOOM_FITNESS_MARGIN, threshold + ZOOM_FITNESS_MARGIN)
    plt.xlabel("Generation")
    plt.ylabel("Mean best-so-far fitness")
    plt.title("Convergence near the threshold (zoomed)")
    plt.legend(fontsize=7)
    zoom_name = (
        "average_best_runs_zoomed"
        if not comparison
        else "comparison_" + "_".join(s.strip().lower() for s in comparison) + "_zoomed"
    )
    plt.savefig(output_dir / f"{zoom_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    create_threshold_bar_plot(
        crossings,
        threshold,
        output_dir / f"{zoom_name.replace('_zoomed', '')}_threshold.png",
    )


def create_threshold_bar_plot(crossings: dict, threshold: float, output_path: Path):
    fig, ax = plt.subplots()

    crossings = dict(
        sorted(
            crossings.items(),
            key=lambda item: float("inf") if item[1] is None else item[1],
        )
    )

    for position, (schedule, generation) in enumerate(crossings.items()):
        if generation is None:
            # Leave no bar rather than inventing a crossing time.
            ax.text(
                position,
                0.03,
                "Not reached",
                ha="center",
                transform=ax.get_xaxis_transform(),
            )
        else:
            ax.bar(position, generation, color=f"C{position}")
            ax.annotate(
                str(generation),
                (position, generation),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
            )
    ax.set_xticks(range(len(crossings)), labels=list(crossings))
    ax.set_xlim(-0.5, len(crossings) - 0.5)
    highest = max((g for g in crossings.values() if g is not None), default=0)
    ax.set_ylim(0, max(1, highest) * 1.15)
    ax.set_ylabel("First generation at or below threshold")
    ax.set_title(f"Average best-so-far fitness reaches {threshold:.3f}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_decrease_schedules(n_generations):

    schedules = ["exponential", "logarithmic", "linear", "fixed"]

    for schedule in schedules:
        if schedule == "fixed":
            adaptive = False
            y_values = [FIXED_MUTATION_RATE for _ in range(n_generations + 1)]
        else:
            adaptive = True
            y_values = [
                pick_mutation_rate(x, n_generations, adaptive, schedule)
                for x in range(n_generations + 1)
            ]

        x_values = [x for x in range(n_generations + 1)]
        plt.plot(x_values, y_values, label=schedule)
        plt.xlabel("Generation")
        plt.ylabel("Mutation rate")
        plt.legend()

    output_dir = HERE / "outputs" / "averages"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_dir / "mutation_rates.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
