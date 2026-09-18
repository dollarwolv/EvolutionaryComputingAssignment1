import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import math
import re

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
        visualize_decrease_schedules(100)

    elif args.plot_results:
        min_value = get_best_run()
        value_to_compare = min_value * 1.1

        create_averaged_plot(value_to_compare)

    elif args.compare_results:
        min_value = get_best_run()
        value_to_compare = min_value * 1.1

        enabled_args = [name for name, value in vars(args).items() if value is True]
        enabled_args.remove("compare_results")

        create_averaged_plot(value_to_compare, enabled_args)


def pick_mutation_rate(
    x,
    n_generations,
    adaptive=False,
    schedule="linear",
    start=0.5,
    end=0.01,
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
        path = Path("assignments") / "assignment_1" / "outputs" / file
        df = pd.read_csv(path)
        dataframes.append(df)

    big_df = pd.concat(dataframes, ignore_index=True)

    min_value = big_df["best_so_far"].min()

    return min_value


def create_averaged_plot(threshold: float, comparison: None | list[str] = None):

    if comparison is False:
        files = [
            "dataset_exponential.csv",
            "dataset_logarithmic.csv",
            "dataset_linear.csv",
            "dataset_fixed.csv",
        ]

    else:
        files = [f"dataset_{s.strip().lower()}.csv" for s in comparison]

    for file in files:
        path = Path("assignments") / "assignment_1" / "outputs" / file
        df = pd.read_csv(path)

        stats = df.groupby("generation")["best_so_far"].agg(["mean", "std"])

        plot_type = re.search(r"(?<=_)[^.]+(?=\.)", file)

        plt.plot(stats.index, stats["mean"], label=f"{plot_type.group().capitalize()}")

        if comparison is False:
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
        y=threshold, color="red", linestyle=":", label=f"Target: {threshold:.2f}"
    )

    plt.legend()

    # Create the output folder if it does not exist.
    output_dir = Path("assignments") / "assignment_1" / "outputs" / "averages"
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


def visualize_decrease_schedules(n_generations):

    schedules = ["exponential", "logarithmic", "linear", "fixed"]

    for schedule in schedules:
        if schedule == "fixed":
            adaptive = False
            y_values = [0.5 for _ in range(n_generations)]
        else:
            adaptive = True
            y_values = [
                pick_mutation_rate(x, n_generations, adaptive, schedule)
                for x in range(n_generations)
            ]

        x_values = [x for x in range(n_generations)]
        plt.plot(x_values, y_values, label=schedule)
        plt.xlabel("Generation")
        plt.ylabel("Mutation rate")
        plt.legend()

    output_dir = Path("assignments") / "assignment_1" / "outputs" / "averages"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_dir / "mutation_rates.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
