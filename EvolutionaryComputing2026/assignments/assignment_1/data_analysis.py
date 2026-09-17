import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_fixed.csv")

for run in df["run"].unique():
    run_data = df[df["run"] == run]
    plt.plot(
        run_data["generation"],
        run_data["best_so_far"],
        label=f"Run {run}"
    )

plt.xlabel("Generation")
plt.ylabel("Best-so-far fitness")
plt.title("Fixed Mutation Rate")
plt.legend()
plt.savefig("fixed.png", dpi=300, bbox_inches="tight")