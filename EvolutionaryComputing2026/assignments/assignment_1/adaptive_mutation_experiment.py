"""Adaptive vs. fixed mutation rate experiment for Assignment 1.

Research question:
    Does a mutation rate that decreases over generations
    (exploration -> exploitation) improve convergence speed compared to a
    fixed rate, for tree-based genomes?

Everything (population size, selection, crossover, the mutation OPERATOR
itself, budget, seed) is held identical between the two runs. The ONLY
thing that differs is the schedule that produces the per-generation
mutation probability:

    fixed    : constant rate for the whole run
    adaptive : linear decay from --initial-rate down to --final-rate

Run with e.g.:
    python adaptive_mutation_experiment.py --variant fixed    --seed 42
    python adaptive_mutation_experiment.py --variant adaptive --seed 42
"""


import argparse
import copy
import csv
import random
from pathlib import Path
from typing import Literal

import numpy as np

from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,
    crossover_subtree,
    mutate_replace_node,
    random_tree,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from tree_edit_distance import mean_plus_std_tree_edit_distance

# Reuse the template's own target-loading logic.
from A1_template_2026 import load_targets, TARGET_DIR

# --- CLI ------------------------------------------------------------------ #

parser = argparse.ArgumentParser(description="Adaptive vs. fixed mutation rate experiment")
parser.add_argument(
    "--variant",
    choices=["fixed", "adaptive"],
    required=True,
    help="fixed = constant mutation rate. adaptive = linear decay over generations.",
)
parser.add_argument("--pop", type=int, default=100, help="Population size")
parser.add_argument("--budget", type=int, default=100, help="Number of generations")
parser.add_argument("--max-modules", type=int, default=20, help="Module budget")
parser.add_argument(
    "--initial-rate",
    type=float,
    default=0.6,
    help="Mutation probability at generation 0. Used directly by 'fixed' for "
    "the whole run, and as the START of the decay for 'adaptive'.",
)
parser.add_argument(
    "--final-rate",
    type=float,
    default=0.1,
    help="Only used by 'adaptive': the mutation probability by the last "
    "generation, reached via linear decay from --initial-rate.",
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

VARIANT: Literal["fixed", "adaptive"] = args.variant
POP_SIZE: int = args.pop
BUDGET: int = args.budget
NUM_MODULES: int = args.max_modules
INITIAL_RATE: float = args.initial_rate
FINAL_RATE: float = args.final_rate
MAX_DEPTH: int = 12

SEED: int = args.seed
RNG = np.random.default_rng(SEED)
random.seed(SEED)

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / f"{SCRIPT_NAME}_{VARIANT}_seed{SEED}"
DATA.mkdir(parents=True, exist_ok=True)

TARGETS = load_targets(TARGET_DIR)


# ============================================================================ #
#  GENOTYPE HELPERS
# ============================================================================ #


def is_connected_tree(genome: TreeGenome) -> bool:
    """Check the genome is a single-rooted, fully connected tree."""
    if len(genome.nodes) == 0:
        return False
    graph = genome.to_networkx()
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if len(roots) != 1:
        return False
    reachable: set[int] = set()
    stack = [roots[0]]
    while stack:
        node = stack.pop()
        reachable.add(node)
        stack.extend(s for s in graph.successors(node) if s not in reachable)
    return len(reachable) == graph.number_of_nodes()


def fitness_of(genome: TreeGenome) -> float:
    """Real assignment fitness: mean + std tree edit distance to targets."""
    if not is_connected_tree(genome):
        return float("inf")
    body = genome.to_networkx()
    return mean_plus_std_tree_edit_distance(body, TARGETS)


# ============================================================================ #
#  THE ONE MUTATION OPERATOR (identical for both variants)
# ============================================================================ #


def mutate_operator(genome: TreeGenome) -> TreeGenome:
    """Single mutation operator used by BOTH variants.

    Only the PROBABILITY of calling this changes between fixed/adaptive -
    the operator itself never does, so it can't be a confound.
    """
    new = copy.deepcopy(genome)
    mutate_replace_node(new)  # confirmed API: mutates one node's type/rotation in place
    _prune_invalid_edges(new)
    return new


def apply_mutation(genome: TreeGenome) -> TreeGenome:
    mutated = mutate_operator(genome)
    if not is_connected_tree(mutated) or not validate_tree_depth(mutated, MAX_DEPTH):
        return genome  # safety net: never inject a broken individual
    return mutated


# ============================================================================ #
#  THE RATE SCHEDULE (the only thing that differs between variants)
# ============================================================================ #


def mutation_rate_for(generation: int) -> float:
    """Return the mutation probability to use at this generation."""
    if VARIANT == "fixed":
        return INITIAL_RATE
    # adaptive: linear decay from INITIAL_RATE (gen 0) to FINAL_RATE (last gen)
    if BUDGET <= 1:
        return INITIAL_RATE
    progress = min(generation / (BUDGET - 1), 1.0)
    return INITIAL_RATE + (FINAL_RATE - INITIAL_RATE) * progress


# ============================================================================ #
#  EA STEPS (identical structure for both variants)
# ============================================================================ #


def create_individual() -> Individual:
    while True:
        genome = random_tree(max_modules=NUM_MODULES)
        if len(genome.nodes) > 0:
            break
    ind = Individual()
    ind.genotype = genome.to_dict()
    ind.tags = {"selected": False}
    return ind


def evaluate(population: Population) -> Population:
    for ind in population.unevaluated:
        genome = TreeGenome.from_dict(ind.genotype)
        ind.fitness = fitness_of(genome)
    return population


def parent_selection(population: Population) -> Population:
    """Binary tournament: shuffle, pair up, mark the better of each pair."""
    shuffled = population.shuffle()
    for idx in range(0, len(shuffled) - 1, 2):
        a, b = shuffled[idx], shuffled[idx + 1]
        if a.fitness_ is None or b.fitness_ is None:
            continue
        better = a if a.fitness_ <= b.fitness_ else b  # lower is better
        worse = b if better is a else a
        better.tags = {"selected": True}
        worse.tags = {"selected": False}
    return shuffled


def crossover(population: Population) -> Population:
    """Subtree crossover between selected parents. Same for both variants."""
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))
    for idx in range(0, len(parents) - 1, 2):
        p_a = TreeGenome.from_dict(parents[idx].genotype)
        p_b = TreeGenome.from_dict(parents[idx + 1].genotype)

        child_a_genome, child_b_genome = crossover_subtree(p_a, p_b)

        for child_genome in (child_a_genome, child_b_genome):
            if not is_connected_tree(child_genome) or not validate_tree_depth(
                child_genome, MAX_DEPTH,
            ):
                continue
            child = Individual()
            child.genotype = child_genome.to_dict()
            child.tags = {"mutate": True}
            population.extend([child])
    return population


def mutate(population: Population) -> Population:
    """Apply the (shared) mutation operator at THIS generation's rate."""
    rate = mutation_rate_for(_generation_counter["n"])
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
        if random.random() < rate:
            genome = TreeGenome.from_dict(ind.genotype)
            mutated = apply_mutation(genome)
            ind.genotype = mutated.to_dict()
            ind.requires_eval = True
    return population


def survivor_selection(population: Population) -> Population:
    """Keep the best `target_population_size` individuals (lower = better)."""
    alive = population.alive
    ranked = sorted(
        alive, key=lambda ind: ind.fitness_ if ind.fitness_ is not None else float("inf"),
    )
    survivors = set(id(ind) for ind in ranked[:POP_SIZE])
    for ind in alive:
        if id(ind) not in survivors:
            ind.alive = False
    return population


# ============================================================================ #
#  LOGGING (fitness AND the mutation rate actually used, for your report)
# ============================================================================ #

LOG_PATH = DATA / "convergence.csv"
_log_rows: list[tuple[int, float, float, float, float]] = []
_generation_counter = {"n": 0}


def log_generation(population: Population) -> Population:
    """Final EA step each generation: record stats, pass population through."""
    fits = [
        ind.fitness_
        for ind in population.alive
        if ind.fitness_ is not None and ind.fitness_ != float("inf")
    ]
    if fits:
        gen = _generation_counter["n"]
        _log_rows.append(
            (
                gen,
                float(np.mean(fits)),
                float(np.std(fits)),
                float(np.min(fits)),
                mutation_rate_for(gen),
            ),
        )
        if gen % 10 == 0:
            print(
                f"gen {gen:4d} | best={min(fits):.4f} "
                f"| mutation_rate={mutation_rate_for(gen):.3f}",
            )
    _generation_counter["n"] += 1
    return population


def save_log() -> None:
    with LOG_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["generation", "mean_fitness", "std_fitness", "best_fitness", "mutation_rate"],
        )
        writer.writerows(_log_rows)
    print(f"saved convergence log to {LOG_PATH}")


# ============================================================================ #
#  MAIN
# ============================================================================ #


def main() -> None:
    print(f"variant       : {VARIANT}")
    print(f"pop / budget  : {POP_SIZE} / {BUDGET}")
    if VARIANT == "fixed":
        print(f"mutation rate : {INITIAL_RATE} (constant)")
    else:
        print(f"mutation rate : {INITIAL_RATE} -> {FINAL_RATE} (linear decay)")
    print(f"seed          : {SEED}")

    population = Population([create_individual() for _ in range(POP_SIZE)])
    population = evaluate(population)
    log_generation(population)  # generation 0, before any evolution

    ops: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
        EAOperation(log_generation),
    ]

    # is_maximisation=False: tree edit distance is LOWER-is-better, matching
    # the direction already used in parent_selection/survivor_selection above.
    ea = EA(population, ops, num_steps=BUDGET, is_maximisation=False)
    ea.run()

    save_log()

    best = ea.get_solution("best", only_alive=False)
    print(f"\nbest individual fitness: {best.fitness_ if best else 'N/A'}")


if __name__ == "__main__":
    main()
