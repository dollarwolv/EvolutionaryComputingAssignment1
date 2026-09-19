from typing import cast
import copy
import contextlib
import numpy as np
import pandas as pd

from rich.console import Console
from rich.traceback import install

from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    Individual,
    IntegerMutator,
    IntegersGenerator,
    Population,
    config,
)

from data_analysis import create_plot
from experiment_settings import (
    NUM_GENERATIONS,
    NUM_RUNS,
    POPULATION_SIZE,
    BASE_SEED,
    NUM_MODULES,
    FIXED_MUTATION_RATE,
    START_MUTATION_RATE,
    END_MUTATION_RATE,
    MUTATION_TYPES,
    MUTATION_WEIGHTS,
    ROTATION_MUTATION_RATE,
    TARGET_DIR,
    MODE,
    SPAWN_POS,
    VIDEO_DURATION,
)

import math

import argparse
import subprocess
import sys

from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    validate_tree_depth,
)

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.simulation.environments._simple_flat_with_target import (
    SimpleFlatWorldWithTarget,
)

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)

# Standard library
import random
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import networkx as nx
import numpy as np
import torch
from mujoco import viewer

import pprint

# Local scripts
from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type GenotypeTypes = Literal["nde", "tree"]
type ViewerTypes = Literal["launcher", "video", "frame", "none"]


parser = argparse.ArgumentParser(
    description="Mutation mode",
)

parser.add_argument(
    "--run-all",
    action="store_true",
    help="Run fixed, linear, logarithmic, and exponential mutation experiments",
)

parser.add_argument(
    "--adaptive",
    action="store_true",
    help="Decreasing mutation rate. Default: Fixed mutation rate",
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


args = parser.parse_args()

if (
    not args.run_all
    and args.adaptive
    and not (args.linear or args.logarithmic or args.exponential)
):
    raise ValueError(
        "adaptive needs to be provided with another argument: --linear, --logarithmic or --exponential"
    )

run_type = None

if not args.adaptive:
    run_type = "fixed"
elif args.linear:
    run_type = "linear"
elif args.logarithmic:
    run_type = "logarithmic"
elif args.exponential:
    run_type = "exponential"

print("Adaptive:", args.adaptive)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# ============================================================================ #
#  1. THE TARGET BODIES
# ============================================================================ #
#
# The targets are plain nx.DiGraph JSON files.
# They vary in size on purpose. A body that just matches the average module
# count will not score well against all of them.
#
# ============================================================================ #


def show_body(
    body: nx.DiGraph,
    mode: ViewerTypes = MODE,
    file_name: str = "body",
) -> None:
    """Build a body graph in MuJoCo and look at it.

    There is no controller and no physics worth speaking of - this exists so
    you can SEE what your fitness function is actually rewarding. Do this
    early and often. A number going down is not evidence that the bodies look
    anything like the targets.
    """
    if mode == "none":
        return

    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(body)
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    match mode:
        case "launcher":
            # Interactive window. Drag the modules around; nothing drives them.
            viewer.launch(model=model, data=data)
        case "frame":
            # A still image - the cheapest way to eyeball a body.
            save_path = str(DATA / f"{file_name}.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
            console.log(f"saved {save_path}")
        case "video":
            # Mostly useful for showing a body slumping under gravity.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model, data, duration=VIDEO_DURATION, video_recorder=recorder
            )


install()
console = Console()


def get_module_count(genome: TreeGenome) -> int:
    """Count the number of modules in a morphology."""
    return len(genome.nodes)


def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory.

    Returns
    -------
    list of nx.DiGraph
        One graph per JSON file, sorted by filename.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


def make_individual() -> Individual:
    while True:
        # create random tree until it has modules
        genome = random_tree(NUM_MODULES)
        if get_module_count(genome) > 0:
            break

    ind = Individual()
    ind.genotype = genome.to_dict()

    return ind


def fitness_function(
    body: nx.DiGraph,
    targets: list[nx.DiGraph],
) -> float:
    """Score one body against the whole target set. LOWER IS BETTER.

    Some things worth thinking about:
      * The std term charges for unevenness - body that is mediocre against every target
        and one that is excellent on most but bad on one can still land close
        in fitness, but the latter is penalized a bit more.
      * Nothing here rewards small bodies. Does your EA bloat? Should a size
        penalty be part of fitness, or is that the encoding's job?
    """
    return mean_plus_std_tree_edit_distance(body, targets)


def evaluate(population: Population, targets: list) -> Population:
    for ind in population.unevaluated:
        genotype = TreeGenome.from_dict(ind.genotype)
        genome = genotype.to_networkx()

        ind.fitness = fitness_function(genome, targets)

    return population


def parent_selection(population: Population) -> Population:
    shuffled = population.shuffle()
    for idx in range(0, len(shuffled) - 1, 2):
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]

        if ind_a.fitness_ is not None and ind_b.fitness_ is not None:
            if ind_a.fitness_ <= ind_b.fitness_:
                ind_a.tags = {"selected": True}
                ind_b.tags = {"selected": False}
            else:
                ind_a.tags = {"selected": False}
                ind_b.tags = {"selected": True}

    return shuffled


def is_connected_tree(genome: TreeGenome) -> bool:
    """Check if genome is a valid connected tree with single root."""
    if len(genome.nodes) == 0:
        return False
    robot_graph = genome.to_networkx()
    # Check for single root (one node with no predecessors)
    roots = [n for n in robot_graph.nodes() if robot_graph.in_degree(n) == 0]
    if len(roots) != 1:
        return False
    # Check connectivity: all nodes reachable from root
    root = roots[0]
    reachable = set()
    stack = [root]
    while stack:
        node = stack.pop()
        reachable.add(node)
        stack.extend(
            succ for succ in robot_graph.successors(node) if succ not in reachable
        )
    return len(reachable) == robot_graph.number_of_nodes()


def crossover_morphologies(
    parent1: Individual,
    parent2: Individual,
) -> tuple[TreeGenome, TreeGenome]:
    """One-point crossover for morphologies. Recovers from invalid results."""
    t1 = parent1.genotype
    t2 = parent2.genotype
    if isinstance(t1, dict):
        t1 = TreeGenome.from_dict(t1)
    if isinstance(t2, dict):
        t2 = TreeGenome.from_dict(t2)

    child1, child2 = crossover_subtree(t1, t2)

    output_1, output_2 = child1, child2

    # If disconnected, return copy of a parent instead
    if not is_connected_tree(child1):
        output_1 = TreeGenome.from_dict(t1.to_dict())

    if not is_connected_tree(child2):
        output_2 = TreeGenome.from_dict(t2.to_dict())

    return output_1, output_2


def crossover(population: Population) -> Population:
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))
    for idx in range(0, len(parents) - 1, 2):
        p_a = parents[idx]
        p_b = parents[idx + 1]
        g_a, g_b = crossover_morphologies(p_a, p_b)

        child_a = Individual()
        child_a.genotype = g_a.to_dict()
        child_a.tags = {"mutate": True}

        child_b = Individual()
        child_b.genotype = g_b.to_dict()
        child_b.tags = {"mutate": True}

        population.extend([child_a, child_b])
    return population


def mutate_morphology(genome: TreeGenome) -> TreeGenome:
    """Apply mutations to morphology with multiple GP operators."""
    new = copy.deepcopy(genome)

    # Choose mutation type (standard GP mutation operators)
    mutation_type = RNG.choice(
        MUTATION_TYPES,
        p=MUTATION_WEIGHTS,
    )

    if mutation_type == "point":
        # Point mutation: change node type/rotation
        mutate_replace_node(new)
    elif mutation_type == "subtree":
        # Subtree mutation: replace subtree with new random tree
        mutate_subtree_replacement(new, max_modules=NUM_MODULES)
    elif mutation_type == "shrink":
        # Shrink mutation: replace node+subtree with single leaf
        mutate_shrink(new)
    elif mutation_type == "hoist":
        # Hoist mutation: promote child to replace parent
        mutate_hoist(new)

    # Additional rotation mutation has its own probability.
    if RNG.random() < ROTATION_MUTATION_RATE:
        noncore = [nid for nid in new.nodes if nid != IDX_OF_CORE]
        if noncore:
            nid = random.choice(noncore)
            # choose a new rotation allowed for its type
            mtype = ModuleType[new.nodes[nid]["type"]]
            rots = [r.name for r in ALLOWED_ROTATIONS[mtype]]
            if rots:
                new.nodes[nid]["rotation"] = random.choice(rots)

    # prune any invalid edges before returning
    _prune_invalid_edges(new)
    with contextlib.suppress(ValueError):
        validate_genome_dict(new.to_dict())
    return new


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
        return start * (end / start) ** (x / n_generations)

    return start + (end - start) * progress


def mutate(population: Population) -> Population:
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):

        mutate = False

        # find the current generation by looping through the individuals
        # and taking max time of death + 1
        n_generation = max(ind.time_of_death for ind in population) + 1

        if args.adaptive:
            ind.tags["adaptive_mutation"] = True

            if args.linear:
                mutation_rate = pick_mutation_rate(
                    n_generation, NUM_GENERATIONS, True, "linear"
                )

            elif args.logarithmic:
                mutation_rate = pick_mutation_rate(
                    n_generation, NUM_GENERATIONS, True, "logarithmic"
                )

            elif args.exponential:
                mutation_rate = pick_mutation_rate(
                    n_generation, NUM_GENERATIONS, True, "exponential"
                )

            if RNG.random() < mutation_rate:
                mutate = True

        else:
            if RNG.random() < FIXED_MUTATION_RATE:
                mutate = True

        if mutate:
            ind.genotype = mutate_morphology(
                TreeGenome.from_dict(ind.genotype)
            ).to_dict()
            ind.requires_eval = True

    return population


def survivor_selection(population: Population) -> Population:
    shuffled = population.alive.shuffle()
    alive_count = len(shuffled)
    for idx in range(0, len(shuffled) - 1, 2):
        if alive_count <= config.target_population_size:
            break
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if (ind_a.fitness_ or 0.0) <= (ind_b.fitness_ or 0.0):
            ind_b.alive = False
        else:
            ind_a.alive = False
        alive_count -= 1
    return population


# function to get the stats cuz fuck sgl thign
def get_stats(population: Population) -> dict:
    fitnesses = []

    for ind in population.alive:
        if ind.fitness_ is not None:
            fitnesses.append(ind.fitness_)

    return {
        "best_fitness": min(fitnesses),
        "mean_fitness": np.mean(fitnesses),
        "std_fitness": np.std(fitnesses),
    }


# function to set seed for reach run
def set_seed(seed):
    global RNG
    RNG = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def log_stats(
    population: Population,
    this_run: list,
    run: int,
) -> Population:
    previous = this_run[-1]
    generation = previous["generation"] + 1

    stats = get_stats(population)
    best_so_far = min(previous["best_so_far"], stats["best_fitness"])
    mutation_rate = pick_mutation_rate(
        generation,
        NUM_GENERATIONS,
        args.adaptive,
        run_type,
        start=START_MUTATION_RATE if args.adaptive else FIXED_MUTATION_RATE,
    )

    this_run.append(
        {
            "generation": generation,
            "best_fitness": stats["best_fitness"],
            "mean_fitness": stats["mean_fitness"],
            "std_fitness": stats["std_fitness"],
            "best_so_far": best_so_far,
            "mutation_rate": mutation_rate,
            "run": run + 1,
        }
    )

    return population


def main():
    config.target_population_size = POPULATION_SIZE
    config.is_maximisation = False

    initial = Population(
        [make_individual() for _ in range(config.target_population_size)]
    )

    targets = load_targets()
    initial = evaluate(initial, targets)

    all_results = []

    # initialize first gen and best_so_far variables
    generation = 0
    best_so_far = None

    for run in range(NUM_RUNS):
        seed = BASE_SEED + run
        set_seed(seed)

        initial = Population([make_individual() for _ in range(POPULATION_SIZE)])

        initial = evaluate(initial, targets)

        this_run = []
        best_so_far = initial.best(sort="min", attribute="fitness_", n=1)[0].fitness_

        initial_stats = get_stats(initial)

        this_run.append(
            {
                "generation": 0,
                "best_fitness": initial_stats["best_fitness"],
                "mean_fitness": initial_stats["mean_fitness"],
                "std_fitness": initial_stats["std_fitness"],
                "best_so_far": best_so_far,
                "mutation_rate": (
                    START_MUTATION_RATE if args.adaptive else FIXED_MUTATION_RATE
                ),
                "run": run + 1,
            }
        )

        ops: list[EAOperation] = [
            EAOperation(parent_selection),
            EAOperation(crossover),
            EAOperation(mutate),
            EAOperation(evaluate, targets=targets),
            EAOperation(survivor_selection),
            EAOperation(log_stats, this_run=this_run, run=run),
        ]

        ea = EA(
            initial,
            ops,
            num_steps=NUM_GENERATIONS,
        )

        ea.run()

        all_results.extend(this_run)

    df = pd.DataFrame(all_results)
    df.to_csv(HERE / "outputs" / f"dataset_{run_type}.csv", index=False)
    create_plot(run_type)

    return


if __name__ == "__main__":
    if args.run_all:
        # Separate processes keep each experiment's settings and plots independent.
        for schedule in ["fixed", "linear", "logarithmic", "exponential"]:
            print(f"Running {schedule} mutation experiments...", flush=True)
            command = [sys.executable, str(Path(__file__).resolve())]
            if schedule != "fixed":
                command.extend(["--adaptive", f"--{schedule}"])
            subprocess.run(command, check=True)
    else:
        main()
