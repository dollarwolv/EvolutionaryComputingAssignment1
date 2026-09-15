from typing import cast
import copy
import contextlib
import numpy as np

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

import argparse

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
    "--adaptive",
    type=bool,
    default=False,
    help="Decreasing mutation rate. Default: Fixed mutation rate (False)",
)

args = parser.parse_args()

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
# NOTE: the tree operators use the `random` module, the NDE uses numpy for its
# own genotype vectors AND is a torch.nn.Module for its internal network - that
# network's weight initialisation uses torch's own RNG, entirely separate from
# numpy/random. If you're using "nde", seed all THREE or your runs will not be
# reproducible across separate script runs, even with the same seed value.
SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
TARGET_DIR: Path = HERE / "target_bodies"  # the bodies you must approach
NUM_OF_MODULES: int = 20  # module budget per evolved body
GENOTYPE: GenotypeTypes = "tree"  # "nde" | "tree"
MODE: ViewerTypes = "frame"  # see show_body() for the options
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]

NUM_GENERATIONS = 50
MUTATION_RATES = [i / 100 for i in range(NUM_GENERATIONS, 0, -1)]


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
            video_renderer(model, data, duration=5.0, video_recorder=recorder)


install()
console = Console()

NUM_MODULES = 20

HERE = Path(__file__).parent
TARGET_DIR: Path = HERE / "target_bodies"  # the bodies you must approach


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

    # pprint.pprint(ind.genotype)

    # ind.tags["ps"] = False
    # ind.tags["valid"] = True
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
        ["point", "subtree", "shrink", "hoist"],
        p=[0.4, 0.4, 0.1, 0.1],
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

    # Additional rotation mutation (20% chance)
    if RNG.random() < 0.2:
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


def mutate(population: Population) -> Population:
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):

        mutate = False

        if args.adaptive:
            ind.tags["adaptive_mutation"] = True

            # find the current generation by looping through the individuals
            # and taking max time of death + 1
            n_generation = max(ind.time_of_death for ind in population) + 1

            mutation_rate = MUTATION_RATES[n_generation - 1]

            if RNG.random() < mutation_rate:
                mutate = True
                print(f"mutated: {mutate} at rate {mutation_rate}")

        else:
            if RNG.random() < 0.2:
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


def main():
    config.target_population_size = 20
    config.is_maximisation = False

    initial = Population([make_individual() for _ in range(20)])
    targets = load_targets()

    initial = evaluate(initial, targets)

    print(MUTATION_RATES)

    ops: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate),
        EAOperation(evaluate, targets=targets),
        EAOperation(survivor_selection),
    ]

    ea = EA(initial, ops, num_steps=NUM_GENERATIONS)
    ea.run()

    console.log("--- Results ---")
    console.log(f"best = {ea.get_solution('best', only_alive=False)}")
    console.log(f"median = {ea.get_solution('median', only_alive=False)}")
    console.log(f"worst = {ea.get_solution('worst', only_alive=False)}")

    # Population API examples
    db_pop = ea._fetch(only_alive=False)
    top_10 = db_pop.best(sort="max", attribute="fitness_", n=10)
    console.log(f"top-10 from DB = {top_10}")

    sampled_best = db_pop.sample(30).best(sort="max", attribute="fitness_", n=5)
    console.log(f"sample(30).best(n=5) = {sampled_best}")

    return


if __name__ == "__main__":
    main()
