from pathlib import Path

# Experiment size and reproducible random seeds (one new seed per run).
NUM_GENERATIONS = 150
NUM_RUNS = 30
POPULATION_SIZE = 50
BASE_SEED = 67

# Tree generation and mutation settings.
# NUM_MODULES is passed to the tree operators; it is not a global size check.
NUM_MODULES = 20
FIXED_MUTATION_RATE = 0.5
START_MUTATION_RATE = 0.5
END_MUTATION_RATE = 0.01
MUTATION_TYPES = ["point", "subtree", "shrink", "hoist"]
MUTATION_WEIGHTS = [0.4, 0.4, 0.1, 0.1]
ROTATION_MUTATION_RATE = 0.2

# Input bodies and optional robot rendering.
TARGET_DIR = Path(__file__).parent / "target_bodies"
MODE = "frame"
SPAWN_POS = [0.0, 0.0, 0.1]
VIDEO_DURATION = 5.0

# Analysis: threshold is 10% above the best observed fitness.
THRESHOLD_MULTIPLIER = 1.1
ZOOM_START_GENERATION = 40
ZOOM_FITNESS_MARGIN = 0.5
