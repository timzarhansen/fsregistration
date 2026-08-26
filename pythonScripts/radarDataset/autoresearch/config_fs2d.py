# ============================================================================
#  FS2D full-sequence Boreas registration - configuration
#
#  Edit any value below and re-run:
#
#      python fullSequencefs2dRun.py
#
#  The output CSV in results/ is named from the settings below, so changing
#  a value produces a new result file (previous runs are kept).
# ============================================================================

# ----------------------------------------------------------------------------
# Boreas data
# ----------------------------------------------------------------------------
# Path to the Boreas radar dataset folder (contains the sequence folders).
DATA_DIR = "/home/tim-external/dataFolder/radar_boreas"

# Sequence to register (folder name inside DATA_DIR).
SEQUENCE_NAME = "boreas-2020-11-26-13-58"

# ----------------------------------------------------------------------------
# Pair sampling
# ----------------------------------------------------------------------------
# Register every Nth frame: 1 = adjacent pairs, 4/5/7 = every 4th/5th/7th, ...
# Pairs are always formed from frame 0, i.e. (0,step), (step,2*step), ...
MATCHING_STEP = 5

# None = run the whole sequence. Set e.g. 500 for a quick subset
# (only pairs with curr_frame < MAX_FRAMES are processed).
MAX_FRAMES = None

# ----------------------------------------------------------------------------
# Image geometry
# ----------------------------------------------------------------------------
# Image grid size (N x N).
N = 256

# Scene radius in meters. pixel_size = 2*RADIUS/N is derived automatically.
RADIUS = 140.0

# ----------------------------------------------------------------------------
# FS2D registration parameters
# ----------------------------------------------------------------------------
# Direct 1-angle registration (True) vs SO(3) multi-angle correlation (False).
USE_DIRECT = True

# Number of angles sampled for the direct 1D correlation curve (-1 = auto).
NUM_ANGLES = 4096

# Persistence threshold for rotation peak filtering (only for USE_DIRECT=False).
LEVEL_POTENTIAL_ROTATION = 0.001

# 2D peak detection threshold.
POTENTIAL_NECCESSARY_FOR_PEAK = 0.01

# Radial frequency band in FFT grid units (px); 0.0 = auto (N-dependent).
R_MIN = 25.0
R_MAX = 120.0

# Correlation normalization: 0 = 1, 1 = 1/sqrt(norm), 2 = 1/norm.
NORMALIZATION = 1

# Use phase correlation instead of standard cross-correlation.
USE_PHASE_CORRELATION = False

# Apply circular mask (zero out image corners).
ROUND = False

# CLAHE contrast enhancement.
USE_CLAHE = False

# Hamming window before FFT.
USE_HAMMING = True

# Average over multiple radial bands.
MULTIPLE_RADII = True

# Gaussian weighting of the image.
USE_GAUSS = False

# Weight translation peaks by the rotation correlation score.
USE_WEIGHTED_PEAK_SCORE = True

# Wrapper-level debug output (verbose, only for troubleshooting).
DEBUG_MODE = False

# ----------------------------------------------------------------------------
# Outlier definitions (used for the summary table in the results CSV)
# ----------------------------------------------------------------------------
# Rotation outlier: |rot_error_deg| > this threshold.
OUTLIER_ROT_THRESH_DEG = 5.0

# Translation outlier: trans_error_m > this threshold.
# (Pairs can be counted in both; stats in the summary are computed over
#  inliers = pairs failing NEITHER criterion.)
OUTLIER_TRANS_THRESH_M = 2.0

# ----------------------------------------------------------------------------
# Parallelism
# ----------------------------------------------------------------------------
# Number of parallel worker processes (one core each). 12 = full machine here.
NUM_WORKERS = 12