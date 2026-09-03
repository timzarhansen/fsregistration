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
LEVEL_POTENTIAL_ROTATION = 0.0

# 2D peak detection threshold.
POTENTIAL_NECCESSARY_FOR_PEAK = 0.01

# Radial frequency band in FFT grid units (px); 0.0 = auto (N-dependent).
R_MIN = 20.0
R_MAX = 120.0

# Correlation normalization: 0 = 1, 1 = 1/sqrt(norm), 2 = 1/norm.
NORMALIZATION = 0

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

# ----------------------------------------------------------------------------
# Hidden-component rotation scan (kernel-fit, port of rotation_curve_analysis.ipynb)
# ----------------------------------------------------------------------------
# Requires USE_DIRECT=True (skipped with a warning otherwise). Adds plateau /
# shoulder rotation candidates that persistence peak detection cannot see;
# each candidate is emitted twice (mu and mu + pi), the translation stage
# disambiguates the copy. The scan runs a second feedback pass over the newly
# found candidates. False = classic persistence-only rotation candidates.
USE_HIDDEN_COMPONENT_SCAN = True

# Hidden-scan parameters (defaults mirror the notebook analysis)
HIDDEN_SCAN_WIN_HALF_RAD = 0.35      # scan window half width around each anchor peak (~20 deg)
HIDDEN_SCAN_COARSE_RAD = 0.01        # hidden-component scan grid step (~0.57 deg)
HIDDEN_SCAN_FINE_RAD = 0.0004        # local refinement step (~0.02 deg)
HIDDEN_SCAN_MIN_SEP_RAD = 0.05       # min separation between components (~2.9 deg)
HIDDEN_SCAN_MIN_IMPROV_RATIO = 2.0   # marginal residual improvement to accept a hidden component
HIDDEN_SCAN_WEAK_FLOOR_RATIO = 1.2   # candidates below min_improv_ratio but >= this are weak
HIDDEN_SCAN_KNOWN_MARGIN_RAD = 0.26  # persistence peaks within +/-this of a window count as known components
HIDDEN_SCAN_MAX_HIDDEN = 2           # max additional components tested per window
HIDDEN_SCAN_INCLUDE_WEAK = True      # include weak candidates in the output peak list

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