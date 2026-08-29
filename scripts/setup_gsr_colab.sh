#!/usr/bin/env bash
# =============================================================================
# scripts/setup_gsr_colab.sh
# Reproducible environment setup for the SoccerNet Game State Reconstruction
# (sn-gamestate) project in a fresh Google Colab Linux runtime.
#
# Owner  : ball-free-game-state-reconstruction (main project repository)
# Target : Google Colab — Ubuntu, CUDA GPU recommended
#
# Design decisions
# ----------------
# * uv creates /content/soccernet-gamestate/.venv with Python 3.9, downloading
#   the interpreter automatically if it is not already present on the host.
#   The Colab system Python version is irrelevant.
# * ALL install commands explicitly target the venv interpreter:
#     /content/soccernet-gamestate/.venv/bin/python
#   --system is never used.
# * Dependencies are resolved from the repository's own pyproject.toml so the
#   package list is never duplicated here.
# * Installation order:
#     1. Create venv (uv, Python 3.9)
#     2. uv pip install -e .  (resolves pyproject.toml; may touch setuptools)
#     3. Pin setuptools==80.10.2  (done AFTER project install so the project
#        cannot upgrade it again)
#     4. Verify pkg_resources is importable
#     5. mim install mmcv==2.0.1  (via the venv's own mim binary, after torch)
# * MPLBACKEND=Agg is exported for the verification block and documented for
#   later TrackLab commands.  The activate script is NOT modified.
#
# Usage (run once at the top of your Colab notebook)
# ---------------------------------------------------
#   !git clone https://github.com/<you>/ball-free-game-state-reconstruction /content/capstone
#   !git clone https://github.com/SoccerNet/sn-gamestate /content/soccernet-gamestate
#   !bash /content/capstone/scripts/setup_gsr_colab.sh
#
# For TrackLab / GSR commands run AFTER this script, prefix with:
#   MPLBACKEND=Agg /content/soccernet-gamestate/.venv/bin/tracklab ...
#
# The script does NOT:
#   - download the SoccerNet dataset
#   - run TrackLab or the GSR baseline
#   - modify pyproject.toml, README.md, or .gitignore
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Colour / logging helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Paths (defined early so all steps share the same variables)
# ---------------------------------------------------------------------------
REPO_DIR="/content/soccernet-gamestate"
VENV_DIR="$REPO_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_MIM="$VENV_DIR/bin/mim"

# ---------------------------------------------------------------------------
# 1. Install uv (if not already present)
#    uv manages Python versions internally — no system python3.9 required.
# ---------------------------------------------------------------------------
step "1/6  Installing uv..."

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    info "uv already installed: $(uv --version)"
    export PATH="$HOME/.cargo/bin:$PATH"
fi
uv --version

# ---------------------------------------------------------------------------
# 2. Validate the sn-gamestate repository clone
# ---------------------------------------------------------------------------
step "2/6  Validating repository..."

if [[ ! -d "$REPO_DIR" ]]; then
    error "Repository not found at $REPO_DIR." \
          "Clone it first:" \
          "  git clone https://github.com/SoccerNet/sn-gamestate $REPO_DIR"
fi

if [[ ! -f "$REPO_DIR/pyproject.toml" ]]; then
    error "pyproject.toml not found inside $REPO_DIR." \
          "Make sure the clone is complete."
fi
info "Repository: $REPO_DIR  (pyproject.toml present)"

# ---------------------------------------------------------------------------
# 3. Create the dedicated virtual environment with Python 3.9
#    uv downloads Python 3.9 automatically if it is not on the host.
#    The Colab default Python version does not matter.
# ---------------------------------------------------------------------------
step "3/6  Creating virtual environment at $VENV_DIR (Python 3.9)..."

if [[ ! -d "$VENV_DIR" ]]; then
    uv venv --python 3.9 "$VENV_DIR"
    info "Virtual environment created."
else
    warn "Virtual environment already exists — reusing it."
fi

# Verify the venv interpreter reports Python 3.9.x
VENV_PY_VER=$("$VENV_PYTHON" --version 2>&1 | awk '{print $2}')
VENV_PY_MINOR=$(echo "$VENV_PY_VER" | cut -d. -f2)

if [[ "$VENV_PY_MINOR" -ne 9 ]]; then
    error "Expected Python 3.9.x inside the venv but found $VENV_PY_VER." \
          "Delete $VENV_DIR and re-run this script."
fi
info "Venv interpreter: $VENV_PYTHON ($VENV_PY_VER) — OK"

# ---------------------------------------------------------------------------
# 4. Install sn-gamestate and all its dependencies from pyproject.toml
#    uv reads [project.dependencies] and [tool.uv.sources]:
#      - VCS packages  : prtreid, torchreid (GitHub)
#      - Local editable: tracklab_calibration -> plugins/calibration
#    Nothing is hardcoded here — the package list lives in pyproject.toml.
# ---------------------------------------------------------------------------
step "4/6  Installing sn-gamestate from pyproject.toml..."
cd "$REPO_DIR"

uv pip install \
    --python "$VENV_PYTHON" \
    -e "."

# ---------------------------------------------------------------------------
# 5. Pin setuptools==80.10.2 AFTER the project install
#    Done last so the project resolver cannot upgrade it again.
#    OpenMIM and MMCV rely on pkg_resources at both install-time and runtime;
#    setuptools==80.10.2 ships it unconditionally.
# ---------------------------------------------------------------------------
step "5/6  Pinning setuptools==80.10.2 and verifying pkg_resources..."

uv pip install \
    --python "$VENV_PYTHON" \
    "setuptools==80.10.2"

"$VENV_PYTHON" -c "import pkg_resources" \
    || error "pkg_resources is missing inside the venv after setuptools pin. Aborting."
info "pkg_resources — OK"

# ---------------------------------------------------------------------------
# 6. MMCV 2.0.1 via OpenMIM
#    Runs AFTER torch is installed (step 4) so MIM selects the correct wheel.
#    The mim binary is sourced from the same venv.
# ---------------------------------------------------------------------------
step "6/6  Installing MMCV 2.0.1 via OpenMIM..."

TORCH_VER=$("$VENV_PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null \
    || echo "unknown")
info "Detected torch version inside venv: $TORCH_VER"

"$VENV_MIM" install "mmcv==2.0.1"

# ---------------------------------------------------------------------------
# MPLBACKEND=Agg
# Export for the verification subprocess below.
# NOTE: The activate script is NOT modified.
# For all subsequent TrackLab / GSR commands in Colab cells, prepend:
#   MPLBACKEND=Agg /content/soccernet-gamestate/.venv/bin/tracklab ...
# ---------------------------------------------------------------------------
export MPLBACKEND=Agg
info "MPLBACKEND=Agg exported for this session."

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
info "Running verification checks..."
echo "============================================================"

"$VENV_PYTHON" - <<'PYEOF'
import sys, importlib

CHECKS_PASSED = 0
CHECKS_FAILED = 0

def check(label, fn):
    global CHECKS_PASSED, CHECKS_FAILED
    try:
        result = fn()
        print(f"  [PASS]  {label}: {result}")
        CHECKS_PASSED += 1
    except Exception as exc:
        print(f"  [FAIL]  {label}: {exc}", file=sys.stderr)
        CHECKS_FAILED += 1

# Python version — must be 3.9.x
check(
    "Python version",
    lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
)

# torch
check(
    "torch version",
    lambda: __import__("torch").__version__
)

# CUDA
check(
    "CUDA available",
    lambda: str(__import__("torch").cuda.is_available())
)

# tracklab
check(
    "tracklab import",
    lambda: importlib.import_module("tracklab") and "OK"
)

# mmcv
check(
    "mmcv import + version",
    lambda: importlib.import_module("mmcv").__version__
)

# mmdet
check(
    "mmdet import + version",
    lambda: importlib.import_module("mmdet").__version__
)

# mmocr
check(
    "mmocr import + version",
    lambda: importlib.import_module("mmocr").__version__
)

# sn_gamestate
check(
    "sn_gamestate import",
    lambda: importlib.import_module("sn_gamestate") and "OK"
)

print("============================================================")
print(f"  Results: {CHECKS_PASSED} passed, {CHECKS_FAILED} failed")
if CHECKS_FAILED > 0:
    sys.exit(1)
PYEOF

echo "============================================================"
info "setup_gsr_colab.sh completed successfully."
info ""
info "Venv location : $VENV_DIR"
info "Interpreter   : $VENV_PYTHON"
info ""
info "To activate in a Colab cell:"
info "  import subprocess, os"
info "  os.environ['PATH'] = '$VENV_DIR/bin:' + os.environ['PATH']"
info ""
info "For TrackLab commands remember to set MPLBACKEND=Agg, e.g.:"
info "  MPLBACKEND=Agg $VENV_DIR/bin/tracklab ..."