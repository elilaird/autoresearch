#!/usr/bin/env zsh

# Submit an autoresearch job to SLURM.
#
# Two modes:
#   MODE=train (default) — submit a single python train.py run.
#       The Claude agent runs on the login node, submits jobs, polls results.
#       Use this when you're connected and watching.
#
#   MODE=agent — submit a full autonomous Claude Code agent as the SLURM job.
#       Claude runs inside the job with GPU access, no login session needed.
#       Use this to run overnight / when you disconnect.
#
# All agents sharing the same TAG share the same results.tsv and directions.md
# on lustre, regardless of mode. This prevents duplicated work across agents.
#
# Usage (from inside an agent's worktree):
#   # Option A: single training run (agent on login node polls for this)
#   TAG=mar23 ./make_sbatch.sh
#
#   # Option B: autonomous agent in SLURM (survives disconnect)
#   TAG=mar23 MODE=agent AGENT=0 ./make_sbatch.sh
#   TAG=mar23 MODE=agent AGENT=1 TIME=0-12:00:00 ./make_sbatch.sh
#
#   # Mix both: some agents on login node, some in SLURM — all share results
#   # Terminal 1 (login node): cd agents/mar23-0 && claude
#   # Terminal 2: cd agents/mar23-1 && TAG=mar23 MODE=agent AGENT=1 ./make_sbatch.sh

DATETIME=$(date +"%Y%m%d_%H%M%S")

TAG=${TAG:?"ERROR: TAG is required. Usage: TAG=mar23 ./make_sbatch.sh"}
MODE=${MODE:-train}
AGENT=${AGENT:-0}

PARTITION=${PARTITION:-batch}
CONDA_ENV=${CONDA_ENV:-world_models}
GPU=${GPU:-1}
CPUS=${CPUS:-16}
MEM=${MEM:-48G}
PY_ARGS="${@}"

ENV_DIR=${ENV_DIR:-"/lustre/smuexa01/client/users/ejlaird/envs"}

# Shared coordination directory — ALL agents with the same TAG use this
SHARED_DIR=${SHARED_DIR:-"/lustre/smuexa01/client/users/ejlaird/autoresearch/shared/${TAG}"}

# The working directory is the agent's worktree
WORKTREE_DIR=$(pwd)

# Default time: 1h for train mode, 8h for agent mode
if [ "${MODE}" = "agent" ]; then
    TIME=${TIME:-0-08:00:00}
else
    TIME=${TIME:-0-01:00:00}
fi

# Agent mode requires API key
if [ "${MODE}" = "agent" ]; then
    if [ -z "${ANTHROPIC_API_KEY}" ]; then
        echo "ERROR: ANTHROPIC_API_KEY must be set for MODE=agent"
        echo "Usage: ANTHROPIC_API_KEY=sk-... TAG=mar23 MODE=agent ./make_sbatch.sh"
        exit 1
    fi
fi

# Ensure directories exist
mkdir -p ${WORKTREE_DIR}/output/train
mkdir -p ${SHARED_DIR}

# Initialize shared files if they don't exist (idempotent)
if [ ! -f "${SHARED_DIR}/results.tsv" ]; then
    printf "agent\tcommit\tval_dt_score\tmemory_gb\tstatus\tdescription\n" > "${SHARED_DIR}/results.tsv"
fi
if [ ! -f "${SHARED_DIR}/directions.md" ]; then
    cat > "${SHARED_DIR}/directions.md" << 'DIRECTIONS_EOF'
# Shared experiment directions

This file is shared across all agents in this run.
Before starting an experiment, check what others have tried.
After completing an experiment, log your findings here.

## Claimed directions
<!-- Format: - agent <id>: <direction description> -->

## Key findings
<!-- Format: - agent <id>: <finding> (commit <hash>, val_dt_score=<score>) -->
DIRECTIONS_EOF
fi

# --- Build sbatch script ---

LOG_PATH="${WORKTREE_DIR}/output/train/${MODE}_${DATETIME}.out"
SBATCH_FILE="${WORKTREE_DIR}/output/train/${MODE}_${DATETIME}.sbatch"

cat > ${SBATCH_FILE} << SBATCH_HEADER
#!/usr/bin/env zsh
#SBATCH -J ar-${TAG}-${MODE}
#SBATCH -A coreyc_coreyc_mp_jepa_0001
#SBATCH -o ${LOG_PATH}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --nodes=1
#SBATCH --gres=gpu:${GPU}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --tasks-per-node=1

module purge
module load conda
module load gcc/11.2.0
conda activate ${ENV_DIR}/${CONDA_ENV}

cd ${WORKTREE_DIR}

# Shared state — available to both python and claude
export AUTORESEARCH_SHARED_DIR="${SHARED_DIR}"
export AUTORESEARCH_AGENT_ID="${AGENT}"
export AUTORESEARCH_TAG="${TAG}"

echo "=== AUTORESEARCH ==="
echo "MODE: ${MODE}"
echo "TAG: ${TAG}"
echo "AGENT: ${AGENT}"
echo "SHARED_DIR: ${SHARED_DIR}"
echo "SLURM_JOB_ID: \${SLURM_JOB_ID}"
echo "Working dir: ${WORKTREE_DIR}"
echo "Branch: \$(git branch --show-current)"
echo "Commit: \$(git rev-parse --short HEAD)"
echo "===================="

SBATCH_HEADER

if [ "${MODE}" = "train" ]; then
    cat >> ${SBATCH_FILE} << TRAIN_EOF

python train.py ${PY_ARGS}
TRAIN_EOF

elif [ "${MODE}" = "agent" ]; then
    cat >> ${SBATCH_FILE} << AGENT_EOF

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
export PYTHONUNBUFFERED=1

PROMPT="Read program.md and start the experiment loop. Tag=${TAG}, Agent=${AGENT}. Shared dir: ${SHARED_DIR}. Run autonomously — do not ask questions. You have direct GPU access — run python train.py directly, do NOT use make_sbatch.sh."

# --output-format=stream-json prints all agent activity (tool calls, reasoning, results) to stdout
# which SLURM captures in the -o log file
claude --dangerously-skip-permissions --output-format=stream-json -p "\${PROMPT}" 2>&1
AGENT_EOF
fi

# --- Submit ---

SBATCH_OUTPUT=$(sbatch ${SBATCH_FILE} 2>&1)
JOB_ID=$(echo ${SBATCH_OUTPUT} | grep -oP '\d+$')

if [ -z "${JOB_ID}" ]; then
    echo "ERROR: sbatch failed: ${SBATCH_OUTPUT}"
    rm -f ${SBATCH_FILE}
    exit 1
fi

echo "SUBMITTED_JOB_ID=${JOB_ID}"
echo "LOG_PATH=${LOG_PATH}"
echo "MODE=${MODE}"
echo "SHARED_DIR=${SHARED_DIR}"

rm -f ${SBATCH_FILE}
