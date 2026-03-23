#!/usr/bin/env zsh

# Set up and launch an autoresearch agent in one step.
# Creates the git worktree (if needed), initializes shared files, and either
# opens an interactive session or submits a SLURM job.
#
# Usage:
#   # Option A: Interactive on login node (requires active session)
#   TAG=mar23 AGENT=0 ./run_agent.sh
#
#   # Option B: Autonomous SLURM job (survives disconnect)
#   TAG=mar23 AGENT=0 MODE=agent ANTHROPIC_API_KEY=sk-... ./run_agent.sh
#
#   # Launch 3 autonomous agents overnight
#   TAG=mar23 AGENT=0 MODE=agent ANTHROPIC_API_KEY=sk-... TIME=0-12:00:00 ./run_agent.sh
#   TAG=mar23 AGENT=1 MODE=agent ANTHROPIC_API_KEY=sk-... TIME=0-12:00:00 ./run_agent.sh
#   TAG=mar23 AGENT=2 MODE=agent ANTHROPIC_API_KEY=sk-... TIME=0-12:00:00 ./run_agent.sh
#
#   # Submit a single training run from an existing worktree (used by login-node agents)
#   TAG=mar23 MODE=train ./run_agent.sh

set -e

TAG=${TAG:?"ERROR: TAG is required. Usage: TAG=mar23 AGENT=0 ./run_agent.sh"}
AGENT=${AGENT:-0}
MODE=${MODE:-interactive}  # interactive, agent, or train

PARTITION=${PARTITION:-batch}
CONDA_ENV=${CONDA_ENV:-world_models}
GPU=${GPU:-1}
CPUS=${CPUS:-16}
MEM=${MEM:-48G}
PY_ARGS="${@}"

HOME_DIR=${HOME_DIR:-"$(cd "$(dirname "$0")" && pwd)"}
AGENTS_DIR=${AGENTS_DIR:-"/users/ejlaird/Projects/autoresearch-agents"}
ENV_DIR=${ENV_DIR:-"/lustre/smuexa01/client/users/ejlaird/envs"}
SHARED_DIR=${SHARED_DIR:-"/lustre/smuexa01/client/users/ejlaird/autoresearch/shared/${TAG}"}

BRANCH="autoresearch/${TAG}-${AGENT}"
WORKTREE_DIR="${AGENTS_DIR}/${TAG}-${AGENT}"

DATETIME=$(date +"%Y%m%d_%H%M%S")

# Default time depends on mode
if [ "${MODE}" = "agent" ]; then
    TIME=${TIME:-0-08:00:00}
else
    TIME=${TIME:-0-01:00:00}
fi

# ---------------------------------------------------------------------------
# Step 1: Create worktree (idempotent — skips if already exists)
# ---------------------------------------------------------------------------

if [ "${MODE}" = "train" ]; then
    # Train mode runs from an existing worktree (the agent's cwd)
    WORKTREE_DIR=$(pwd)
else
    if [ ! -d "${WORKTREE_DIR}" ]; then
        echo "Creating worktree at ${WORKTREE_DIR}..."
        mkdir -p ${AGENTS_DIR}
        cd ${HOME_DIR}

        if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
            git worktree add "${WORKTREE_DIR}" "${BRANCH}"
        else
            git worktree add -b "${BRANCH}" "${WORKTREE_DIR}"
        fi

        mkdir -p "${WORKTREE_DIR}/output/train"
        echo "Worktree created: ${WORKTREE_DIR} (branch: ${BRANCH})"
    else
        echo "Worktree already exists: ${WORKTREE_DIR}"
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Initialize shared coordination files (idempotent)
# ---------------------------------------------------------------------------

mkdir -p ${SHARED_DIR}

if [ ! -f "${SHARED_DIR}/results.tsv" ]; then
    printf "agent\tcommit\tval_dt_score\tmemory_gb\tstatus\tdescription\n" > "${SHARED_DIR}/results.tsv"
    echo "Created shared results: ${SHARED_DIR}/results.tsv"
fi

# Create per-agent notes directory
NOTES_DIR="${SHARED_DIR}/agents/${AGENT}"
mkdir -p "${NOTES_DIR}"

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
    echo "Created shared directions: ${SHARED_DIR}/directions.md"
fi

# ---------------------------------------------------------------------------
# Step 3: Launch based on mode
# ---------------------------------------------------------------------------

if [ "${MODE}" = "interactive" ]; then
    # Option A: Start Claude Code interactively on login node
    echo ""
    echo "=== AGENT READY ==="
    echo "WORKTREE: ${WORKTREE_DIR}"
    echo "BRANCH:   ${BRANCH}"
    echo "SHARED:   ${SHARED_DIR}"
    echo "==================="
    echo ""
    echo "Starting Claude Code session..."
    echo ""

    cd ${WORKTREE_DIR}
    exec claude --dangerously-skip-permissions -p \
        "Read program.md and start the experiment loop. Tag=${TAG}, Agent=${AGENT}. Shared dir: ${SHARED_DIR}. Notes dir: ${SHARED_DIR}/agents/${AGENT}. Do not ask me any questions. Run autonomously until I stop you."

elif [ "${MODE}" = "agent" ]; then
    # Option B: Submit Claude Code as a SLURM job
    if [ -z "${ANTHROPIC_API_KEY}" ]; then
        echo "ERROR: ANTHROPIC_API_KEY must be set for MODE=agent"
        echo "Usage: ANTHROPIC_API_KEY=sk-... TAG=mar23 AGENT=0 MODE=agent ./run_agent.sh"
        exit 1
    fi

    LOG_PATH="${WORKTREE_DIR}/output/train/agent_${DATETIME}.out"
    SBATCH_FILE="${WORKTREE_DIR}/output/train/agent_${DATETIME}.sbatch"

    cat > ${SBATCH_FILE} << SBATCH_EOF
#!/usr/bin/env zsh
#SBATCH -J ar-${TAG}-a${AGENT}
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

export AUTORESEARCH_SHARED_DIR="${SHARED_DIR}"
export AUTORESEARCH_AGENT_ID="${AGENT}"
export AUTORESEARCH_TAG="${TAG}"
export AUTORESEARCH_NOTES_DIR="${SHARED_DIR}/agents/${AGENT}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
export PYTHONUNBUFFERED=1

echo "=== AUTORESEARCH AGENT ==="
echo "TAG: ${TAG}"
echo "AGENT: ${AGENT}"
echo "SHARED_DIR: ${SHARED_DIR}"
echo "SLURM_JOB_ID: \${SLURM_JOB_ID}"
echo "Working dir: ${WORKTREE_DIR}"
echo "Branch: \$(git branch --show-current)"
echo "Commit: \$(git rev-parse --short HEAD)"
echo "==========================="

PROMPT="Read program.md and start the experiment loop. Tag=${TAG}, Agent=${AGENT}. Shared dir: ${SHARED_DIR}. Notes dir: ${SHARED_DIR}/agents/${AGENT}. Run autonomously — do not ask questions. You have direct GPU access — run python train.py directly, do NOT use run_agent.sh."

claude --dangerously-skip-permissions --output-format=stream-json -p "\${PROMPT}" 2>&1
SBATCH_EOF

    SBATCH_OUTPUT=$(sbatch ${SBATCH_FILE} 2>&1)
    JOB_ID=$(echo ${SBATCH_OUTPUT} | grep -oP '\d+$')

    if [ -z "${JOB_ID}" ]; then
        echo "ERROR: sbatch failed: ${SBATCH_OUTPUT}"
        rm -f ${SBATCH_FILE}
        exit 1
    fi

    echo ""
    echo "=== AGENT SUBMITTED ==="
    echo "JOB_ID:    ${JOB_ID}"
    echo "LOG:       ${LOG_PATH}"
    echo "WORKTREE:  ${WORKTREE_DIR}"
    echo "BRANCH:    ${BRANCH}"
    echo "SHARED:    ${SHARED_DIR}"
    echo "TIME:      ${TIME}"
    echo "========================"
    echo ""
    echo "Monitor:  tail -f ${LOG_PATH}"
    echo "Stop:     scancel ${JOB_ID}"
    echo "Results:  cat ${SHARED_DIR}/results.tsv"

    rm -f ${SBATCH_FILE}

elif [ "${MODE}" = "train" ]; then
    # Submit a single python train.py run to SLURM (used by login-node agents)
    LOG_PATH="${WORKTREE_DIR}/output/train/train_${DATETIME}.out"
    SBATCH_FILE="${WORKTREE_DIR}/output/train/train_${DATETIME}.sbatch"

    cat > ${SBATCH_FILE} << SBATCH_EOF
#!/usr/bin/env zsh
#SBATCH -J ar-${TAG}-train
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

export AUTORESEARCH_SHARED_DIR="${SHARED_DIR}"
export AUTORESEARCH_AGENT_ID="${AGENT}"
export AUTORESEARCH_TAG="${TAG}"
export AUTORESEARCH_NOTES_DIR="${SHARED_DIR}/agents/${AGENT}"

echo "=== AUTORESEARCH TRAIN ==="
echo "TAG: ${TAG}"
echo "SLURM_JOB_ID: \${SLURM_JOB_ID}"
echo "Working dir: ${WORKTREE_DIR}"
echo "Branch: \$(git branch --show-current)"
echo "Commit: \$(git rev-parse --short HEAD)"
echo "==========================="

python train.py ${PY_ARGS}
SBATCH_EOF

    SBATCH_OUTPUT=$(sbatch ${SBATCH_FILE} 2>&1)
    JOB_ID=$(echo ${SBATCH_OUTPUT} | grep -oP '\d+$')

    if [ -z "${JOB_ID}" ]; then
        echo "ERROR: sbatch failed: ${SBATCH_OUTPUT}"
        rm -f ${SBATCH_FILE}
        exit 1
    fi

    echo "SUBMITTED_JOB_ID=${JOB_ID}"
    echo "LOG_PATH=${LOG_PATH}"

    rm -f ${SBATCH_FILE}

else
    echo "ERROR: Unknown MODE=${MODE}. Use: interactive, agent, or train"
    exit 1
fi
