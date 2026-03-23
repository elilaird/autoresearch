#!/usr/bin/env zsh

# Submit a single autoresearch training run to SLURM.
# Called from inside an agent's worktree directory.
# Returns the job ID and log path so the agent can poll for completion.
#
# Usage (from inside a worktree):
#   TAG=mar23 ./make_sbatch.sh
#
#   # Override time/partition
#   TAG=mar23 TIME=0-01:00:00 PARTITION=short ./make_sbatch.sh

DATETIME=$(date +"%Y%m%d_%H%M%S")

TAG=${TAG:?"ERROR: TAG is required. Usage: TAG=mar23 ./make_sbatch.sh"}

TIME=${TIME:-0-01:00:00}
PARTITION=${PARTITION:-batch}
CONDA_ENV=${CONDA_ENV:-world_models}
GPU=${GPU:-1}
CPUS=${CPUS:-16}
MEM=${MEM:-16G}
PY_ARGS="${@}"

ENV_DIR=${ENV_DIR:-"/lustre/smuexa01/client/users/ejlaird/envs"}

# The working directory is the agent's worktree (where this script is called from)
WORKTREE_DIR=$(pwd)

COMMAND="python train.py ${PY_ARGS}"

# Ensure output directory exists
mkdir -p ${WORKTREE_DIR}/output/train

# Output log path (the agent will read this after the job finishes)
LOG_PATH="${WORKTREE_DIR}/output/train/run_${DATETIME}.out"

# Write sbatch script
SBATCH_FILE="${WORKTREE_DIR}/output/train/run_${DATETIME}.sbatch"
cat > ${SBATCH_FILE} << SBATCH_EOF
#!/usr/bin/env zsh
#SBATCH -J ar-${TAG}
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

echo "=== AUTORESEARCH RUN ==="
echo "TAG: ${TAG}"
echo "SLURM_JOB_ID: \${SLURM_JOB_ID}"
echo "Working dir: ${WORKTREE_DIR}"
echo "Branch: \$(git branch --show-current)"
echo "Commit: \$(git rev-parse --short HEAD)"
echo "Command: ${COMMAND}"
echo "========================"

${COMMAND}
SBATCH_EOF

# Submit and capture job ID
SBATCH_OUTPUT=$(sbatch ${SBATCH_FILE} 2>&1)
JOB_ID=$(echo ${SBATCH_OUTPUT} | grep -oP '\d+$')

if [ -z "${JOB_ID}" ]; then
    echo "ERROR: sbatch failed: ${SBATCH_OUTPUT}"
    rm -f ${SBATCH_FILE}
    exit 1
fi

# Print machine-readable output for the agent to parse
echo "SUBMITTED_JOB_ID=${JOB_ID}"
echo "LOG_PATH=${LOG_PATH}"

# Clean up sbatch file
rm -f ${SBATCH_FILE}
