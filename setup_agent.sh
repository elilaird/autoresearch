#!/usr/bin/env zsh

# Create an isolated git worktree for an autoresearch agent.
# Run this once per agent before starting the experiment loop.
#
# Usage:
#   # Create agent 0's worktree
#   TAG=mar23 AGENT=0 ./setup_agent.sh
#
#   # Create agent 1's worktree
#   TAG=mar23 AGENT=1 ./setup_agent.sh
#
# Output:
#   Creates worktree at ~/Projects/autoresearch-agents/<tag>-<agent>/
#   Creates shared dir at /lustre/.../autoresearch/shared/<tag>/
#   Prints WORKTREE_DIR and SHARED_DIR for the agent to use.

set -e

TAG=${TAG:?"ERROR: TAG is required. Usage: TAG=mar23 AGENT=0 ./setup_agent.sh"}
AGENT=${AGENT:-0}

HOME_DIR=${HOME_DIR:-"/users/ejlaird/Projects/autoresearch"}
AGENTS_DIR=${AGENTS_DIR:-"/users/ejlaird/Projects/autoresearch-agents"}
SHARED_DIR=${SHARED_DIR:-"/lustre/smuexa01/client/users/ejlaird/autoresearch/shared/${TAG}"}

BRANCH="autoresearch/${TAG}-${AGENT}"
WORKTREE_DIR="${AGENTS_DIR}/${TAG}-${AGENT}"

# --- Create worktree ---

if [ -d "${WORKTREE_DIR}" ]; then
    echo "Worktree already exists at ${WORKTREE_DIR}"
    echo "To remove it: cd ${HOME_DIR} && git worktree remove ${WORKTREE_DIR}"
    exit 1
fi

mkdir -p ${AGENTS_DIR}
cd ${HOME_DIR}

# Create the branch and worktree
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    echo "Branch ${BRANCH} already exists, creating worktree from it"
    git worktree add "${WORKTREE_DIR}" "${BRANCH}"
else
    echo "Creating new branch ${BRANCH} and worktree"
    git worktree add -b "${BRANCH}" "${WORKTREE_DIR}"
fi

# --- Create shared coordination directory ---

mkdir -p ${SHARED_DIR}

# Initialize shared results file if it doesn't exist
if [ ! -f "${SHARED_DIR}/results.tsv" ]; then
    printf "agent\tcommit\tval_dt_score\tmemory_gb\tstatus\tdescription\n" > "${SHARED_DIR}/results.tsv"
    echo "Created shared results: ${SHARED_DIR}/results.tsv"
fi

# Initialize shared directions file if it doesn't exist
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

# --- Create output directory inside worktree ---

mkdir -p "${WORKTREE_DIR}/output/train"

# --- Print summary for the user ---

echo ""
echo "=== AGENT READY ==="
echo "WORKTREE_DIR=${WORKTREE_DIR}"
echo "SHARED_DIR=${SHARED_DIR}"
echo "BRANCH=${BRANCH}"
echo "AGENT=${AGENT}"
echo "==================="
echo ""
echo "To start the agent, open a Claude Code session in the worktree:"
echo "  cd ${WORKTREE_DIR} && claude"
echo ""
echo "Then tell Claude:"
echo "  Read program.md and start the experiment loop. Tag=${TAG}, Agent=${AGENT}."
echo "  Shared dir: ${SHARED_DIR}"
