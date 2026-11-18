# Source this in every shell or job that touches the H100:  source setup/cluster_env.sh
# The H100 module tree must be added before CUDA can be loaded from it.
if command -v module >/dev/null 2>&1; then
  module use /opt/apps/nfs/modules/h100/rocky9.4/Core
  module load cuda/12.9.1
else
  echo "cluster_env.sh: 'module' command not found; skipping CUDA module load" >&2
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
