#!/bin/bash
# Removed — scoring and visualization stages no longer exist in the pipeline.
# Trajectory artifacts (.npz + .json per sample) are the final output of Stage 2.
# Analysis can be done offline with numpy/pandas on the per-sample files.
echo "This template is no longer used. See jobs/templates/forward_pass.sh." >&2
exit 1
