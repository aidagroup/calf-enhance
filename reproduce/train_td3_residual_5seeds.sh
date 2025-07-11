#!/bin/bash

# Run training for seeds 0-4 in parallel
for seed in {0..4}; do
    uv run run/train_td3_residual.py --total-timesteps=3000000 --env-id=UnderwaterDrone-v0 --seed=$seed &
done

# Wait for all background processes to complete
wait