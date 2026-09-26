# All-node ABD diffusion policy

## Base assumptions

- `train.py` remains the normal diffusion baseline.
- `train_abd.py` remains the root-conditioned ABD variant.
- `train_abd_all.py` uses all eight post-message-passing Panda-arm features.
- Each node has 256 features. Their concatenated 2,048 features are supplied
  directly to the diffusion policy without a projection layer.
- The default observation, action prediction, and action execution horizons are
  2, 16, and 12 respectively.
- The full eight-dimensional environment action is retained; the ABD decoder is
  not used.

Run from this directory:

```bash
cd /home/fawad/Graph/Graph-Link/examples/baselines/diffusion_policy
```

```bash
uv run python train_abd_all.py \
  --env-id RollBall-v1 \
  --demo-path /home/fawad/Graph/Graph-Link/demos/RollBall-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cuda.h5 \
  --control-mode pd_joint_delta_pos \
  --max-episode-steps 80 \
  --sim-backend physx_cuda \
  --total-iters 100000 \
  --batch-size 1024 \
  --eval-freq 5000 \
  --orth-coef 1e-4 \
  --seed 1 \
  --exp-name RollBall-v1__abd_all_direct_d256_a12__s1 \
  --track \
  --wandb-project-name ManiSkill
```
