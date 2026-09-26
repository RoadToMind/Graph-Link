# Official Diffusion Policy + ABD root condition

## Base assumptions

- `train.py` is the unchanged ManiSkill state-observation diffusion-policy
  baseline from commit `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`.
- `train_abd.py` is the comparison variant. It keeps the official data loader,
  normalization, U-Net, scheduler, optimizer, EMA, evaluator, and checkpointing.
- ABD uses the Panda arm only: `panda_link0` plus the seven arm links. The raw
  flattened observation and the complete eight-dimensional environment action
  are unchanged.
- Only the post-message-passing root feature is concatenated to each raw state
  before the observation horizon is flattened.
- Both commands below evaluate with sparse reward because the official state
  training script fixes `reward_mode="sparse"`.

Run the commands from this directory:

```bash
cd /home/fawad/Graph/Graph-Link/examples/baselines/diffusion_policy
```

## Official baseline

```bash
uv run python train.py \
  --env-id RollBall-v1 \
  --demo-path /home/fawad/Graph/Graph-Link/demos/RollBall-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cuda.h5 \
  --control-mode pd_joint_delta_pos \
  --max-episode-steps 80 \
  --sim-backend physx_cuda \
  --total-iters 100000 \
  --batch-size 1024 \
  --eval-freq 5000 \
  --seed 1 \
  --exp-name RollBall-v1__official_diffusion__seed1 \
  --track \
  --wandb-project-name ManiSkill-Diffusion-ABD
```

## ABD root-conditioned variant

```bash
uv run python train_abd.py \
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
  --exp-name RollBall-v1__official_diffusion_abd_root__seed1 \
  --track \
  --wandb-project-name ManiSkill-Diffusion-ABD
```

For a fair comparison, run both commands with the same seeds (for example 1,
2, and 3), data, and arguments. Compare `eval/success_once` and
`eval/success_at_end` as the primary metrics. `losses/total_loss` is not directly
comparable because the ABD run includes the weighted orthogonality term; use
`losses/diffusion_loss` to inspect its denoising objective separately.

The official trainer performs evaluation at iteration zero before writing the
first training metrics. W&B may therefore show only the System tab until that
initial evaluation finishes.
