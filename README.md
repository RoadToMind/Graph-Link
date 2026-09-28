# Graph-Link

The [main blog for this implementation repo](https://roadtomind.github.io/Graph-Overdose/) explains the ABD-Net PPO and Diffusion Policy experiments.

## Files

- `main.py`: MLP PPO baseline for ManiSkill locomotion tasks.
- `ppo.py`: PPO with the ABD-Net actor.
- `abdnet_actor.py`: ABD-Net graph, message passing, actor, and orthogonality loss used by `ppo.py`.
- `examples/baselines/diffusion_policy/train.py`: NormalDiff, the state-based Diffusion Policy baseline.
- `examples/baselines/diffusion_policy/train_abd.py`: AllComb, which adds the ABD root feature to the state observation.
- `examples/baselines/diffusion_policy/train_only.py`: GraphOnly, which conditions on the ABD root feature.
- `examples/baselines/diffusion_policy/train_abd_all.py`: separate all-node ABD variant.
- `examples/baselines/diffusion_policy/diffusion_policy/`: shared U-Net, graph encoders, dataset, and evaluation code.
- `demos/`: ManiSkill demonstration metadata. The matching `.h5` trajectory files are needed for training and are not committed.
- `runs/` and `wandb/`: generated checkpoints, videos, and logs.
- `tests/`: diffusion-policy ABD checks.

## Run

Install the root dependencies with `uv sync`. The commands below assume a Bash shell.

### PPO

Run these from the repository root. Part I used seeds 1, 2, and 3; the commands show seed 1. Change `seed` and rerun for the other seeds. Options equal to script defaults are omitted.

```bash
seed=1
uv run python main.py --env-id MS-HopperHop-v1 \
  --exp-name "hopperhop_mlp_matched_seed${seed}" --seed "$seed" \
  --total-timesteps 50000000 --num-eval-steps 128 --track

uv run python ppo.py --env-id MS-HopperHop-v1 \
  --exp-name "hopperhop_abdnet_matched_seed${seed}" --seed "$seed" \
  --total-timesteps 50000000 --num-eval-steps 128 \
  --gamma 0.99 --ent-coef 0.01 --orth-coef 1e-4 \
  --abd-d 256 --abd-phi-hidden 64 --track

uv run python ppo.py --env-id MS-HumanoidWalk-v1 --seed "$seed" \
  --num-steps 50 --num-minibatches 50 \
  --gamma 0.97 --ent-coef 0.001 --orth-coef 1e-4 \
  --target-kl 0.1 --track
```

### Diffusion Policy

Run from `examples/baselines/diffusion_policy`. NormalDiff is `train.py`, AllComb is `train_abd.py`, and GraphOnly is `train_only.py`. Part II used only seed 1. The GraphOnly commands below use the recorded run settings for RollBall, LiftPegUpright, and PushT. Put each matching ManiSkill `.h5` file beside its `.json` metadata first.

```bash
cd examples/baselines/diffusion_policy
common=(--control-mode pd_joint_delta_pos --sim-backend physx_cuda --total-iters 100000 --seed 1)
rollball=../../../demos/RollBall-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cuda.h5
liftpeg=../../../demos/LiftPegUpright-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cuda.h5
pusht=../../../demos/PushT-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cuda.h5

# RollBall-v1
uv run python train_only.py --env-id RollBall-v1 \
  --demo-path "$rollball" --control-mode pd_joint_delta_pos --sim-backend physx_cuda \
  --max-episode-steps 80 --total-iters 100000 --batch-size 1024 --lr 1e-4 \
  --obs-horizon 2 --act-horizon 15 --pred-horizon 16 \
  --diffusion-step-embed-dim 64 --unet-dims 64 128 256 --n-groups 8 \
  --abd-feature-dim 256 --abd-phi-hidden-dim 64 --orth-coef 1e-4 \
  --log-freq 1000 --eval-freq 5000 --num-eval-episodes 100 --num-eval-envs 10 \
  --seed 1 --exp-name RollBall-graphOnly --track --wandb-project-name ManiSkill

# LiftPegUpright-v1
uv run python train_only.py --env-id LiftPegUpright-v1 \
  --demo-path "$liftpeg" --control-mode pd_joint_delta_pos --sim-backend physx_cuda \
  --max-episode-steps 100 --total-iters 100000 --batch-size 1024 --lr 1e-4 \
  --obs-horizon 2 --act-horizon 15 --pred-horizon 16 \
  --diffusion-step-embed-dim 64 --unet-dims 64 128 256 --n-groups 8 \
  --abd-feature-dim 256 --abd-phi-hidden-dim 64 --orth-coef 1e-4 \
  --log-freq 1000 --eval-freq 5000 --num-eval-episodes 100 --num-eval-envs 10 \
  --seed 1 --exp-name LiftPegUpright-graphOnly --track --wandb-project-name ManiSkill

# PushT-v1
uv run python train_only.py --env-id PushT-v1 \
  --demo-path "$pusht" --control-mode pd_joint_delta_pos --sim-backend physx_cuda \
  --max-episode-steps 150 --total-iters 100000 --batch-size 1024 --lr 1e-4 \
  --obs-horizon 2 --act-horizon 15 --pred-horizon 16 \
  --diffusion-step-embed-dim 64 --unet-dims 64 128 256 --n-groups 8 \
  --abd-feature-dim 256 --abd-phi-hidden-dim 64 --orth-coef 1e-4 \
  --log-freq 1000 --eval-freq 5000 --num-eval-episodes 100 --num-eval-envs 10 \
  --seed 1 --exp-name PushT-GraphOnly --track --wandb-project-name ManiSkill

# Separate all-node experiment
uv run python train_abd_all.py --env-id RollBall-v1 --demo-path "$rollball" --max-episode-steps 80 "${common[@]}"
```

To run AllComb, change `train_only.py` to `train_abd.py` and update `--exp-name`. To run NormalDiff, change it to `train.py`, remove `--abd-feature-dim`, `--abd-phi-hidden-dim`, and `--orth-coef`, and update `--exp-name`.

Each Diffusion Policy script evaluates 100 episodes every 5,000 iterations by default. The ABD scripts use 256-dimensional node features by default.
