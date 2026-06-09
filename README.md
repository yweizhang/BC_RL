# Hybrid Control via Behavior Cloning and Residual RL

This repository contains two related experiments for studying hybrid control:

1. **Strong BC setting**: Behavior Cloning already learns a good nominal controller.
2. **Imperfect BC setting**: Behavior Cloning is intentionally limited, and residual RL is used to improve it.

The main idea is:

```text
final_action = BC_action(state) + residual_RL_action(state)
```

The BC policy is trained first and then frozen. PPO or SAC only learns the residual correction.

## Files

| File | Purpose |
|---|---|
| `BC_RL_project.ipynb` | Strong-BC experiment. BC learns a good baseline policy; PPO/SAC residual policies are evaluated mainly as refinement/correction methods. |
| `imperfect_bc_residual.ipynb` | Imperfect-BC experiment. BC is intentionally trained from limited demonstrations, then PPO/SAC residual policies improve performance, especially under disturbance. |

## Experiment 1: Strong BC

Notebook:

```text
BC_RL_project.ipynb
```

This version follows the original proposal more closely:

```text
MLP BC + PPO residual
ActionChunk BC + SAC residual
```

In this setting, BC can already perform well on `Pendulum-v1`. As a result, residual RL has limited room to improve the clean environment because the optimal residual is often close to zero.

This experiment is useful for showing:

- BC can provide a strong nominal controller.
- PPO can be used as a baseline residual corrector.
- SAC can be used as a more advanced off-policy residual corrector.
- If BC is already very strong, residual RL may not significantly improve clean-task reward.

## Experiment 2: Imperfect BC + Residual RL

Notebook:

```text
imperfect_bc_residual.ipynb
```

This is the recommended main-result notebook.

Here, BC is intentionally trained with limited demonstrations:

```python
EXPERT_TRANSITIONS = 5_000
BC_EPOCHS = 30
```

Then an actuator disturbance is added:

```python
ACTION_BIAS = 0.25
ACTION_NOISE_STD = 0.01
```

The goal is not to prove that residual RL always beats a strong BC policy. Instead, the goal is to show that residual RL can improve a frozen, imperfect BC policy under distribution shift or execution bias.

The current tuned setting uses:

```python
RUN_TRAINING = True
PPO_TIMESTEPS = 300_000
SAC_TIMESTEPS = 250_000
SAC_BOOTSTRAP_TRANSITIONS = 5_000
RESIDUAL_EPSILON = 0.25
RESIDUAL_LAMBDA = 0.0005
```

SAC also warm-starts its replay buffer with expert residual targets before online training.

## Previous Imperfect-BC Result

One completed run produced:

| Method | Clean reward | Disturbed reward |
|---|---:|---:|
| Weak BC only | `-1313.96 +/- 192.74` | `-1172.45 +/- 339.17` |
| Weak BC + PPO residual | `-1258.18 +/- 225.90` | `-1101.61 +/- 327.72` |
| Weak BC + SAC residual | `-743.20 +/- 179.10` | `-453.05 +/- 364.77` |

Higher reward is better. This result shows:

- PPO residual gives a modest improvement over weak BC.
- SAC residual gives a large improvement, especially in the disturbed environment.
- Residual RL is most useful when BC is imperfect or when execution conditions shift.

## Google Drive Checkpoints

The imperfect-BC notebook can save and load checkpoints from Google Drive.

Expected Drive directory:

```text
/content/drive/MyDrive/imperfect_bc/outputs_imperfect_bc
```

Typical saved files:

```text
weak_bc_expert.pth
ppo_residual_imperfect_bc.zip
sac_residual_imperfect_bc.zip
best_ppo_imperfect_bc/best_model.zip
best_sac_imperfect_bc/best_model.zip
imperfect_bc_residual_results.npz
```

If checkpoints already exist and you only want to re-evaluate:

```python
RUN_TRAINING = False
```

If you want to retrain from scratch:

```python
RUN_TRAINING = True
```

## How to Run in Colab

Open either notebook in Google Colab.

Install dependencies:

```python
!pip install -q "stable-baselines3[extra]" gymnasium tensorboard
```

For the imperfect-BC notebook, run all cells. If using Drive checkpoints, make sure Google Drive is mounted and the output directory exists.

## Interpretation

The two notebooks tell complementary stories:

```text
Strong BC:
BC is already good, so residual RL has limited room to improve clean performance.

Imperfect BC:
BC is weak or shifted, so residual RL can learn meaningful corrections.
```

This supports the main project hypothesis: BC provides a useful prior, and residual RL can refine or recover the policy when the BC controller is imperfect or faces distribution shift.
