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

### Strong-BC Result

The BC policies already perform well in the nominal environment:

| BC policy | Clean reward |
|---|---:|
| Single-step MLP BC | `-128.78 +/- 113.26` |
| Action-chunk BC | `-128.71 +/- 113.22` |

In a state-dependent disturbed setting, the strong BC baseline degrades, but a small correction is enough to recover much of the loss:

| Method | Clean reward | Disturbed reward |
|---|---:|---:|
| Frozen strong BC | `-175.56 +/- 109.31` | `-339.76 +/- 357.50` |
| Prior compensation only | `-` | `-195.28 +/- 157.01` |
| Strong BC + PPO residual | `-175.47 +/- 109.40` | `-195.43 +/- 157.87` |

This result illustrates why residual RL is harder to showcase when BC is already strong: the clean-task residual should be close to zero, and the main benefit appears under disturbance or execution shift.

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

The current history-stack setting uses:

```python
RETRAIN_BC = False
RETRAIN_RL = True
PPO_TIMESTEPS = 300_000
SAC_TIMESTEPS = 250_000
SAC_BOOTSTRAP_TRANSITIONS = 5_000
HISTORY_LEN = 3
RESIDUAL_EPSILON = 0.25
RESIDUAL_LAMBDA = 0.0005
RESIDUAL_SMOOTH_LAMBDA = 0.0005
```

The residual policy observes a short history of recent observations, while BC still sees only the current observation. A smoothness penalty is added to discourage jittery residual corrections. SAC also warm-starts its replay buffer with expert residual targets before online training.

## Imperfect-BC Result

The current history-stack run produced:

| Method | Clean reward | Disturbed reward |
|---|---:|---:|
| Weak BC only | `-1411.21 +/- 321.06` | `-1358.26 +/- 355.32` |
| Weak BC + PPO residual | `-1184.68 +/- 179.91` | `-1144.68 +/- 228.94` |
| Weak BC + SAC residual | `-961.32 +/- 163.13` | `-585.67 +/- 450.55` |

Higher reward is better. This result shows:

- Observation history and residual smoothing make PPO improve more clearly over weak BC.
- SAC residual still gives the largest improvement, especially in the disturbed environment.
- Residual RL is most useful when BC is imperfect or when execution conditions shift.

## Interpretation

The two notebooks tell complementary stories:

```text
Strong BC:
BC is already good, so residual RL has limited room to improve clean performance.

Imperfect BC:
BC is weak or shifted, so residual RL can learn meaningful corrections.
```

This supports the main project hypothesis: BC provides a useful prior, and residual RL can refine or recover the policy when the BC controller is imperfect or faces distribution shift.
