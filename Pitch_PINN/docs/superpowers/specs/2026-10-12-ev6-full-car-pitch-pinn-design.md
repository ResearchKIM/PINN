# EV6 0.1 g Full-Car Pitch PINN Design

## 1. Purpose

Replace the current time-only, second-order pitch reference PINN with a
four-corner sprung-body model driven by the EV6 0.1 g ACC signals. The first
milestone is to demonstrate that sparse EV6 pitch-rate observations plus a
full-car physics loss can reconstruct the full EV6 pitch-rate trace and
identify physically interpretable front/rear suspension rates.

This milestone is an EV6 reconstruction and parameter-identification test. It
does not yet claim to identify Tesla Model Y parameters because no Tesla Y
response data is used.

## 2. Fixed scope

- Dataset: `E:/PINN/PINN_file/src_cm4sl/Data/3.ACC/0.1g_EV6.mat`
- Scenario: longitudinal 0.1 g acceleration/deceleration
- Sparse supervised target: `PITCH_RATE`
- Exogenous signals: `Sim_Time`, `Ax`, and four `VelZ_*_WC` signals
- Auxiliary observations: four `DampLength_*` and four `DampVel_*` signals
- Neural-network outputs: sprung-body heave, roll, and pitch states
- Trainable physical parameters: effective front/rear wheel stiffness and
  damping (`k_front`, `k_rear`, `c_front`, `c_rear`)
- Skyhook: disabled and fixed to zero
- Existing CarMaker iteration interface remains available but is not required
  for the first Python-only verification run.

Pitch and pitch rate are not neural-network inputs. `PITCH_RATE` is used only
as the sparse supervised target and for held-out evaluation.

## 3. Vehicle parameters and conventions

The EV6 values already defined in
`E:/PINN/PINN_file/src_cm4sl/Initialization.m` are fixed during training:

| Parameter | Value |
|---|---:|
| Sprung mass `ms` | 1711.4 kg |
| Roll inertia `Ix` | 877.833 kg m^2 |
| Pitch inertia `Iy` | 2813.085 kg m^2 |
| CG to front axle `lf` | 1.184 m |
| CG to rear axle `lr` | 1.476 m |
| Front track `ttf` | 1.570 m |
| Rear track `ttr` | 1.570 m |
| CG height `h` | 0.625 m |

The model uses an ISO-like body frame: `x` forward, `y` left, and `z` upward.
Positive pitch is defined by the implemented right-hand-rule convention. MAT
signal polarity is handled by explicit configuration multipliers rather than
hidden sign changes. Unit tests verify every corner's lever-arm sign.

## 4. Data contract and preprocessing

`ScenarioData` is expanded from three signals to named four-corner arrays:

- `wheel_velocity_mps`: shape `[N, 4]`, ordered FL, FR, RL, RR
- `damper_length_m`: shape `[N, 4]`, ordered FL, FR, RL, RR
- `damper_velocity_mps`: shape `[N, 4]`, ordered FL, FR, RL, RR
- existing `time_s`, `ax_mps2`, and `pitch_rate_radps`

Loading fails with a clear missing-key error if any configured signal is
absent. Preprocessing applies one common finite-row mask, stable time sorting,
duplicate-time collapse, and optional bias removal. The existing MAT file has
duplicate timestamps; the last sample at each timestamp is retained as in the
current implementation.

Damper length is converted to relative travel by subtracting the mean over a
configurable initial baseline window. Wheel displacement is obtained by
trapezoidal integration of each `VelZ_*_WC` channel and is initialized at zero.
All normalizers are fitted only on the full input trajectory and stored with
the trained model artifacts.

Sparse selection applies to pitch-rate supervision. Exogenous wheel signals
remain available at every collocation time because they are model inputs, not
labels. Auxiliary stroke losses may use the full trace; their weight is
separate from the sparse pitch-rate data weight.

## 5. Model architecture

### 5.1 Signal interpolation and features

A reusable piecewise-linear multi-signal interpolator evaluates `Ax` and the
four wheel-center velocities at arbitrary collocation times. The neural
network receives the normalized feature vector

`[time, Ax(t), VelZ_FL(t), VelZ_FR(t), VelZ_RL(t), VelZ_RR(t)]`.

The interpolation is performed inside the model from the differentiable time
tensor. Therefore automatic differentiation computes derivatives along the
measured trajectory rather than partial derivatives that ignore changing
exogenous signals.

### 5.2 State network

The MLP returns three smooth states:

`[z_s, roll, pitch]`.

Output scales for heave, roll, and pitch are configured separately. First and
second time derivatives are obtained with PyTorch automatic differentiation.
Pitch rate is the derivative of the predicted pitch angle; it remains a model
output, not an input feature.

### 5.3 Four-corner kinematics

Wheel displacement is the time integral of the four wheel-center velocities.
For small body angles, corner body displacement is calculated from heave,
roll, pitch, and fixed corner coordinates. Relative corner displacement and
velocity are then derived from body and wheel motion. This is the Python
equivalent of supplying `dzu` to the Simulink full-car reference block.

Measured `DampLength_*` and `DampVel_*` are auxiliary observations. Their
baseline-corrected shapes constrain the calculated relative motion. Front and
rear junction ratios from `Initialization.m` are configurable conversions for
this auxiliary comparison; suspension force parameters remain effective wheel
rates so that their reported units are N/m and N s/m.

## 6. Physics model

At each corner `i`, the vertical force on the sprung body is

`F_i = k_axle(i) * stroke_i + c_axle(i) * stroke_rate_i`.

Left and right rates are shared within an axle for this milestone. The model
forms three residuals:

- Heave: `ms * z_ddot - sum(F_i) = 0`
- Roll: `Ix * roll_ddot - sum(y_i * F_i) = 0`
- Pitch: `Iy * pitch_ddot + sum(x_i * F_i) + ms * h * Ax = 0`

These signs follow the frame convention in section 3 and are locked by unit
tests. Constant gravity/static-load terms are omitted because the model uses
motions relative to the initial equilibrium.

`k_front`, `k_rear`, `c_front`, and `c_rear` are positive bounded parameters.
They replace the current normalized `c_eff`, `k_eff`, and learned `g_ax`.
Longitudinal forcing is fixed by `ms * h * Ax`; it is not trainable.

## 7. Losses and training

The total loss contains independently configurable terms:

1. Sparse pitch-rate data loss against EV6 `PITCH_RATE`.
2. Full-car physics loss from normalized heave, roll, and pitch residuals.
3. Auxiliary damper-length loss.
4. Auxiliary damper-velocity loss.
5. Initial-state loss for heave, roll, pitch, and their rates.
6. Parameter regularization relative to configured initial rates.

Physics residuals are divided by representative force/moment scales before
MSE calculation so heave, roll, and pitch terms have comparable numerical
influence. Adam remains the default optimizer, with the existing optional
L-BFGS refinement. The lowest total-loss state is retained.

## 8. Outputs and acceptance criteria

Each run saves:

- resolved configuration
- model weights and feature-normalization values
- loss history with every component
- learned `k_front`, `k_rear`, `c_front`, and `c_rear` in physical units
- full-trace predicted pitch rate
- full-trace physics residuals
- comparison plot for EV6 ground truth, PINN prediction, and sparse points
- metrics split into observed and hidden samples

The first milestone passes when:

- all loader, kinematics, force/moment, preprocessing, and training smoke tests
  pass;
- a zero- or short-epoch CLI run completes and writes every artifact;
- no `Pitch` or `PITCH_RATE` value appears in the model feature tensor;
- the reported parameters remain inside configured physical bounds;
- hidden-sample pitch-rate RMSE is computed and displayed, without requiring a
  predetermined accuracy threshold for this structural migration.

Accuracy tuning is a separate follow-up after the new model is proven to run
correctly. This prevents loss-weight tuning from hiding implementation or sign
errors.

## 9. Compatibility and migration

The default `tesla_ev6.yaml` is migrated to the new EV6 full-car schema. Old
normalized parameter names are rejected with an actionable configuration
error rather than silently reinterpreted. `run_iteration.py` continues to call
the same top-level training function and consumes the new physical parameter
output through an updated mapper.

The old half-car modules may remain temporarily for historical tests, but the
default training path no longer imports or executes the simple Ax-only pitch
ODE. Documentation and examples use the new command and output names.

## 10. Test strategy

- Loader tests for all 15 required MAT mappings and missing-key failures.
- Preprocessing tests for common masking, duplicate timestamps, baseline
  removal, and corner ordering.
- Interpolator tests for exact endpoint and midpoint values.
- Kinematic tests for pure heave, pure roll, pure pitch, and one-corner wheel
  motion.
- Physics tests for symmetry, front/rear pitch-moment signs, fixed Ax moment,
  and zero residual at a constructed equilibrium.
- Leakage test proving pitch and pitch rate are absent from input features.
- Parameter-bound and freeze tests.
- End-to-end smoke test using a small synthetic dataset.
- Native execution smoke test against `0.1g_EV6.mat` with reduced epochs.

## 11. Known limitation

One ACC trajectory does not uniquely identify every real suspension component
or nonlinear damper map. The learned values are effective linear wheel rates
for this scenario. The design deliberately shares left/right parameters and
keeps geometry and inertias fixed to reduce non-identifiability. Additional
ACC levels and bump events can be added later without changing the four-corner
data interface.
