# EV6 Full-Car Pitch PINN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Ax-only pitch ODE with a four-corner EV6 full-car PINN driven by the 0.1 g ACC wheel-center signals and trained from sparse pitch-rate data.

**Architecture:** A validated four-corner scenario loader feeds differentiable signal interpolators and a three-state heave/roll/pitch network. The physics layer derives corner motions and suspension forces, then minimizes heave, roll, and pitch residuals while bounded front/rear wheel rates are identified. Evaluation writes learned physical parameters, residuals, sparse/hidden metrics, and comparison artifacts through the existing CLI.

**Tech Stack:** Python 3, NumPy, SciPy MAT I/O, PyTorch float64/autograd, PyYAML, Matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-10-12-ev6-full-car-pitch-pinn-design.md`

## Global Constraints

- Use only `E:/PINN/PINN_file/src_cm4sl/Data/3.ACC/0.1g_EV6.mat` for the native verification run.
- Neural-network features are time, Ax, and four `VelZ_*_WC` signals; Pitch and Pitch Rate must never enter the feature tensor.
- `PITCH_RATE` is sparse supervision and held-out evaluation truth only.
- EV6 `ms=1711.4`, `Ix=877.833`, `Iy=2813.085`, `lf=1.184`, `lr=1.476`, `ttf=ttr=1.570`, and `h=0.625` remain fixed.
- Train only effective `k_front`, `k_rear`, `c_front`, and `c_rear`; skyhook remains disabled at zero.
- Corner order is always FL, FR, RL, RR.
- Preserve the `run_training_from_config` and `run_iteration_from_config` entry points.
- Reject the former normalized `damping_rate_s_inv`, `stiffness_rate_s2`, and `ax_gain_rad_per_m` schema with an actionable error.
- The workspace is not a Git repository; task checkpoints are test results instead of commits unless Git is initialized externally.

## Review Focus

- Duplicate or non-monotonic MAT timestamps: preprocessing must retain one complete last row per timestamp and return strictly increasing time; covered in Task 1.
- A NaN in any one corner signal: the common finite-row mask must remove the whole sample without misaligning channels; covered in Task 1.
- Constant Ax or wheel-velocity input: normalization must stay finite and use a safe unit scale; covered in Task 2.
- Corner-order or lever-arm sign mistakes: pure roll, pitch, and one-corner tests must lock FL/FR/RL/RR geometry; covered in Task 3.
- Collocation times at or infinitesimally outside signal endpoints: interpolation must clamp to endpoint values without invalid indices; covered in Task 2.

---

### Task 1: Four-Corner Scenario Data Contract

**Files:**
- Modify: `data/scenario.py`
- Modify: `data/mat_loader.py`
- Modify: `data/preprocess.py`
- Modify: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: YAML key mapping with `time`, `ax`, `pitch_rate`, `wheel_velocity`, `damper_length`, and `damper_velocity`; each corner mapping is keyed by `fl`, `fr`, `rl`, `rr`.
- Produces: `ScenarioData(time_s, ax_mps2, pitch_rate_radps, wheel_velocity_mps, damper_length_m, damper_velocity_mps, name, vehicle)` with every corner matrix shaped `[N, 4]`; `preprocess_scenario(..., baseline_samples: int)` returns aligned, strictly increasing, baseline-corrected data.

- [ ] **Step 1: Write failing loader and validation tests**

Add tests named `test_loads_four_corner_signals_in_fl_fr_rl_rr_order`, `test_missing_corner_key_is_reported`, `test_nonfinite_corner_row_is_removed_from_every_signal`, and `test_all_signal_lengths_must_match`. Assert exact `[N, 4]` shapes, values, and missing MAT variable names.

- [ ] **Step 2: Run the new data tests and verify failure**

Run: `python -m pytest Pitch_PINN/tests/test_preprocess.py -v`

Expected: FAIL because `ScenarioData` and `load_mat_scenario` do not accept four-corner arrays.

- [ ] **Step 3: Expand `ScenarioData` and `SparseScenario`**

Implement corner-matrix validation in `data/scenario.py`. Keep sparse accessors limited to sparse pitch-rate labels; expose the full exogenous and auxiliary matrices through `SparseScenario.full`.

- [ ] **Step 4: Expand `load_mat_scenario`**

Load all configured signals without guessing names, stack columns in FL/FR/RL/RR order, and raise one actionable error listing every absent mapping or MAT variable.

- [ ] **Step 5: Update common-mask preprocessing and baseline correction**

Apply one finite mask across scalar and corner signals, stable-sort, keep the last complete duplicate-time row, subtract the initial damper-length mean over `baseline_samples`, and preserve damper velocity without bias removal unless explicitly configured.

- [ ] **Step 6: Verify Task 1**

Run: `python -m pytest Pitch_PINN/tests/test_preprocess.py -v`

Expected: all tests pass, including the review-focus duplicate-time and single-corner-NaN cases.

- [ ] **Step 7: Record checkpoint**

Record the passing pytest summary in the implementation log; do not initialize Git solely for this task.

### Task 2: Differentiable Input Signals and Three-State Network

**Files:**
- Create: `models/input_signals.py`
- Modify: `models/state_network.py`
- Modify: `models/pitch_pinn.py`
- Create: `tests/test_input_signals.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: strictly increasing `time_s`, `[N, 5]` exogenous matrix ordered Ax, FL, FR, RL, RR, and model output scales.
- Produces: `FeatureNormalizer.fit(values: np.ndarray)`, `MultiSignalInterpolator(time_s, values)`, `FullCarStateNetwork(...).forward(features) -> Tensor[N,3]`, and `FullCarPitchPINN.predict_with_derivatives(time_s) -> FullCarPrediction` containing state, first derivative, second derivative, features, and wheel displacement.

- [ ] **Step 1: Write failing interpolation and leakage tests**

Add `test_multisignal_interpolator_matches_endpoints_and_midpoint`, `test_interpolator_clamps_just_outside_endpoints`, `test_constant_channel_normalization_is_finite`, and `test_feature_tensor_contains_only_time_ax_and_four_wheel_velocities`. Assert six feature columns and explicitly assert that changing supplied truth pitch rate cannot change features.

- [ ] **Step 2: Run Task 2 tests and verify failure**

Run: `python -m pytest Pitch_PINN/tests/test_input_signals.py Pitch_PINN/tests/test_model.py -v`

Expected: FAIL because multi-signal interpolation and full-car prediction types do not exist.

- [ ] **Step 3: Implement `FeatureNormalizer` and `MultiSignalInterpolator`**

Use piecewise-linear torch operations, safe scale `1.0` for constant channels, endpoint clamping, registered buffers for source data, and float64-compatible tensors.

- [ ] **Step 4: Replace the scalar pitch network with `FullCarStateNetwork`**

Accept six normalized features and return scaled `[z_s, roll, pitch]`. Configure `heave_scale_m`, `roll_scale_rad`, and `pitch_scale_rad` separately.

- [ ] **Step 5: Replace `PitchPINN` behavior with `FullCarPitchPINN`**

Build features by interpolating signals from differentiable time inside `forward`. Integrate four wheel velocities with a precomputed cumulative trapezoidal displacement interpolator. Obtain state velocity and acceleration with autograd and expose pitch rate as column 2 of state velocity.

- [ ] **Step 6: Verify Task 2**

Run: `python -m pytest Pitch_PINN/tests/test_input_signals.py Pitch_PINN/tests/test_model.py -v`

Expected: all tests pass, including constant-channel and endpoint cases.

- [ ] **Step 7: Record checkpoint**

Record the passing pytest summary in the implementation log.

### Task 3: Four-Corner Kinematics, Bounded Rates, and Full-Car Residuals

**Files:**
- Create: `physics/full_car.py`
- Modify: `models/trainable_parameters.py`
- Create: `tests/test_full_car.py`
- Create: `tests/test_parameters.py`

**Interfaces:**
- Produces: `FullCarSpec`, `SuspensionRates`, `SuspensionRateBounds`, `BoundedSuspensionRates.values()`, `corner_kinematics(state, state_rate, wheel_z, wheel_velocity, spec)`, `corner_forces(stroke, stroke_rate, rates)`, and `full_car_residual(prediction, ax, spec, rates)`.
- Corner coordinates are generated once from the fixed spec in FL, FR, RL, RR order.

- [ ] **Step 1: Write failing geometry and physics tests**

Add tests for pure heave, pure roll, pure pitch, one-corner wheel displacement, left/right symmetry, front force causing the documented pitch-moment sign, rear force causing the opposite sign, fixed `ms*h*Ax` moment, zero equilibrium residual, invalid fixed vehicle values, parameter bounds, and parameter regularization.

- [ ] **Step 2: Run Task 3 tests and verify failure**

Run: `python -m pytest Pitch_PINN/tests/test_full_car.py Pitch_PINN/tests/test_parameters.py -v`

Expected: FAIL because full-car functions and suspension-rate parameters do not exist.

- [ ] **Step 3: Implement immutable physical value types**

Define validated `FullCarSpec` with the eight fixed EV6 values and validated positive `SuspensionRates`/increasing `SuspensionRateBounds` for front/rear K and C.

- [ ] **Step 4: Implement bounded trainable suspension rates**

Generalize the existing inverse-sigmoid technique to four physical parameters. `values()` returns `SuspensionRates`; `regularization_loss()` uses normalized distance from configured initial rates.

- [ ] **Step 5: Implement corner kinematics and force mapping**

Use the section-3 frame convention from the spec and effective wheel rates. Keep junction ratios out of force calculation; expose a separate conversion helper for auxiliary damper comparison.

- [ ] **Step 6: Implement normalized full-car residuals**

Return named heave, roll, and pitch residual tensors plus raw corner forces. Include the fixed longitudinal moment and no gravity/static term.

- [ ] **Step 7: Verify Task 3**

Run: `python -m pytest Pitch_PINN/tests/test_full_car.py Pitch_PINN/tests/test_parameters.py -v`

Expected: all tests pass with the exact corner-order and moment-sign assertions.

- [ ] **Step 8: Record checkpoint**

Record the passing pytest summary in the implementation log.

### Task 4: Full-Car Losses and Trainer

**Files:**
- Modify: `training/losses.py`
- Modify: `training/trainer.py`
- Create: `tests/test_full_car_losses.py`
- Modify: `tests/test_training.py`

**Interfaces:**
- Consumes: `FullCarPitchPINN`, sparse `ObservationBatch`, full auxiliary damper arrays, collocation times, `FullCarSpec`, junction ratios, and `FullCarLossWeights`.
- Produces: `FullCarLossResult(total, pitch_data, physics_heave, physics_roll, physics_pitch, damper_length, damper_velocity, initial, regularization)` and `TrainingResult` containing bounded learned `SuspensionRates`.

- [ ] **Step 1: Write failing loss-component tests**

Add tests that each isolated mismatch activates only its named loss, physics residuals are scale-normalized, initial loss contains three states and three rates, sparse pitch supervision ignores hidden points, and no pitch truth is passed into model feature construction.

- [ ] **Step 2: Run Task 4 tests and verify failure**

Run: `python -m pytest Pitch_PINN/tests/test_full_car_losses.py Pitch_PINN/tests/test_training.py -v`

Expected: FAIL because full-car weights and result fields do not exist.

- [ ] **Step 3: Implement `FullCarLossWeights` and `compute_full_car_losses`**

Keep every loss independently named and weighted. Convert calculated corner travel to damper travel using configured front/rear junction ratios and configured polarity before auxiliary comparison.

- [ ] **Step 4: Migrate trainer to full-car dependencies**

Create full-trace auxiliary tensors, collocation times, optimizer parameter list, per-component CSV history, best-state retention, optional L-BFGS, and `SuspensionRates` extraction.

- [ ] **Step 5: Verify Task 4**

Run: `python -m pytest Pitch_PINN/tests/test_full_car_losses.py Pitch_PINN/tests/test_training.py -v`

Expected: all tests pass and a two-epoch synthetic training test writes `model.pt` and component loss history.

- [ ] **Step 6: Record checkpoint**

Record the passing pytest summary in the implementation log.

### Task 5: Configuration and End-to-End Training Pipeline

**Files:**
- Modify: `config/tesla_ev6.yaml`
- Modify: `train_inverse_pinn.py`
- Modify: `physics/reference_model.py`
- Modify: `training/evaluate.py`
- Create: `tests/test_pipeline.py`
- Modify: `tests/test_reference_model.py`

**Interfaces:**
- Consumes: the new YAML sections `data.keys`, `model.output_scales`, `model.parameters`, `vehicle`, `auxiliary`, and expanded loss weights.
- Produces: unchanged `run_training_from_config(config, epochs=None, output_dir=None) -> PipelineResult`, plus full-car metrics/artifacts and an actionable legacy-schema error.

- [ ] **Step 1: Write failing config and artifact tests**

Add `test_legacy_parameter_schema_is_rejected`, `test_default_config_has_all_ev6_signal_keys`, `test_pipeline_uses_fixed_ev6_vehicle_values`, `test_pipeline_writes_full_car_parameters_and_residuals`, and `test_pipeline_features_exclude_pitch_truth`.

- [ ] **Step 2: Run Task 5 tests and verify failure**

Run: `python -m pytest Pitch_PINN/tests/test_pipeline.py Pitch_PINN/tests/test_reference_model.py -v`

Expected: FAIL because the default YAML and pipeline still use normalized pitch coefficients.

- [ ] **Step 3: Migrate `tesla_ev6.yaml`**

Map the exact EV6 MAT names, add fixed EV6 spec and junction ratios (`front=0.614`, `rear=0.797`), physical parameter bounds/initials, output scales, residual scales, auxiliary polarity, and all loss weights. Keep CarMaker dry-run settings.

- [ ] **Step 4: Migrate pipeline construction and validation**

Validate the schema before loading data, instantiate new data/model/physics/trainer types, and preserve CLI arguments. Raise a message naming the replacement parameters if old keys are present.

- [ ] **Step 5: Replace Ax-only reference evaluation**

Remove the integrated simple-ODE comparison from the default path. Save PINN pitch-rate prediction, observed mask, named full-car residuals, and learned physical rates. Keep the existing sparse/full/hidden RMSE and peak metrics for the PINN prediction.

- [ ] **Step 6: Verify Task 5**

Run: `python -m pytest Pitch_PINN/tests/test_pipeline.py Pitch_PINN/tests/test_reference_model.py -v`

Expected: all tests pass and synthetic end-to-end output contains `metrics.json`, `reference.csv`, `physics_residuals.csv`, `loss_history.csv`, `reference_comparison.png`, and `resolved_config.yaml`.

- [ ] **Step 7: Record checkpoint**

Record the passing pytest summary in the implementation log.

### Task 6: CarMaker Parameter Update Compatibility

**Files:**
- Modify: `carmaker/parameter_mapper.py`
- Modify: `run_iteration.py`
- Modify: `tests/test_carmaker.py`

**Interfaces:**
- Consumes: learned `SuspensionRates`.
- Produces: `ParameterUpdate(k_front_npm, k_rear_npm, c_front_nspm, c_rear_nspm)` serialized to `parameter_update.json`; `run_iteration_from_config` remains unchanged to callers.

- [ ] **Step 1: Write failing parameter-update tests**

Assert exact JSON names and physical values, absence of former normalized coefficients, dry-run behavior, and iteration output creation.

- [ ] **Step 2: Run Task 6 tests and verify failure**

Run: `python -m pytest Pitch_PINN/tests/test_carmaker.py -v`

Expected: FAIL because `ParameterUpdate` still represents `c_eff`, `k_eff`, and `g_ax`.

- [ ] **Step 3: Replace mapper types and iteration wiring**

Serialize the four effective wheel rates directly and remove `VehiclePitchSpec` conversion from the default full-car iteration path. Do not perform a live CarMaker run unless both the CLI and YAML explicitly enable it.

- [ ] **Step 4: Verify Task 6**

Run: `python -m pytest Pitch_PINN/tests/test_carmaker.py -v`

Expected: all tests pass and dry-run remains the default.

- [ ] **Step 5: Record checkpoint**

Record the passing pytest summary in the implementation log.

### Task 7: Documentation, Regression Suite, and Native EV6 Smoke Run

**Files:**
- Modify: `README.md`
- Modify: `tests/test_half_car.py` or remove obsolete assertions after equivalent coverage exists in `tests/test_full_car.py`
- Create: `docs/full_car_model.md`

**Interfaces:**
- Documents: exact native commands, signal roles, output parameter units, sign convention, sparse-data meaning, and the EV6-only limitation.

- [ ] **Step 1: Update user documentation**

Document that stroke signals are auxiliary observations, pitch is excluded from inputs, `VelZ_*_WC` drives the four-corner model, and this run reconstructs EV6 rather than calibrating Tesla Y.

- [ ] **Step 2: Run the complete regression suite**

Run: `python -m pytest Pitch_PINN/tests -v`

Expected: all tests pass; no default-path test depends on the Ax-only ODE.

- [ ] **Step 3: Run a zero-epoch native data validation**

Run: `python Pitch_PINN/train_inverse_pinn.py --config Pitch_PINN/config/tesla_ev6.yaml --epochs 0 --output-dir Pitch_PINN/outputs/full_car_native_validation`

Expected: exit code 0; all required artifacts are written; learned rates stay inside configured bounds.

- [ ] **Step 4: Run a short native optimization smoke test**

Run: `python Pitch_PINN/train_inverse_pinn.py --config Pitch_PINN/config/tesla_ev6.yaml --epochs 5 --output-dir Pitch_PINN/outputs/full_car_smoke`

Expected: exit code 0, finite loss values, finite hidden RMSE, and physical parameter JSON fields in N/m and N s/m.

- [ ] **Step 5: Run the iteration dry-run**

Run: `python Pitch_PINN/run_iteration.py --config Pitch_PINN/config/tesla_ev6.yaml --epochs 1 --output-root Pitch_PINN/outputs/full_car_iteration_smoke`

Expected: exit code 0, `carmaker_live=False`, and a four-parameter `parameter_update.json`.

- [ ] **Step 6: Inspect artifacts**

Open `reference_comparison.png`, inspect `metrics.json`, `physics_residuals.csv`, and `parameter_update.json`, and record any numerical limitation without changing acceptance criteria merely to make the smoke run pass.

- [ ] **Step 7: Record final checkpoint**

Record the commands, test counts, output directories, and any remaining accuracy risk. The structural migration is complete only if every required artifact exists and all tests pass.
