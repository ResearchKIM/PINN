# Sparse-Data Pitch PINN Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Tesla/EV6 pitch PINN demo that reconstructs the full EV6 Pitch Rate reference from sparse `ACC 0.1 g` observations plus a pitch physics residual, then exposes learned equivalent K/C values to a safe CarMaker dry-run iteration.

**Architecture:** Load and normalize the existing EV6 MAT signals into one typed scenario object, retain one observation every 40 samples, and train a PyTorch state network whose time derivatives obey a forced second-order pitch ODE. Keep the first result in equivalent rotational coefficients so no unverified Tesla specifications are invented; when the user supplies `I_y`, axle lever arms, and baseline suspension values, a separate mapper converts them to `alpha_K` and `alpha_C`. CarMaker integration is dry-run by default and never edits an installed vehicle model.

**Tech Stack:** Python 3.14, PyTorch, NumPy, SciPy, Matplotlib, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-09-29-pitch-pinn-design.md`

## Global Constraints

- The initial scenario is `E:/PINN/PINN_file/src_cm4sl/Data/3.ACC/0.1g_EV6.mat` with keys `Sim_Time`, `Ax`, and `PITCH_RATE`.
- Preserve all MAT samples for evaluation; train on deterministic sparse observations selected with stride 40, always including both endpoints.
- Treat `PITCH_RATE` as `rad/s`, `Ax` as `m/s^2`, and time as seconds; expose unit overrides in YAML but do not auto-guess units.
- Train positive bounded equivalent damping and stiffness coefficients; do not invent Tesla mass, inertia, CG, or suspension values.
- Skyhook is disabled and fixed to zero by default; the physics API must support enabling a bounded gain later.
- CarMaker runs in dry-run mode unless an explicit command, working project, target parameter file, and `dry_run: false` are all present.
- Do not modify the installed CarMaker `Demo_Tesla_Y` model.
- `E:/PINN` is not currently a Git repository; do not initialize one automatically. Commit steps are conditional on the user providing or initializing a repository.

## Review Focus

- Duplicate or non-finite MAT timestamps must be removed or rejected before interpolation; Task 1 tests both cases.
- A sparse stride larger than the signal length must still include the first and last samples without duplicate indices; Task 1 tests it.
- Constant or nearly constant time/signal ranges must not cause normalization division by zero; Task 3 tests finite output and derivatives.
- Missing Tesla physical values must result in a coefficient-only report, never fabricated `alpha_K`/`alpha_C`; Task 5 tests the omission path.
- A non-dry-run CarMaker request with incomplete command/target configuration must fail before any file change or process launch; Task 5 tests the guard.

---

### Task 1: Configuration and scenario data pipeline

**Files:**
- Create: `Pitch_PINN/__init__.py`
- Create: `Pitch_PINN/requirements.txt`
- Create: `Pitch_PINN/config/tesla_ev6.yaml`
- Create: `Pitch_PINN/data/__init__.py`
- Create: `Pitch_PINN/data/scenario.py`
- Create: `Pitch_PINN/data/mat_loader.py`
- Create: `Pitch_PINN/data/preprocess.py`
- Create: `Pitch_PINN/tests/test_preprocess.py`

**Interfaces:**
- Consumes: MAT keys `Sim_Time`, `Ax`, `PITCH_RATE`; YAML path and preprocessing settings.
- Produces: `ScenarioData`, `SparseScenario`, `load_mat_scenario(path, keys)`, `preprocess_scenario(data, settings)`, and `select_sparse_observations(data, stride)`.

- [ ] **Step 1: Add package requirements and default YAML**

Pin minimum dependencies in `requirements.txt` and set the actual EV6 MAT path/key mapping, `sparse_stride: 40`, seed `42`, output directory, physics coefficient bounds, training defaults, and `carmaker.dry_run: true` in `tesla_ev6.yaml`. Tesla physical fields must be explicit `null` values until measured values are supplied. Add the root package marker.

- [ ] **Step 2: Install the declared dependencies**

Run: `python -m pip install -r Pitch_PINN/requirements.txt`

Expected: exit code 0; `python -c "import torch, scipy, yaml, pytest"` exits 0.

- [ ] **Step 3: Write failing data tests**

Add tests named `test_loads_column_vectors_from_mat`, `test_duplicate_times_keep_last_finite_sample`, `test_non_finite_time_is_rejected`, `test_sparse_stride_includes_endpoints`, and `test_stride_larger_than_signal_has_unique_endpoints`. Assert exact 1-D shapes, strictly increasing time, expected retained values, and boolean observation masks.

- [ ] **Step 4: Run the data tests and verify failure**

Run: `python -m pytest Pitch_PINN/tests/test_preprocess.py -v`

Expected: FAIL because `data.scenario`, `data.mat_loader`, and `data.preprocess` do not exist.

- [ ] **Step 5: Implement typed scenario containers**

Implement immutable dataclasses `ScenarioData(time_s: np.ndarray, ax_mps2: np.ndarray, pitch_rate_radps: np.ndarray, name: str, vehicle: str)` and `SparseScenario(full: ScenarioData, observation_indices: np.ndarray)` with length/shape/finite validation and `observation_mask` plus observation-array properties.

- [ ] **Step 6: Implement MAT loading and preprocessing**

Implement `load_mat_scenario(path: Path, keys: Mapping[str, str], *, name: str, vehicle: str) -> ScenarioData`, `preprocess_scenario(data: ScenarioData, *, remove_bias_samples: int = 0) -> ScenarioData`, and `select_sparse_observations(data: ScenarioData, stride: int) -> SparseScenario`. Flatten MATLAB row/column vectors, sort timestamps stably, keep the last finite signal sample at duplicate times, reject non-finite time, and include the first/last sparse indices exactly once.

- [ ] **Step 7: Run data tests**

Run: `python -m pytest Pitch_PINN/tests/test_preprocess.py -v`

Expected: all Task 1 tests PASS.

- [ ] **Step 8: Commit if a Git repository is available**

Run only if `git rev-parse --show-toplevel` succeeds: `git add Pitch_PINN/config Pitch_PINN/data Pitch_PINN/tests/test_preprocess.py Pitch_PINN/requirements.txt && git commit -m "feat: add pitch scenario data pipeline"`.

### Task 2: Pitch physics and reference integrator

**Files:**
- Create: `Pitch_PINN/physics/__init__.py`
- Create: `Pitch_PINN/physics/half_car.py`
- Create: `Pitch_PINN/physics/suspension.py`
- Create: `Pitch_PINN/physics/skyhook.py`
- Create: `Pitch_PINN/physics/reference_model.py`
- Create: `Pitch_PINN/tests/test_half_car.py`
- Create: `Pitch_PINN/tests/test_reference_model.py`

**Interfaces:**
- Consumes: time, `Ax`, pitch angle/rate/acceleration, equivalent coefficient values, and optional complete Tesla `VehiclePitchSpec`.
- Produces: `EffectivePitchParameters`, `pitch_ode_residual(...)`, `simulate_reference(...)`, `SkyhookConfig`, `skyhook_moment(...)`, `VehiclePitchSpec`, and `effective_to_suspension_scales(...)`.

- [ ] **Step 1: Write failing physics tests**

Add tests `test_equilibrium_has_zero_residual`, `test_positive_coefficients_reject_invalid_values`, `test_zero_skyhook_gain_returns_zero`, `test_reference_zero_input_remains_at_rest`, `test_positive_ax_uses_configured_moment_sign`, `test_effective_coefficients_convert_to_expected_scales`, and `test_missing_vehicle_values_returns_no_scales`. Use a small analytic parameter set whose scale factors equal exactly 1.0.

- [ ] **Step 2: Run physics tests and verify failure**

Run: `python -m pytest Pitch_PINN/tests/test_half_car.py Pitch_PINN/tests/test_reference_model.py -v`

Expected: FAIL because the physics modules do not exist.

- [ ] **Step 3: Implement the equivalent pitch residual**

Create `EffectivePitchParameters(damping_rate_s_inv, stiffness_rate_s2, ax_gain_rad_per_m)` and implement `pitch_ode_residual(theta, theta_dot, theta_ddot, ax, parameters, skyhook_moment_value=None)` for `theta_ddot + c_eff*theta_dot + k_eff*theta - g_ax*Ax - M_sky/I_y = 0`. The skyhook term is omitted when inertia is unavailable or skyhook is disabled.

- [ ] **Step 4: Implement physical conversion and skyhook boundaries**

Create `VehiclePitchSpec` with optional `pitch_inertia_kgm2`, front/rear CG distances, baseline front/rear K and C. Implement `effective_to_suspension_scales(parameters, spec) -> SuspensionScales | None` using `K_theta = I_y*k_eff`, `C_theta = I_y*c_eff`, `K_theta0 = Kf*a^2 + Kr*b^2`, and the analogous damping equation. Implement a bounded continuous `skyhook_moment(pitch_rate, gain, limit_nm)` whose gain-zero result is exactly zero.

- [ ] **Step 5: Implement the numerical reference model**

Implement `simulate_reference(time_s: np.ndarray, ax_mps2: np.ndarray, parameters: EffectivePitchParameters, initial_theta_rad: float = 0.0, initial_pitch_rate_radps: float = 0.0) -> ReferenceTrajectory` using `scipy.integrate.solve_ivp` and linear interpolation of `Ax`; preserve input timestamps and return angle/rate arrays.

- [ ] **Step 6: Run physics tests**

Run: `python -m pytest Pitch_PINN/tests/test_half_car.py Pitch_PINN/tests/test_reference_model.py -v`

Expected: all Task 2 tests PASS.

- [ ] **Step 7: Commit if a Git repository is available**

Run only if Git is available: `git add Pitch_PINN/physics Pitch_PINN/tests/test_half_car.py Pitch_PINN/tests/test_reference_model.py && git commit -m "feat: add pitch physics reference model"`.

### Task 3: PINN network, bounded parameters, and loss

**Files:**
- Create: `Pitch_PINN/models/__init__.py`
- Create: `Pitch_PINN/models/state_network.py`
- Create: `Pitch_PINN/models/trainable_parameters.py`
- Create: `Pitch_PINN/models/pitch_pinn.py`
- Create: `Pitch_PINN/training/__init__.py`
- Create: `Pitch_PINN/training/losses.py`
- Create: `Pitch_PINN/tests/test_pitch_pinn.py`

**Interfaces:**
- Consumes: normalized time, interpolated Ax, sparse pitch-rate observations, and YAML coefficient bounds.
- Produces: `PitchStateNetwork`, `BoundedPitchParameters`, `PitchPINN.predict_with_derivatives`, `PitchLossWeights`, and `compute_pitch_losses`.

- [ ] **Step 1: Write failing model/loss tests**

Add tests `test_bounded_parameters_stay_inside_limits`, `test_network_outputs_finite_values_for_constant_time_range`, `test_autograd_returns_angle_rate_and_acceleration_shapes`, `test_zero_exact_solution_has_near_zero_physics_loss`, and `test_data_loss_uses_only_sparse_observations`.

- [ ] **Step 2: Run model tests and verify failure**

Run: `python -m pytest Pitch_PINN/tests/test_pitch_pinn.py -v`

Expected: FAIL because the model and loss modules do not exist.

- [ ] **Step 3: Implement the state network**

Implement `PitchStateNetwork(t_min_s: float, t_max_s: float, hidden_width: int = 48, hidden_layers: int = 3)` as a float64 tanh MLP that outputs relative pitch angle. Clamp the normalization denominator to machine epsilon so constant ranges remain finite.

- [ ] **Step 4: Implement bounded trainable coefficients**

Implement `BoundedPitchParameters(bounds: PitchParameterBounds, initial: EffectivePitchParameters, learn_ax_gain: bool = True)` using sigmoid-transformed raw parameters. Expose `values() -> EffectivePitchParameters` and keep all returned values inside YAML bounds.

- [ ] **Step 5: Assemble the PINN and derivatives**

Implement `PitchPINN.forward(t)` and `predict_with_derivatives(t) -> PitchPrediction`, using autograd to return `theta`, `theta_dot`, and `theta_ddot`. Implement linear differentiable `AxInterpolator` over the loaded signal and use it for the physics residual.

- [ ] **Step 6: Implement loss components**

Implement `compute_pitch_losses(model, observation_batch, collocation_time, ax_interpolator, weights) -> PitchLossResult` with sparse Pitch Rate data loss, ODE residual loss, initial angle/rate loss, and coefficient regularization. Return each component and weighted total separately.

- [ ] **Step 7: Run model/loss tests**

Run: `python -m pytest Pitch_PINN/tests/test_pitch_pinn.py -v`

Expected: all Task 3 tests PASS.

- [ ] **Step 8: Commit if a Git repository is available**

Run only if Git is available: `git add Pitch_PINN/models Pitch_PINN/training/__init__.py Pitch_PINN/training/losses.py Pitch_PINN/tests/test_pitch_pinn.py && git commit -m "feat: add sparse-data pitch PINN"`.

### Task 4: Trainer, evaluation artifacts, and training CLI

**Files:**
- Create: `Pitch_PINN/training/trainer.py`
- Create: `Pitch_PINN/training/evaluate.py`
- Create: `Pitch_PINN/train_inverse_pinn.py`
- Create: `Pitch_PINN/tests/test_training.py`

**Interfaces:**
- Consumes: YAML config, `SparseScenario`, `PitchPINN`, and loss functions from Tasks 1-3.
- Produces: `TrainingConfig`, `train_pitch_pinn`, `evaluate_reference`, `save_evaluation_artifacts`, and CLI command `python Pitch_PINN/train_inverse_pinn.py --config ...`.

- [ ] **Step 1: Write failing training/evaluation tests**

Add `test_short_training_reduces_total_loss`, `test_hidden_point_metrics_exclude_observations`, `test_physics_integrated_reference_is_reported_separately`, `test_artifacts_include_checkpoint_metrics_csv_and_plot`, and `test_seed_reproduces_observation_indices_and_initial_loss`. Use a deterministic short synthetic trajectory for speed, while leaving the CLI default pointed at the actual EV6 MAT file.

- [ ] **Step 2: Run training tests and verify failure**

Run: `python -m pytest Pitch_PINN/tests/test_training.py -v`

Expected: FAIL because trainer/evaluator/CLI functions do not exist.

- [ ] **Step 3: Implement deterministic training**

Implement `train_pitch_pinn(model, sparse_scenario, config, output_dir) -> TrainingResult` with Adam, optional LBFGS finishing, deterministic seeds, a fixed collocation grid, best-checkpoint retention, CSV history, and progress logging. CPU must be the default device.

- [ ] **Step 4: Implement hidden-point evaluation**

Implement `evaluate_reference(time_s, truth_pitch_rate, predicted_pitch_rate, observation_indices) -> ReferenceMetrics` with full RMSE, observed RMSE, hidden-point RMSE, peak magnitude error, and peak time error. Run `simulate_reference` with the learned coefficients and evaluate its result separately from the direct PINN output. Save both predictions in `metrics.json`, `reference.csv`, `loss_history.csv`, `model.pt`, and a plot distinguishing sparse observations from hidden ground truth.

- [ ] **Step 5: Implement YAML-driven CLI**

Implement config loading and `main()` in `train_inverse_pinn.py`. Support `--config`, `--epochs`, and `--output-dir`; print the learned equivalent coefficients and hidden-point RMSE, and save the resolved configuration with the run artifacts.

- [ ] **Step 6: Run training tests**

Run: `python -m pytest Pitch_PINN/tests/test_training.py -v`

Expected: all Task 4 tests PASS.

- [ ] **Step 7: Run a short actual-data smoke training**

Run: `python Pitch_PINN/train_inverse_pinn.py --config Pitch_PINN/config/tesla_ev6.yaml --epochs 20 --output-dir Pitch_PINN/outputs/smoke`

Expected: exit code 0 and finite learned coefficients, hidden-point RMSE, checkpoint, JSON/CSV files, and plot.

- [ ] **Step 8: Commit if a Git repository is available**

Run only if Git is available: `git add Pitch_PINN/training Pitch_PINN/train_inverse_pinn.py Pitch_PINN/tests/test_training.py && git commit -m "feat: train and evaluate sparse pitch reference"`.

### Task 5: Safe CarMaker mapping and runner

**Files:**
- Create: `Pitch_PINN/carmaker/__init__.py`
- Create: `Pitch_PINN/carmaker/parameter_mapper.py`
- Create: `Pitch_PINN/carmaker/runner.py`
- Create: `Pitch_PINN/carmaker/result_loader.py`
- Create: `Pitch_PINN/tests/test_carmaker.py`

**Interfaces:**
- Consumes: learned `EffectivePitchParameters`, optional complete `VehiclePitchSpec`, explicit CarMaker command and working target paths, and result MAT key mapping.
- Produces: `build_parameter_update`, `write_parameter_diff`, `run_carmaker`, and `load_carmaker_result`.

- [ ] **Step 1: Write failing CarMaker safety tests**

Add `test_missing_vehicle_spec_produces_coefficients_only`, `test_complete_spec_produces_alpha_k_and_alpha_c`, `test_dry_run_does_not_modify_target`, `test_live_run_rejects_incomplete_configuration_before_launch`, `test_target_outside_pinn_file_is_rejected`, and `test_result_loader_returns_scenario_data`.

- [ ] **Step 2: Run CarMaker tests and verify failure**

Run: `python -m pytest Pitch_PINN/tests/test_carmaker.py -v`

Expected: FAIL because CarMaker modules do not exist.

- [ ] **Step 3: Implement parameter report/mapping**

Implement `build_parameter_update(parameters, vehicle_spec) -> ParameterUpdate` so incomplete physical data returns only `c_eff`, `k_eff`, and `g_ax`; complete data additionally returns rotational K/C and `alpha_K`/`alpha_C`. Implement `write_parameter_diff(update, path)` as JSON output without editing a CarMaker file.

- [ ] **Step 4: Implement guarded runner and result loader**

Implement `run_carmaker(config: CarMakerRunConfig) -> CompletedProcess | DryRunResult`. Resolve absolute paths, require the target to remain under `E:/PINN/PINN_file`, reject install-directory targets, and validate all live-run fields before `subprocess.run`. Reuse `load_mat_scenario` in `load_carmaker_result`.

- [ ] **Step 5: Run CarMaker tests**

Run: `python -m pytest Pitch_PINN/tests/test_carmaker.py -v`

Expected: all Task 5 tests PASS and no CarMaker project file changed.

- [ ] **Step 6: Commit if a Git repository is available**

Run only if Git is available: `git add Pitch_PINN/carmaker Pitch_PINN/tests/test_carmaker.py && git commit -m "feat: add guarded CarMaker iteration boundary"`.

### Task 6: End-to-end iteration entry point and user guide

**Files:**
- Create: `Pitch_PINN/run_iteration.py`
- Create: `Pitch_PINN/README.md`
- Modify: `Pitch_PINN/config/tesla_ev6.yaml`

**Interfaces:**
- Consumes: outputs and public functions from Tasks 1-5.
- Produces: one command that trains, evaluates, writes a parameter diff, and optionally invokes CarMaker; documentation for actual data and Tesla-spec activation.

- [ ] **Step 1: Implement the iteration orchestrator**

Implement CLI `python Pitch_PINN/run_iteration.py --config <path> [--epochs N] [--allow-carmaker]`. Always run training/evaluation and write `parameter_update.json`; invoke CarMaker only when both `--allow-carmaker` and YAML `dry_run: false` are present. Store each run in a timestamped output directory.

- [ ] **Step 2: Write the user guide**

Document environment creation, dependency installation, short smoke run, full training run, output interpretation, sparse-vs-hidden validation plot, required Tesla physical fields for K/C scale conversion, and the explicit live CarMaker opt-in. State that the supplied workbook does not currently contain Tesla numeric specifications.

- [ ] **Step 3: Run the full automated test suite**

Run: `python -m pytest Pitch_PINN/tests -v`

Expected: all tests PASS.

- [ ] **Step 4: Run end-to-end dry-run smoke verification**

Run: `python Pitch_PINN/run_iteration.py --config Pitch_PINN/config/tesla_ev6.yaml --epochs 20`

Expected: exit code 0; output directory contains resolved config, reference plot/CSV, loss history, checkpoint, metrics, and `parameter_update.json`; no CarMaker input file is changed and no external process is launched.

- [ ] **Step 5: Verify source tree and unresolved markers**

Run: `rg --files Pitch_PINN` and `rg -n "TODO|TBD|FIXME|NotImplementedError" Pitch_PINN`

Expected: every planned file exists and the second command returns no unresolved core implementation markers.

- [ ] **Step 6: Commit if a Git repository is available**

Run only if Git is available: `git add Pitch_PINN/run_iteration.py Pitch_PINN/README.md Pitch_PINN/config/tesla_ev6.yaml && git commit -m "docs: add pitch PINN iteration workflow"`.
