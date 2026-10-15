# SDD ledger — plan: E:/PINN/Pitch_PINN/docs/superpowers/plans/2026-10-12-ev6-full-car-pitch-pinn.md

Setup ruling: E:/PINN is not a Git repository, so an isolated worktree, commit ranges, and the supplied Git-oriented task scripts are unavailable. Work remains confined to E:/PINN/Pitch_PINN; checkpoints use test commands and outputs. Cost if wrong: changes are not isolated or recoverable through Git.
Pre-flight: Tasks 1→2 share ScenarioData exogenous arrays; Tasks 2→3 share FullCarPrediction; Tasks 3→4 share FullCarSpec/SuspensionRates/residuals; Tasks 4→5 share TrainingResult; Tasks 5→6 share learned rates. Interfaces agree with the spec.
Task 7: Ruling: drop the final unique MAT sample by configuration — the 8.001 s logger-stop row simultaneously resets Ax, Pitch Rate, and wheel velocity to zero while damper states are held, creating a nonphysical interpolation slope and a 2.72 rad/s predicted spike — cost if wrong: a genuine final 1 ms response sample would be excluded.
Task 1: complete (no Git range, tests: python -m pytest Pitch_PINN/tests/test_preprocess.py -q → pass)
Task 2: complete (no Git range, tests: python -m pytest Pitch_PINN/tests/test_full_car_stack.py -q → pass)
Task 3: complete (no Git range, tests: python -m pytest Pitch_PINN/tests/test_full_car_stack.py -q → pass)
Task 4: complete (no Git range, tests: python -m pytest Pitch_PINN/tests/test_full_car_losses.py -q → pass)
Task 5: complete (no Git range, native validation and artifact generation → pass)
Task 6: complete (no Git range, tests: python -m pytest Pitch_PINN/tests/test_iteration.py -q → pass)
Task 7: complete (no Git range, tests: python -m pytest Pitch_PINN/tests -q → 44 passed; native 5-epoch and iteration dry-run → exit 0)
Final review: self-review (subagents prohibited for this run; no Git diff available).
Final: fixed missing checkpoint normalization state — test_model_features_exclude_pitch_truth RED→GREEN, feature_mean/feature_scale now registered buffers, suite 45/45.
Final: fixed ignored optional L-BFGS setting — test_full_car_trainer_runs_optional_lbfgs RED→GREEN, suite 45/45.
Final: Ruling: preserve this plan workspace and ledger — no Git history exists to replace the scratch record — cost if wrong: a user may later treat .superpowers as removable and lose the execution audit only, not product code or outputs.
Final verification: python Pitch_PINN/train_inverse_pinn.py --epochs 5 → exit 0; python Pitch_PINN/run_iteration.py --epochs 1 → exit 0 and carmaker_live=False; python -m pytest Pitch_PINN/tests -q → 45 passed.
