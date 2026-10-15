from __future__ import annotations

import json
from pathlib import Path

from Pitch_PINN.run_iteration import run_iteration_from_config
from Pitch_PINN.train_inverse_pinn import load_config


def test_iteration_creates_parameter_update_without_live_carmaker(
    tmp_path: Path,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "tesla_ev6.yaml"
    config = load_config(config_path)

    result = run_iteration_from_config(
        config,
        epochs=0,
        output_root=tmp_path,
        allow_carmaker=False,
    )

    update_path = result.output_dir / "parameter_update.json"
    assert update_path.is_file()
    update = json.loads(update_path.read_text(encoding="utf-8"))
    assert set(update) == {"k_front_npm", "k_rear_npm", "c_front_nspm", "c_rear_nspm"}
    assert result.carmaker_was_live is False
    assert (result.output_dir / "metrics.json").is_file()
