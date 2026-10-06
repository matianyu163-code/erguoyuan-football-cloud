"""Production trial test helpers; all match inputs are labelled SYNTHETIC_TEST."""

from pathlib import Path

import pytest

from production.config import ProductionConfig


@pytest.fixture
def production_config(tmp_path: Path) -> ProductionConfig:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "production.yaml"
    config_file.write_text("""mode: TRIAL
features:
  enable_live_data: true
  enable_prediction: true
  enable_record: true
  enable_update: true
safety:
  auto_bet: false
  auto_publish: false
storage:
  trial_record_database: .workspace/trial_prediction_records.sqlite
  update_history: .workspace/update_history.json
""", encoding="utf-8")
    return ProductionConfig.load(config_file)
