import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts import train_joint_model as training


training.joint_config_path = (
    project_root
    / "configs"
    / "joint_model_v2.json"
)

training.feature_config_path = (
    project_root
    / "configs"
    / "seer_features_v2.json"
)

training.preprocessor_state_path = (
    project_root
    / "artifacts"
    / "features"
    / "seer_v2_preprocessor.json"
)

training.seer_model_table_path = (
    project_root
    / "data"
    / "processed"
    / "seer_v2_model_table.parquet"
)

training.checkpoint_directory = (
    project_root
    / "artifacts"
    / "checkpoints"
    / "joint_v2"
)

training.metrics_directory = (
    project_root
    / "artifacts"
    / "metrics"
    / "joint_v2"
)


if __name__ == "__main__":
    training.main()
