import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts import analyze_seer_drift as analysis


analysis.model_table_path = (
    project_root
    / "data"
    / "processed"
    / "seer_v2_model_table.parquet"
)

analysis.feature_config_path = (
    project_root
    / "configs"
    / "seer_features_v2.json"
)

analysis.feature_drift_path = (
    project_root
    / "artifacts"
    / "metrics"
    / "seer_v2_development_feature_drift.csv"
)

analysis.prevalence_path = (
    project_root
    / "artifacts"
    / "metrics"
    / "seer_v2_development_prevalence_drift.csv"
)

analysis.summary_path = (
    project_root
    / "artifacts"
    / "metrics"
    / "seer_v2_development_drift_summary.csv"
)

analysis.audit_path = (
    project_root
    / "artifacts"
    / "metrics"
    / "seer_v2_development_drift_audit.json"
)


if __name__ == "__main__":
    analysis.main()
