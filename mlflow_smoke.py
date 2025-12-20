import os
import tempfile
from pathlib import Path

import mlflow


def main() -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_name="smoke-test") as run:
        mlflow.log_param("purpose", "docker-compose-smoke")
        mlflow.log_metric("ok", 1.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "artifact.txt"
            artifact_path.write_text("mlflow smoke test\n", encoding="utf-8")
            mlflow.log_artifact(str(artifact_path))

        print(f"tracking_uri={tracking_uri}")
        print(f"run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
