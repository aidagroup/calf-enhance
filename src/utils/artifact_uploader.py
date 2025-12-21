import shutil
import threading
import time
from pathlib import Path

from loguru import logger
from mlflow.tracking import MlflowClient

from src.config import config


class ArtifactUploader:
    """Background worker that batches and uploads artifacts to MLflow."""

    def __init__(self, base_staging_dir: Path, poll_interval: float = 30.0):
        self.base_staging_dir = Path(base_staging_dir)
        self.poll_interval = poll_interval

        self._client = MlflowClient()
        self._run_id: str | None = None
        self._staging_dir: Path | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def staging_dir(self) -> Path:
        if self._staging_dir is None:
            raise RuntimeError("ArtifactUploader not started")
        return self._staging_dir

    def start(self, run_id: str):
        self._run_id = run_id
        self._staging_dir = self.base_staging_dir / run_id
        self._staging_dir.mkdir(parents=True, exist_ok=True)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=False)
        self._thread.start()
        logger.info(f"ArtifactUploader started for run {run_id}")

    def stop(self):
        """Block until all staged artifacts are uploaded."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        if self._staging_dir and self._staging_dir.exists():
            shutil.rmtree(self._staging_dir)
        logger.info("ArtifactUploader stopped, all artifacts uploaded")

    def _worker_loop(self):
        while not self._stop_event.is_set():
            self._upload_all()
            self._stop_event.wait(self.poll_interval)
        self._flush_remaining()

    def _flush_remaining(self):
        """Keep retrying until all files are uploaded."""
        max_retries = 5
        for attempt in range(max_retries):
            if not self._has_files():
                return
            file_count = self._count_files()
            logger.info(
                f"Final flush: {file_count} artifact(s), attempt {attempt + 1}/{max_retries}"
            )
            try:
                self._client.log_artifacts(self._run_id, str(self._staging_dir))
                self._clear_staging()
                return
            except Exception as e:
                logger.warning(f"Flush attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
        logger.error(
            f"Failed to upload remaining artifacts after {max_retries} attempts"
        )

    def _upload_all(self):
        if not self._has_files():
            return

        file_count = self._count_files()
        logger.debug(f"Uploading {file_count} artifact(s)")

        try:
            self._client.log_artifacts(self._run_id, str(self._staging_dir))
            self._clear_staging()
        except Exception as e:
            logger.error(f"Failed to upload artifacts: {e}")

    def _has_files(self) -> bool:
        return any(self._staging_dir.rglob("*"))

    def _count_files(self) -> int:
        return sum(1 for f in self._staging_dir.rglob("*") if f.is_file())

    def _clear_staging(self):
        for item in self._staging_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


_uploader: ArtifactUploader | None = None


def init_artifact_uploader(
    run_id: str,
    base_staging_dir: Path = config.LOG_ARTIFACT_DIR,
    poll_interval: float = 30.0,
):
    global _uploader
    _uploader = ArtifactUploader(base_staging_dir, poll_interval)
    _uploader.start(run_id)


def shutdown_artifact_uploader():
    global _uploader
    if _uploader:
        _uploader.stop()
        _uploader = None


def get_artifact_uploader() -> ArtifactUploader | None:
    return _uploader
