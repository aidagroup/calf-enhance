from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

REPO_PATH = Path(__file__).parent.parent


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_PATH / ".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str
    MLFLOW_S3_ENDPOINT_URL: str = "http://localhost:9000"


config = Config()
