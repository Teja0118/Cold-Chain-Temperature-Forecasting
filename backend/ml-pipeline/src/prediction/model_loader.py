from pathlib import Path

import joblib


class ModelLoader:
    """
    Loads trained model artifacts once and exposes
    them through convenient properties.
    """

    def __init__(self, version: str = "v4"):

        self.version = version

        self.model_path = (
            Path("models")
            / "classification"
            / version
            / "logistics_action_model.joblib"
        )

        self._loaded = False

        self._pipeline = None
        self._regression = None
        self._label_encoder = None
        self._warning_threshold = None
        self._input_columns = None
        self._model_name = None

    def load(self) -> None:
        """
        Load artifacts only once.
        """

        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path}"
            )

        artifact = joblib.load(self.model_path)

        self._pipeline = artifact["pipeline"]
        self._regression = artifact["risk_model"]
        self._label_encoder = artifact["label_encoder"]
        self._warning_threshold = artifact["warning_threshold"]
        self._input_columns = artifact["input_columns"]
        self._model_name = artifact["model_name"]

        self._loaded = True

    @property
    def classifier(self):
        self.load()
        return self._pipeline

    @property
    def regression(self):
        self.load()
        return self._regression

    @property
    def label_encoder(self):
        self.load()
        return self._label_encoder

    @property
    def warning_threshold(self):
        self.load()
        return self._warning_threshold

    @property
    def input_columns(self):
        self.load()
        return self._input_columns

    @property
    def model_name(self):
        self.load()
        return self._model_name