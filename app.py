from lightning.app import LightningApp, LightningFlow

from apps.diverse_samples import DiverseSamplesSelection
from apps.space_enumerator import SpaceEnumerator
from apps.ml_predictor import MLModelSelection


class LitApp(LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.space_enumerator = SpaceEnumerator(parallel=True)
        self.diverse_samples_selection = DiverseSamplesSelection(parallel=True)
        self.ml_model_selection = MLModelSelection(parallel=True)

    def configure_layout(self):
        return [
            dict(name="Space enumerator", content=self.space_enumerator),
            dict(name="Initial Samples Selection", content=self.diverse_samples_selection),
            dict(name="ML model selection", content=self.ml_model_selection),
        ]

    def run(self):
        self.space_enumerator.run()
        self.diverse_samples_selection.run()
        self.ml_model_selection.run()


app = LightningApp(LitApp())
