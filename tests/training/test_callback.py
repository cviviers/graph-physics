import unittest
import pytorch_lightning as pl
from unittest.mock import MagicMock
from graphphysics.training.callback import LogPyVistaPredictionsCallback
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from tests.mock import (
    MOCK_XDMF_FOLDER,
    MOCK_H5_META10_SAVE_PATH,
)
from graphphysics.dataset.preprocessing import build_preprocessing


class MockModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, graph):
        return None, None, graph.x


transform = build_preprocessing(add_edges_features=True)
dataset = XDMFDataset(
    xdmf_folder=MOCK_XDMF_FOLDER,
    meta_path=MOCK_H5_META10_SAVE_PATH,
    preprocessing=transform,
)


class TestLogPyVistaPredictionsCallback(unittest.TestCase):
    def setUp(self):
        self.dataset = dataset
        self.indices = [0, 1]
        self.output_dir = "test_predictions"
        self.model = MockModel()
        self.callback = LogPyVistaPredictionsCallback(
            dataset=self.dataset, indices=self.indices, output_dir=self.output_dir
        )

        # Mock trainer and logger
        self.trainer = MagicMock()
        self.trainer.logger = MagicMock(spec=pl.loggers.WandbLogger)
        self.trainer.logger.experiment = MagicMock()
        self.trainer.logger.log_image = MagicMock()

    def test_on_validation_epoch_end(self):
        # Call the callback's method
        self.callback.on_validation_epoch_end(self.trainer, self.model)

        # Check that images were logged
        self.assertTrue(self.trainer.logger.log_image.called)
        args, kwargs = self.trainer.logger.log_image.call_args
        key = kwargs.get("key", args[0] if args else None)
        images = kwargs.get("images", args[1] if len(args) > 1 else None)
        captions = kwargs.get("caption", args[2] if len(args) > 2 else None)

        self.assertEqual(key, "pyvista_predictions")
        self.assertEqual(len(images), len(self.indices))
        self.assertEqual(len(captions), len(self.indices))

    """def tearDown(self):
        # Clean up any generated files or directories
        import shutil
        import os

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)"""


if __name__ == "__main__":
    unittest.main()
