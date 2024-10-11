import unittest
from unittest.mock import MagicMock, patch
import torch
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.data import Data
import lightning as L

from graphphysics.training.lightning_module import LightningModule
from tests.mock import (
    MOCK_H5_META_SAVE_PATH,
    MOCK_H5_SAVE_PATH,
)

with patch("graphphysics.training.parse_parameters.get_model") as mock_get_model, patch(
    "graphphysics.training.parse_parameters.get_simulator"
) as mock_get_simulator, patch("graphphysics.utils.loss.L2Loss") as MockL2Loss, patch(
    "graphphysics.utils.scheduler.CosineWarmupScheduler"
) as MockCosineWarmupScheduler:

    class MockDataset(Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            x = torch.randn(10, 8)
            x = torch.abs(x)
            edge_index = torch.randint(0, 10, (2, 20))
            edge_attr = torch.randn(20, 4)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.y = torch.randn(10, 3)
            return data

    class TestLitLightningModule(unittest.TestCase):
        def setUp(self):
            self.parameters = {
                "transformations": {
                    "preprocessing": {
                        "noise": 0.1,
                        "noise_index_start": 0,
                        "noise_index_end": 3,
                    },
                    "world_pos_parameters": {
                        "use": True,
                        "world_pos_index_start": 3,
                        "world_pos_index_end": 6,
                    },
                },
                "index": {
                    "node_type_index": 7,
                    "feature_index_start": 0,
                    "feature_index_end": 6,
                    "output_index_start": 0,
                    "output_index_end": 3,
                },
                "model": {
                    "type": "epd",
                    "message_passing_num": 5,
                    "node_input_size": 6,
                    "edge_input_size": 4,
                    "output_size": 3,
                    "hidden_size": 128,
                },
                "dataset": {
                    "extension": "h5",
                    "h5_path": MOCK_H5_SAVE_PATH,
                    "meta_path": MOCK_H5_META_SAVE_PATH,
                    "khop": 2,
                },
            }
            self.learning_rate = 0.001
            self.num_steps = 100
            self.warmup = 10
            self.only_processor = False

            # Mock the processor and simulator models
            self.mock_processor = MagicMock()
            self.mock_simulator = MagicMock()
            self.mock_simulator.node_type_index = 0
            # Mock the model's forward method
            self.mock_simulator.return_value = (
                torch.randn(10, 3),
                torch.randn(10, 3),
                None,
            )

            mock_get_model.return_value = self.mock_processor
            mock_get_simulator.return_value = self.mock_simulator

            self.model = LightningModule(
                parameters=self.parameters,
                learning_rate=self.learning_rate,
                num_steps=self.num_steps,
                warmup=self.warmup,
                only_processor=self.only_processor,
            )

            self.dataset = MockDataset()
            self.dataloader = DataLoader(self.dataset, batch_size=2)

        def test_forward(self):
            batch = next(iter(self.dataloader))
            output = self.model.forward(batch)
            self.assertIsNotNone(output)

        def test_training_step(self):
            batch = next(iter(self.dataloader))
            loss = self.model.training_step(batch)
            self.assertIsNotNone(loss)
            self.assertTrue(isinstance(loss, torch.Tensor))

        def test_configure_optimizers(self):
            optimizers = self.model.configure_optimizers()
            self.assertIn("optimizer", optimizers)
            self.assertIn("lr_scheduler", optimizers)
            self.assertIsNotNone(optimizers["optimizer"])
            self.assertIsNotNone(optimizers["lr_scheduler"])

        def test_full_training_loop(self):
            trainer = L.Trainer(fast_dev_run=True)
            trainer.fit(self.model, train_dataloaders=self.dataloader)

    if __name__ == "__main__":
        unittest.main()
