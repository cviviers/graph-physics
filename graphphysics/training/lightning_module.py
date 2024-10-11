import lightning as L
import torch
from loguru import logger
from torch_geometric.data import Batch

from graphphysics.training.parse_parameters import get_model, get_simulator
from graphphysics.utils.loss import L2Loss
from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.scheduler import CosineWarmupScheduler


class LightningModule(L.LightningModule):
    """
    PyTorch Lightning Module for training a Simulator model.

    This module encapsulates the model, loss function, optimizer, and learning rate scheduler.
    It handles the training loop and integrates with PyTorch Lightning's Trainer.
    """

    def __init__(
        self,
        parameters: dict,
        learning_rate: float,
        num_steps: int,
        warmup: int,
        only_processor: bool = False,
        masks: list[NodeType] = [NodeType.NORMAL, NodeType.OUTFLOW],
    ):
        """
        Initializes the LightningModule.

        Args:
            parameters (Dict[str, Any]): Configuration parameters for the model and simulator.
            learning_rate (float): Initial learning rate for the optimizer.
            num_steps (int): Total number of training steps.
            warmup (int): Number of warmup steps for the learning rate scheduler.
            only_processor (bool, optional): Whether to use only the processor part of the model.
                Defaults to False.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
        """
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = get_model(param=parameters, only_processor=only_processor)

        print(processor)

        self.model = get_simulator(param=parameters, model=processor, device=device)

        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.success(f"Nb. of trainable parameters: {pytorch_total_params}")

        self.loss = L2Loss()
        self.loss_masks = masks

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.warmup = warmup

    def forward(self, graph: Batch):
        return self.model(graph)

    def training_step(self, batch: Batch):
        node_type = batch.x[:, self.model.node_type_index]
        network_output, target_delta_normalized, _ = self.model(batch)

        loss = self.loss(
            target_delta_normalized,
            network_output,
            node_type,
            masks=self.loss_masks,
        )
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        """Initialize the optimizer"""
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0001,
            betas=(0.9, 0.95),
        )
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cosine_t_max)
        sch = CosineWarmupScheduler(opt, warmup=self.warmup, max_iters=self.num_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
                "interval": "step",
                "frequency": 1,
            },
        }
