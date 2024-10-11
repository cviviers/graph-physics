import lightning as L
import torch
from loguru import logger
from torch_geometric.data import Batch

from graphphysics.training.parse_parameters import get_model, get_simulator
from graphphysics.utils.loss import L2Loss
from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.scheduler import CosineWarmupScheduler


def build_mask(param: dict, graph: Batch):
    if len(graph.x.shape) > 2:
        node_type = graph.x[:, 0, param["index"]["node_type_index"]]
    else:
        node_type = graph.x[:, param["index"]["node_type_index"]]
    mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)
    mask = torch.logical_not(mask)

    return mask


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
        trajectory_length: int = 599,
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

        self.param = parameters

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

        self.val_step_outputs = []
        self.val_step_targets = []
        self.trajectory_length = trajectory_length
        self.current_val_trajectory = 0
        self.last_val_prediction = None

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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        # Determine if we need to reset the trajectory
        if batch_idx // self.trajectory_length > self.current_val_trajectory:
            self.current_val_trajectory += 1
            self.last_val_prediction = None

        # Prepare the batch for the current step
        batch = batch.clone()  # Avoid in-place modification
        if self.last_val_prediction is not None:
            # Update the batch with the last prediction
            batch.x[:, self.model.output_index_start : self.model.output_index_end] = (
                self.last_val_prediction.detach()
            )

        mask = build_mask(self.param, batch)
        node_type = batch.x[:, self.model.node_type_index]
        target = batch.y

        with torch.no_grad():
            _, _, predicted_outputs = self.model(batch)

        # Apply mask to predicted outputs
        predicted_outputs[mask] = target[mask]
        self.val_step_outputs.append(predicted_outputs.cpu())
        self.val_step_targets.append(target.cpu())

        self.last_val_prediction = predicted_outputs

        val_loss = self.loss(
            target,
            predicted_outputs,
            node_type,
            masks=self.loss_masks,
        )

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Concatenate outputs and targets
        predicteds = torch.cat(self.val_step_outputs, dim=0)
        targets = torch.cat(self.val_step_targets, dim=0)

        # Compute RMSE over all rollouts
        squared_diff = (predicteds - targets) ** 2
        all_rollout_rmse = torch.sqrt(squared_diff.mean()).item()

        self.log(
            "val_all_rollout_rmse",
            all_rollout_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Clear stored outputs
        self.val_step_outputs.clear()
        self.val_step_targets.clear()
        self.current_val_trajectory = 0
        self.last_val_prediction = None

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
