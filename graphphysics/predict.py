import json
import warnings

import torch
from absl import app, flags
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch_geometric.loader import DataLoader

import wandb
from graphphysics.external.aneurysm import build_features
from graphphysics.training.lightning_module import LightningModule
from graphphysics.training.parse_parameters import get_dataset, get_preprocessing

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "prediction_project", "Name of the WandB project")
flags.DEFINE_string("model_path", None, "Path to the checkpoint (.ckpt) file")
flags.DEFINE_bool("no_edge_feature", False, "Whether to use edge features")
flags.DEFINE_string(
    "predict_parameters_path", None, "Path to the training parameters JSON file"
)


def main(argv):
    del argv

    # Check that the parameters path is provided
    if not FLAGS.predict_parameters_path:
        raise ValueError("The 'predict_parameters_path' flag must be provided.")

    # Load training parameters from JSON file
    parameters_path = FLAGS.predict_parameters_path
    logger.info(f"Opening prediction parameters from {parameters_path}")
    try:
        with open(parameters_path, "r") as fp:
            parameters = json.load(fp)
    except Exception as e:
        logger.error(f"Error reading training parameters: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_project_name = FLAGS.project_name
    model_path = FLAGS.model_path
    use_edge_feature = not FLAGS.no_edge_feature

    # Build preprocessing function
    preprocessing = get_preprocessing(
        param=parameters,
        device=device,
        use_edge_feature=use_edge_feature,
        remove_noise=True,
        extra_node_features=None,
    )

    # Get predict datasets
    predict_dataset = get_dataset(
        param=parameters,
        preprocessing=preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=True,
    )

    print(f"Predict dataset: {predict_dataset}")
    # show the first element in the dataset
    if len(predict_dataset) > 0:
        first_data = predict_dataset[0]
        print("First data in predict dataset:")
        print(f"Node features: {first_data.x}")
        print(f"Edge features: {first_data.edge_attr if hasattr(first_data, 'edge_attr') else 'No edge features'}")
        print(f"Target: {first_data.y if hasattr(first_data, 'y') else 'No target'}")

    predict_dataloader_kwargs = {
        "dataset": predict_dataset,
        "shuffle": False,
        "batch_size": 1,
        "num_workers": 0,
    }

    # Create DataLoader
    predict_dataloader = DataLoader(**predict_dataloader_kwargs)

    # Load trained model

    logger.info(f"Loading model from checkpoint: {model_path}")
    lightning_module = LightningModule.load_from_checkpoint(
        checkpoint_path=model_path,
        parameters=parameters,
        trajectory_length=predict_dataset.trajectory_length,
        timestep=predict_dataset.dt,
    )

    # Initialize WandbLogger
    wandb_run = wandb.init(project=wandb_project_name)
    wandb_logger = WandbLogger(experiment=wandb_run)

    wandb_logger.experiment.config.update(
        {
            "architecture": parameters["model"]["type"],
            "#_layers": parameters["model"]["message_passing_num"],
            "#_neurons": parameters["model"]["hidden_size"],
            "#_hops": parameters["dataset"]["khop"],
        }
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        devices=1,
        inference_mode=True,
    )

    # Start prediction
    logger.success("Starting prediction")
    trainer.predict(model=lightning_module, dataloaders=predict_dataloader)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)
