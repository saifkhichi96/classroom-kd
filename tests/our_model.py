import pytorch_lightning as pl

pl.seed_everything(42)  # seed to make randomness deterministic

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# from datasets import load_dataset, config
import torch
import torch.nn as nn
from torchvision.models import (
    RegNet_Y_32GF_Weights,
    regnet_y_32gf,
    vit_b_16,
    ViT_B_16_Weights,
    mobilenet_v2,
    MobileNet_V2_Weights,
    RegNet_Y_400MF_Weights,
    regnet_y_400mf,
    SqueezeNet1_1_Weights,
    squeezenet1_1,
    MobileNet_V2_Weights,
)
import torch.optim as optim
from torchmetrics import Accuracy
from pathlib import Path
import os
import json
from torchvision.datasets import ImageFolder
import argparse
import json
import os
from pytorch_lightning import Trainer

# Set the path to the root directory of the ILSVRC2012 dataset
data_dir = "/netscratch/sarode/Thesis/data/in1k_msn5perclass_split3"

# Define data transformations for training and validation
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create ImageFolder datasets for training and validation
train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
val_dataset = ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

# train_size = int(0.7 * len(dataset))
# val_size = len(dataset) - train_size

# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
# Create data loaders for training and validation
batch_size = 8
val_batch_size = 1
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
)
val_loader = DataLoader(
    val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8
)


class MNISTClassifier(nn.Module):
    def __init__(self, num_classes=10, name="", teacher=False, train=False):
        super(MNISTClassifier, self).__init__()

        self.name = name
        if self.name == "vit_teacher":
            self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            # Modify the output layer for 10 classes
            if num_classes != 1000:
                self.model.heads.head = nn.Linear(
                    self.model.heads.head.in_features, num_classes
                )
        elif self.name == "regnet_teacher":
            self.model = regnet_y_32gf(weights=RegNet_Y_32GF_Weights.IMAGENET1K_V1)
            # Modify the output layer for 10 classes
            if num_classes != 1000:
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif self.name == "mobilenet_student":
            # with weights
            self.model = mobilenet_v2(weights=MobileNet_V2_Weights)
            # Modify the output layer for 10 classes
            if num_classes != 1000:
                self.model.classifier[1] = nn.Linear(
                    self.model.classifier[1].in_features, num_classes
                )
        elif self.name == "regnet_peer":
            self.model = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V2)
            if num_classes != 1000:
                # Modify the output layer for 10 classes
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif self.name == "squeezenet_peer":
            self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
            # Modify the output layer for 10 classes
            if num_classes != 1000:
                self.model.classifier[1] = nn.Conv2d(
                    self.model.classifier[1].in_channels, num_classes, kernel_size=1
                )

    def forward(self, x):
        return self.model(x)


class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()

    def forward(self, logit1, logit2, T):
        soft_t1 = nn.functional.softmax(logit2 / T, dim=-1)
        soft_s1 = nn.functional.log_softmax(logit1 / T, dim=-1)
        soft_targets_loss = -torch.sum(soft_t1 * soft_s1) / soft_s1.size()[0] * (T**2)
        return soft_targets_loss


class ClassificationModule(pl.LightningModule):
    """A PyTorch Lightning module for contains both the network and the
    training logic, unlike simple PyTorch code we saw in the first tutorial."""

    def __init__(
        self,
        teacher_models=[],
        student_models=[],
        peer_models=[],
        learning_rate=0.001,
        num_classes=10,
        teacher_weight=0.6,
        peer_weight=0.1,
        self_conf=0.3,
        confidence_threshold=0.6,
    ):
        super(ClassificationModule, self).__init__()
        self.save_hyperparameters()

        self.self_conf = self_conf
        self.teacher_weight = teacher_weight
        self.peer_weight = peer_weight
        self.confidence_threshold = confidence_threshold

        # Use nn.ModuleList for teacher models
        self.teacher_models = nn.ModuleList(
            [
                MNISTClassifier(num_classes=num_classes, name=teacher)
                for teacher in teacher_models
                if len(teacher_models)
            ]
        )
        self.student_models = MNISTClassifier(
            num_classes=num_classes, name=student_models[0]
        )
        self.peer_models = nn.ModuleList(
            [
                MNISTClassifier(num_classes=num_classes, name=peer)
                for peer in peer_models
                if len(peer_models)
            ]
        )

        if len(teacher_models):
            for teacher in self.teacher_models:
                teacher.to(self.device).eval()
                for param in teacher.parameters():
                    param.requires_grad = False
        if len(peer_models):
            for peer in self.peer_models:
                peer.to(self.device).eval()
                for param in peer.parameters():
                    param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = DistillationLoss()
        self.metric = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return (
            [
                teacher(x)
                for teacher in self.teacher_models
                if len(self.teacher_models) > 0
            ],
            [peer(x) for peer in self.peer_models if len(self.peer_models) > 0],
            [self.student_models(x)],
        )

    def training_step(self, batch, batch_idx):
        images, labels = batch
        T = 12
        teacher_outputs, peer_outputs, student_outputs = self(images)
        student_output = student_outputs[0]
        base_loss = self.loss_fn(student_output, labels)

        # Compute the confidence score (softmax) for the student's predictions
        student_probs = nn.functional.softmax(student_output, dim=-1)
        student_confidence, _ = torch.max(student_probs, dim=-1)
        # print(student_confidence.item())

        # Compute the decay factor for the threshold
        decay_factor = max(0, 1 - (self.current_epoch / self.trainer.max_epochs))

        # Check if any element of student confidence is below the threshold
        if torch.any(student_confidence < self.hparams.confidence_threshold):
            # If any below threshold, use the teacher weight with decay
            teacher_weight = self.hparams.teacher_weight * decay_factor
        else:
            # If all above threshold, set teacher weight to 0
            teacher_weight = 0

        teacher_loss = 0
        if len(teacher_outputs):
            for teacher_output in teacher_outputs:
                teacher_loss += self.distill_loss_fn(student_output, teacher_output, T)

        peer_loss = 0
        if len(peer_outputs):
            for peer_output in peer_outputs:
                peer_loss += self.distill_loss_fn(student_output, peer_output, T)

        loss = (
            self.self_conf * base_loss
            + teacher_weight * teacher_loss
            + self.peer_weight * peer_loss
        )
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)

        return loss

    def on_validation_epoch_start(self):
        self.metric.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        T = 12
        teacher_outputs, peer_outputs, student_outputs = self(images)
        student_output = student_outputs[0]
        base_loss = self.loss_fn(student_output, labels)
        loss = base_loss
        self.log(
            "val_loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True
        )

        # Update accuracy for current batch
        _, preds = torch.max(student_output, 1)
        self.metric.update(preds, labels)
        return loss

    def on_validation_epoch_end(self):
        avg_accuracy = self.metric.compute()
        self.log("val_accuracy", avg_accuracy, prog_bar=True, sync_dist=True)

    def on_test_epoch_start(self):
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        # to do
        images, labels = batch
        outputs = self(images)

        # Note: We do not need to calculate loss when evaluating
        # on the test dataset, only the performance metric!

        # Update accuracy for current batch
        _, preds = torch.max(outputs, 1)
        self.metric.update(preds, labels)
        return {"test_accuracy": self.metric}

    def on_test_epoch_end(self):
        # to do
        avg_accuracy = self.metric.compute()
        self.log("test_accuracy", avg_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.student_models.parameters(), lr=self.hparams.learning_rate
        )
        return optimizer


# train
# Initialize the classifier
import json


def main(config_file_path):
    # Load the configuration file
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)
    num_classes = config["num_classes"]
    # Extract teacher and student models from the config
    teacher_models = config["teachers"]
    student_models = config["students"]
    peer_models = config["peers"]
    # Initialize the classifier with the specified models
    classifier = ClassificationModule(
        teacher_models=teacher_models,
        student_models=student_models,
        peer_models=peer_models,
        num_classes=num_classes,
    )
    # Create a logger # WandbLogger
    logger = pl.loggers.CSVLogger("./logs")

    # Initialize a trainer with SLURMCluster
    trainer = Trainer(
        deterministic=True,
        logger=logger,
        max_epochs=50,
        devices=-1,
        accelerator="gpu",
        strategy="ddp",
    )
    # Train the model
    trainer.fit(classifier, train_loader, val_loader)


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Imagenet Knowledge Distillation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function
    main(args.config)
