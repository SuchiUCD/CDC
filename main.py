import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Optional, Tuple
from models.resnet import ResNet18
import os
from PIL import Image
import wandb
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy

class LitResNet(pl.LightningModule):
    """
    PyTorch Lightning module for Cat vs Dog binary classification using ResNet18.
    Implements training, validation, and test steps with appropriate metrics logging.
    """
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        # Initialize ResNet18 with 1 input channel (grayscale) and 2 output classes (cat/dog)
        self.model = ResNet18(num_classes=2, num_input_channels=1)

        # Initialize binary classification metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step logic with loss calculation and metric logging.
        Uses CrossEntropyLoss for binary classification.
        """
        x, y = batch
        logits = self(x)
        # Calculate cross entropy loss for binary classification
        loss = F.cross_entropy(logits, y)

        # Calculate and log accuracy metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        # Log both loss and accuracy metrics for monitoring
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step logic with metric logging"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate and log metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step logic with metric logging"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate and log metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)

        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        Uses Adam optimizer with ReduceLROnPlateau scheduler monitoring validation loss.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"  # Metric to monitor for scheduling
            }
        }


class CatDogDataset(Dataset):
    """Custom Dataset for loading cat and dog images"""
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['cat', 'dog']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        # Walk through the directory
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class CatDogDataModule(pl.LightningDataModule):
    """
    DataModule for Cat vs Dog classification dataset.
    Implements data loading, transforms, and dataset splitting.
    """
    def __init__(
            self,
            data_dir: str = "dataset",
            batch_size: int = 32,
            num_workers: int = 4,
            image_size: Tuple[int, int] = (64, 64)
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Define data augmentation for training
        self.transform_train = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(1),  # Convert to grayscale
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.RandomRotation(10),      # Data augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226])  # Grayscale normalization
        ])

        # Define transforms for validation/testing (no augmentation)
        self.transform_val = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226])
        ])

    def setup(self, stage: Optional[str] = None):
        """Initialize the datasets"""
        if stage == 'fit' or stage is None:
            self.train_dataset = CatDogDataset(
                root_dir=os.path.join(self.data_dir, 'train'),
                transform=self.transform_train
            )
            self.val_dataset = CatDogDataset(
                root_dir=os.path.join(self.data_dir, 'val'),
                transform=self.transform_val
            )

        if stage == 'test' or stage is None:
            self.test_dataset = CatDogDataset(
                root_dir=os.path.join(self.data_dir, 'test'),
                transform=self.transform_val
            )

    def train_dataloader(self):
        """Create the train data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """Create the validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        """Create the test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def main():
    """
    Main training function implementing the required components:
    - Loss function: CrossEntropyLoss
    - Optimizer: Adam with ReduceLROnPlateau scheduler
    - Metrics: Binary accuracy and loss
    - Logging: WandB integration for metric tracking
    - Model checkpointing: Saves best models based on validation loss
    """
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project="cat-dog-classifier",
        name="resnet18-grayscale",
        log_model=True
    )

    # Initialize model and data
    model = LitResNet(learning_rate=0.001)
    data_module = CatDogDataModule(
        data_dir="dataset",
        batch_size=32,
        num_workers=4,
        image_size=(64, 64)
    )

    # Initialize trainer with checkpointing and early stopping
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',  # Automatically detect if you have GPU
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                filename='{epoch}-{val_loss:.2f}'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ],
        logger=wandb_logger,
    )

    # Log hyperparameters
    wandb_logger.log_hyperparams({
        'learning_rate': model.hparams.learning_rate,
        'batch_size': data_module.batch_size,
        'image_size': data_module.image_size,
        'model_type': 'ResNet18',
        'optimizer': 'Adam',
        'image_channels': 'grayscale'
    })

    # Train and evaluate the model
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    # Close wandb run
    wandb.finish()


if __name__ == '__main__':
    # Initialize wandb
    wandb.login(key="use your key")

    # Create and set up data module
    data_module = CatDogDataModule(
        data_dir="dataset",
        batch_size=32,
        num_workers=4,
        image_size=(64, 64)
    )
    data_module.setup()

    # Print dataset information
    train_loader = data_module.train_dataloader()
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of training samples: {len(data_module.train_dataset)}")
    print(f"Number of validation samples: {len(data_module.val_dataset)}")
    print(f"Number of test samples: {len(data_module.test_dataset)}")

    # Start training
    main()