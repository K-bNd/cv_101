import os
import torch
from torch.utils.data import DataLoader, IterableDataset
import lightning as L
import huggingface_hub
from datasets import load_dataset, Image  # Image is useful for type hinting
from PIL import Image as PILImage  # To handle potential errors
from torchvision.transforms import v2

from configs.config_models import ImageNetTrainConfig

# Define standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageNetDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for ImageNet-1k using Hugging Face datasets.

    Args:
        batch_size (int): Batch size for DataLoader. Defaults to 32.
        num_workers (int): Number of workers for DataLoader. Defaults to 4.
        image_size (int): Target image size (height and width). Defaults to 224.
        cache_dir (str, optional): Directory for Hugging Face datasets cache.
                                   Defaults to None (uses default HF cache).
        shuffle_buffer_size (int): Size of the buffer for shuffling the
                                   streaming dataset. Only used for training.
                                   Defaults to 10000. A larger buffer gives
                                   better shuffling but uses more memory.
    """

    def __init__(
        self,
        config: ImageNetTrainConfig,
        data_dir: str = "../datasets/imagenet",
    ):
        super().__init__()
        self.save_hyperparameters("data_dir")  # Saves args to self.hparams
        # Define transforms
        self.config = config
        self.train_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((config.train_res, config.train_res)),
                (
                    v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET)
                    if config.auto_augment
                    else v2.Identity()
                ),
                v2.RandAugment() if config.rand_augment else v2.Identity(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.val_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((config.val_res, config.val_res)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.test_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((config.val_res, config.val_res)),
                v2.TenCrop((config.val_res, config.val_res)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _apply_transforms(self, examples, transform):
        """
        Helper function to apply transforms to a batch of examples.
        Handles the 'image' key provided by the HF dataset.
        Converts images to RGB if they are not already.
        """
        # Check if 'image' key exists
        if "image" not in examples:
            raise KeyError("Expected 'image' key in the dataset examples.")

        # handling corrupt images in the dataset
        if not hasattr(self, "_pillow_patched"):
            original_getexif = PILImage.Image.getexif

            def safe_getexif(self):
                try:
                    return original_getexif(self)
                except UnicodeDecodeError:
                    return PILImage.Exif()

            PILImage.Image.getexif = safe_getexif
            # Apply the patch
            self._pillow_patched = True
        transformed_images = []
        for img in examples["image"]:
            try:
                # handle RGBA images for normalization
                rgb_img = img.convert("RGB")
                transformed_images.append(transform(rgb_img))
            except Exception as e:
                print(f"Error processing image: {e}")
                transformed_images.append(None)

        # Filter out None values if any were added
        valid_indices = [
            i for i, img in enumerate(transformed_images) if img is not None
        ]

        if not valid_indices:
            # Return empty tensors if no valid images processed in the batch
            # Adjust dimensions based on your model's expected input
            return {
                "x": torch.empty((0, 3, self.config.train_res, self.config.train_res)),
                "y": torch.empty(0, dtype=torch.long),
            }

        # Stack only the valid transformed images
        output_batch = {
            "x": torch.stack([transformed_images[i] for i in valid_indices]),
            # Filter labels corresponding to valid images
            "y": torch.tensor(
                [examples["label"][i] for i in valid_indices], dtype=torch.long
            ),
        }
        return output_batch

    def prepare_data(self):
        huggingface_hub.login(add_to_git_credential=False)
        self.full_train_dataset = load_dataset(
            "imagenet-1k",
            split="train",
            data_dir=self.hparams.data_dir,
            token=True,
            streaming=False,
            trust_remote_code=True,  # ImageNet-1k from ILSVRC requires remote code execution
        ).with_format("torch")
        self.test_dataset = load_dataset(
            "imagenet-1k",
            split="validation",
            data_dir=self.hparams.data_dir,
            token=True,
            streaming=False,
            trust_remote_code=True,  # Sometimes needed depending on dataset version/HF changes
        ).with_format('torch')

    def setup(self, stage: str | None = None):
        """
        Loads the dataset splits and applies transforms.
        Called automatically by PyTorch Lightning.
        """
        print(f"Setting up data for stage: {stage}")
        try:
            if stage == "fit":
                self.train_dataset, self.val_dataset = (
                    self.full_train_dataset.train_test_split(
                        test_size=50000, seed=42
                    ).values()
                )

                self.train_dataset.set_transform(
                    lambda x: self._apply_transforms(x, self.train_transform)
                )
                self.val_dataset.set_transform(
                    lambda x: self._apply_transforms(x, self.val_transform)
                )

            # Add setup for 'test' or 'predict' stages if needed similarly
            if stage == "test":
                # Use streaming=True for IterableDataset behavior
                self.test_dataset.set_transform(
                    lambda x: self._apply_transforms(x, self.test_transform)
                )

        except Exception as e:
            print("\n" + "=" * 40)
            print("ERROR loading dataset:")
            print(f"    {e}")
            print("Potential Issues & Solutions:")
            print("  1. Authentication: Did you run `huggingface-cli login`?")
            print(
                "  2. Dataset Access: Did you accept terms on https://huggingface.co/datasets/imagenet-1k ?"
            )
            print("  3. Network: Check your internet connection.")
            print(
                "  4. Cache Issues: Try clearing the cache dir specified or the default HF cache (~/.cache/huggingface/datasets)."
            )
            print(
                "  5. `trust_remote_code=True`: Sometimes required, uncomment it in `load_dataset` if needed, but be aware of security implications."
            )
            print("=" * 40 + "\n")
            raise e  # Re-raise the exception after printing help

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Run setup() first.")
        # For IterableDataset, set shuffle=False in DataLoader; shuffling is handled by .shuffle() on the dataset itself.
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,  # Usually good for GPU training
            persistent_workers=self.config.num_workers
            > 0,  # Avoid worker restart overhead
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def test_dataloader(self):  # Implement if you have a test split/stage
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )
