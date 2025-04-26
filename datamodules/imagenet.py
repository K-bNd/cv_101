import os
import torch
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
from datasets import load_dataset, Image # Image is useful for type hinting
from PIL import Image as PILImage # To handle potential errors
from torchvision.transforms import v2
# Define standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class HFImageNetDataModule(pl.LightningDataModule):
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
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        cache_dir: str | None = None,
        shuffle_buffer_size: int = 10000 # For IterableDataset shuffling
    ):
        super().__init__()
        self.save_hyperparameters() # Saves args to self.hparams

        self.train_dataset = None
        self.val_dataset = None

        # Define transforms
        self.train_transform = v2.Compose([
            v2.RandomResizedCrop(self.hparams.image_size),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.val_transform = v2.Compose([
            v2.Resize(256), # Standard practice for validation
            v2.CenterCrop(self.hparams.image_size),
            v2.ToImage(),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _apply_transforms(self, examples, transform):
        """
        Helper function to apply transforms to a batch of examples.
        Handles the 'image' key provided by the HF dataset.
        Converts images to RGB if they are not already.
        """
        # Check if 'image' key exists
        if 'image' not in examples:
            raise KeyError("Expected 'image' key in the dataset examples.")

        transformed_images = []
        for img in examples['image']:
            if isinstance(img, PILImage.Image):
                # Ensure image is RGB - some ImageNet images might be grayscale
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                transformed_images.append(transform(img))
            else:
                # Handle cases where data might not be a PIL image as expected
                # You might need more robust error handling or data validation
                print(f"Warning: Encountered non-PIL image data: {type(img)}. Skipping.")
                # Optionally append a placeholder or raise an error
                # For simplicity, let's append None and filter later or handle in collate_fn
                transformed_images.append(None) # Or handle appropriately

        # Filter out None values if any were added
        valid_indices = [i for i, img in enumerate(transformed_images) if img is not None]

        if not valid_indices:
             # Return empty tensors if no valid images processed in the batch
             # Adjust dimensions based on your model's expected input
             return {'pixel_values': torch.empty(0, 3, self.hparams.image_size, self.hparams.image_size),
                     'label': torch.empty(0, dtype=torch.long)}


        # Stack only the valid transformed images
        output_batch = {
            'pixel_values': torch.stack([transformed_images[i] for i in valid_indices]),
            # Filter labels corresponding to valid images
            'label': torch.tensor([examples['label'][i] for i in valid_indices], dtype=torch.long)
        }
        return output_batch


    def setup(self, stage: str | None = None):
        """
        Loads the dataset splits and applies transforms.
        Called automatically by PyTorch Lightning.
        """
        print(f"Setting up data for stage: {stage}")
        try:
            if stage == "fit" or stage is None:
                print("Loading training dataset...")
                # Use streaming=True for IterableDataset behavior
                train_ds_raw = load_dataset(
                    "imagenet-1k",
                    split='train',
                    streaming=True,
                    cache_dir=self.hparams.cache_dir,
                    # trust_remote_code=True # Sometimes needed depending on dataset version/HF changes
                )
                 # Important: Shuffle the streaming dataset!
                train_ds_raw = train_ds_raw.shuffle(
                    buffer_size=self.hparams.shuffle_buffer_size,
                    seed=42 # for reproducibility if needed
                )
                self.train_dataset = train_ds_raw.map(
                    self._apply_transforms,
                    batched=True, # Process multiple examples at once
                    fn_kwargs={"transform": self.train_transform},
                    remove_columns=["image"] # Remove original image column
                )
                # Alternative using set_transform (applies function sample by sample)
                # self.train_dataset.set_transform(lambda x: self._apply_transforms(x, self.train_transform))


                print("Loading validation dataset...")
                val_ds_raw = load_dataset(
                    "imagenet-1k",
                    split='validation',
                    streaming=True, # Also stream validation if desired
                    cache_dir=self.hparams.cache_dir,
                    # trust_remote_code=True
                )
                self.val_dataset = val_ds_raw.map(
                    self._apply_transforms,
                    batched=True,
                    fn_kwargs={"transform": self.val_transform},
                    remove_columns=["image"]
                )
                # Alternative using set_transform:
                # self.val_dataset.set_transform(lambda x: self._apply_transforms(x, self.val_transform))

                print("Dataset setup complete.")

            # Add setup for 'test' or 'predict' stages if needed similarly
            # if stage == "test":
            #    test_ds_raw = load_dataset(...)
            #    self.test_dataset = ...

        except Exception as e:
            print("\n" + "="*40)
            print("ERROR loading dataset:")
            print(f"    {e}")
            print("Potential Issues & Solutions:")
            print("  1. Authentication: Did you run `huggingface-cli login`?")
            print("  2. Dataset Access: Did you accept terms on https://huggingface.co/datasets/imagenet-1k ?")
            print("  3. Network: Check your internet connection.")
            print("  4. Cache Issues: Try clearing the cache dir specified or the default HF cache (~/.cache/huggingface/datasets).")
            print("  5. `trust_remote_code=True`: Sometimes required, uncomment it in `load_dataset` if needed, but be aware of security implications.")
            print("="*40 + "\n")
            raise e # Re-raise the exception after printing help

    def train_dataloader(self):
        if self.train_dataset is None:
             raise RuntimeError("Train dataset not initialized. Run setup() first.")
        # For IterableDataset, set shuffle=False in DataLoader; shuffling is handled by .shuffle() on the dataset itself.
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True, # Usually good for GPU training
            persistent_workers=self.hparams.num_workers > 0, # Avoid worker restart overhead
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Run setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self): # Implement if you have a test split/stage
        if self.test_dataset is None:
             raise RuntimeError("Test dataset not initialized. Run setup() first.")
        return DataLoader(...)