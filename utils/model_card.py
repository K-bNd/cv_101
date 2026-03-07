from typing import Optional

from huggingface_hub import ModelCard, ModelCardData

from configs.config_models import TrainConfig


_METRIC_LABELS = {
    "test/loss": "Loss",
    "test/acc": "Accuracy",
    "test/top5": "Top-5 Accuracy",
    "test/iou": "Mean IoU",
}


def create_model_card(
    model_name: str,
    dataset: str,
    pipeline_tag: str,
    config: TrainConfig,
    test_metrics: dict[str, float],
    wandb_url: Optional[str] = None,
) -> ModelCard:
    """Build a HuggingFace ModelCard from training results.

    Args:
        model_name: Short model identifier (e.g. "resnet50")
        dataset: Dataset name (e.g. "imagenet")
        pipeline_tag: HuggingFace pipeline tag (e.g. "image-classification")
        config: Pydantic TrainConfig used for the run
        test_metrics: Dict of logged metric names to scalar values, e.g. {"test/acc": 0.92}
        wandb_url: Optional WandB run URL for linking to training logs

    Returns:
        ModelCard ready to push to the HuggingFace Hub
    """

    # Build structured metric entries for the card YAML front-matter
    metric_entries = []
    for key, value in test_metrics.items():
        label = _METRIC_LABELS.get(key, key)
        metric_entries.append({"type": key.replace("/", "_"), "value": round(value, 4), "name": label})

    card_data = ModelCardData(
        language="en",
        license="mit",
        library_name="pytorch",
        tags=[pipeline_tag, "pytorch", "lightning", dataset],
        datasets=[dataset],
        pipeline_tag=pipeline_tag,
    )

    # Format hyperparameters table
    config_dict = config.model_dump()
    config_rows = "\n".join(
        f"| `{k}` | `{v}` |" for k, v in config_dict.items()
    )

    # Format test metrics table
    metrics_rows = "\n".join(
        f"| {_METRIC_LABELS.get(k, k)} | {v:.4f} |"
        for k, v in test_metrics.items()
    )
    metrics_section = (
        f"## Test Results\n\n| Metric | Value |\n|--------|-------|\n{metrics_rows}"
        if metrics_rows
        else ""
    )

    wandb_section = (
        f"## Training Logs\n\nFull training curves available on [WandB]({wandb_url})."
        if wandb_url
        else ""
    )

    content = f"""{card_data.to_yaml()}

# {model_name} trained on {dataset}

This model was trained using the [cv_101](https://github.com/K-bNd/cv_101) deep learning framework.

- **Task:** {pipeline_tag}
- **Architecture:** {model_name}
- **Dataset:** {dataset}
- **Framework:** PyTorch + PyTorch Lightning

{metrics_section}

## Training Configuration

| Parameter | Value |
|-----------|-------|
{config_rows}

{wandb_section}
"""
    return ModelCard(content)
