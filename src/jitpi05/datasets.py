import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

from jitpi05.config import DATASET_ID, EPISODE_INDEX


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(
        "uint8"
    )
    return Image.fromarray(array)


def load_episode() -> tuple[LeRobotDataset, dict, torch.Tensor]:
    dataset = LeRobotDataset(DATASET_ID, episodes=[EPISODE_INDEX])
    frame = dict(dataset[0])
    action_rows = dataset.select_columns("action")[: min(50, len(dataset))]["action"]
    demonstration = torch.stack(
        [torch.as_tensor(action, dtype=torch.float32) for action in action_rows]
    )
    return dataset, frame, demonstration
