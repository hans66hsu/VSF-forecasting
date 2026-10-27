from __future__ import annotations

from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm.auto import tqdm


def image_transform(image_size: int):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(**spt.data.dataset_stats.ImageNet),
            v2.Resize((image_size, image_size)),
        ]
    )


class _WindowedEmbeddingDataset(Dataset):
    def __init__(self, source, image_embeddings, action_embeddings=None):
        self.source = source
        self.image_embeddings = image_embeddings
        self.action_embeddings = action_embeddings
        self.steps = torch.arange(source.num_steps) * source.frameskip
        if action_embeddings is not None:
            counts = torch.as_tensor(
                np.maximum(source.lengths - source.frameskip + 1, 0),
                dtype=torch.long,
            )
            self.action_offsets = torch.cat(
                (torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1])
            )

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        episode, start = self.source.clip_indices[index]
        rows = int(self.source.offsets[episode]) + start + self.steps
        actions = (
            self.source[index]["action"]
            if self.action_embeddings is None
            else self.action_embeddings[
                self.action_offsets[episode] + start + self.steps
            ]
        )
        return self.image_embeddings[rows], actions


def _column_normalizer(dataset, column: str):
    values = torch.from_numpy(np.asarray(dataset.get_col_data(column)))
    values = values[~torch.isnan(values).any(dim=1)]
    scaler = swm.data.ZScoreScaler(
        mean=values.mean(0, keepdim=True).numpy(),
        std=values.std(0, keepdim=True).numpy(),
    )
    return spt.data.transforms.WrapTorchTransform(scaler, source=column, target=column)


def load_embedding_dataset(
    cfg: DictConfig,
    model,
    device: torch.device,
    loader_kwargs: dict,
) -> _WindowedEmbeddingDataset:
    data_path = Path(to_absolute_path(cfg.dataset.path))
    image_path = data_path.with_name(f"{data_path.stem}_image_embeddings.pt")
    if not image_path.is_file():
        print(f"Encoding images to {image_path}")
        image_dataset = swm.data.load_dataset(
            str(data_path),
            transform=None,
            num_steps=1,
            frameskip=1,
            keys_to_load=["pixels"],
            keys_to_cache=[],
        )
        image_dataset.transform = spt.data.transforms.WrapTorchTransform(
            image_transform(int(cfg.dataset.image_size)),
            source="pixels",
            target="pixels",
        )
        embeddings = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(
                DataLoader(image_dataset, shuffle=False, **loader_kwargs),
                desc="Encoding images",
                unit="batch",
            ):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    embedding = model.encode(
                        {"pixels": batch["pixels"].to(device, non_blocking=True)}
                    )["emb"][:, 0]
                embeddings.append(embedding.cpu())
        torch.save(torch.cat(embeddings), image_path)
        del embeddings
        print(f"Saved image embeddings to {image_path}")
    image_embeddings = torch.load(
        image_path, map_location="cpu", weights_only=True, mmap=True
    )

    action_embeddings = None
    if model.freeze_action_encoder:
        action_path = data_path.with_name(f"{data_path.stem}_action_embeddings_f5.pt")
        if not action_path.is_file():
            print(f"Encoding actions to {action_path}")
            action_dataset = swm.data.load_dataset(
                str(data_path),
                transform=None,
                num_steps=1,
                frameskip=5,
                keys_to_load=["action"],
                keys_to_cache=["action"],
            )
            action_dataset.transform = _column_normalizer(action_dataset, "action")
            embeddings = []
            model.eval()
            with torch.no_grad():
                for batch in DataLoader(action_dataset, shuffle=False, **loader_kwargs):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        embedding = model.action_encoder(
                            torch.nan_to_num(
                                batch["action"].to(device, non_blocking=True), 0.0
                            )
                        )[:, 0]
                    embeddings.append(embedding.cpu())
            torch.save(torch.cat(embeddings), action_path)
            del embeddings
            print(f"Saved action embeddings to {action_path}")
        action_embeddings = torch.load(
            action_path, map_location="cpu", weights_only=True, mmap=True
        )

    source = swm.data.load_dataset(
        str(data_path),
        transform=None,
        num_steps=int(cfg.dataset.num_steps),
        frameskip=int(cfg.dataset.frameskip),
        keys_to_load=["action"],
        keys_to_cache=[] if action_embeddings is not None else ["action"],
    )
    if action_embeddings is None:
        source.transform = _column_normalizer(source, "action")
    return _WindowedEmbeddingDataset(source, image_embeddings, action_embeddings)
