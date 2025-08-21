from typing import Any, Tuple
import torch
import torchaudio
from torch.utils.data import IterableDataset
import gin  # type: ignore
import random
import os
from skey.key_detection import load_audio


@gin.configurable
class FMAKeyDataset(IterableDataset):  # works for FMAKv2 and GiantSteps (and other datasets if the format of audio and labels is the same)
    def __init__(
        self,
        split_file: str,
        root_dir: str,
        sr: int,
        duration: int,
        batch_size: int,
        device: str,
        dataset_type: str,  # "supervised", "selfsupervised", "mix"
        shuffle: bool = True,
    ):
        assert dataset_type in ["supervised", "selfsupervised", "mix"], "Invalid dataset_type!"
        self.sr = sr
        self.duration = duration
        self.batch_size = batch_size
        self.device = device
        self.audio_len = sr * duration
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.iter_count = 0  # for alternating in "mix"
        self.shuffle = shuffle

        with open(split_file, "r") as f:
            self.data = [line.strip().split("|") for line in f.readlines()]  # [(path, label)]
            
    def __iter__(self):
        if self.shuffle:
            while True:
                batch_audio = []
                batch_labels = []
                batch_paths = []
    
                for _ in range(self.batch_size):
                    path, label = random.choice(self.data)
                    full_path = os.path.join(self.root_dir, os.path.basename(path))
    
                    try:
                        if self.dataset_type == "supervised":
                            audio = load_audio(full_path, self.sr, self.audio_len)
                            batch_audio.append(audio)
                            batch_labels.append(label)
                            batch_paths.append(full_path)
    
                        elif self.dataset_type == "selfsupervised":
                            audio_1 = load_audio(full_path, self.sr, self.audio_len)
                            audio_2 = load_audio(full_path, self.sr, self.audio_len)
                            batch_audio.append(audio_1)
                            batch_audio.append(audio_2)
                            batch_labels.extend(["-1", "-1"])
                            batch_paths.extend([full_path, full_path])
    
                        elif self.dataset_type == "mix":
                            if self.iter_count % 2 == 0:
                                audio = load_audio(full_path, self.sr, self.audio_len)
                                batch_audio.append(audio.unsqueeze(1))
                                batch_labels.append(label)
                                batch_paths.append(full_path)
                            else:
                                audio_1 = load_audio(full_path, self.sr, self.audio_len)
                                audio_2 = load_audio(full_path, self.sr, self.audio_len)
                                batch_audio.append(audio_1)
                                batch_audio.append(audio_2)
                                batch_labels.extend(["-1", "-1"])
                                batch_paths.extend([full_path, full_path])
    
                    except Exception as e:
                        print(f"Error loading {full_path}: {e}")
                        continue
    
                self.iter_count += 1
                audio_tensor = torch.stack(batch_audio).to(self.device)
                yield {
                    "audio": audio_tensor,
                    "keymode": (batch_labels,),
                    "path": batch_paths,
                }
        else:
            for path, label in self.data:
                full_path = os.path.join(self.root_dir, os.path.basename(path))
                try:
                    audio = load_audio(full_path, self.sr, self.audio_len)
                    audio_tensor = audio.unsqueeze(0).to(self.device)
                    yield {
                        "audio": audio_tensor,
                        "keymode": ([label],),
                        "path": [full_path],
                    }
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")
                    continue

@gin.configurable
def get_datasets(
        device: str,
        batch_size: int,
        sr: int,
        duration: int,
        dataset_type_train: str,
        dataset_type_test: str,
        train_txt_path: str,
        train_audio_dir: str,
        test_txt_path: str,
        test_audio_dir: str,
) -> Tuple[Any, Any]:
    from training_utils.dataloader import FMAKeyDataset

    ds_train = FMAKeyDataset(
        split_file=train_txt_path,
        root_dir=train_audio_dir,
        sr=sr,
        duration=duration,
        batch_size=batch_size,
        device=device,
        dataset_type=dataset_type_train
    )
    ds_test = FMAKeyDataset(
        split_file=test_txt_path,
        root_dir=test_audio_dir,
        sr=sr,
        duration=duration,
        batch_size=batch_size,
        device=device,
        dataset_type=dataset_type_test,
        shuffle=False
    )
    
    return ds_train, ds_test
    