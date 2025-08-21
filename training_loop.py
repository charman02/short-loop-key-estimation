import gc
import os
from pathlib import Path
from typing import Any, List, Dict

import gin  # type: ignore
import numpy as np
import random
from collections import Counter
from datetime import datetime

from training_utils.dataloader import get_datasets
from training_utils.skey import Stone
from training_utils.skey_loss import CrossPowerSpectralDensityLoss
from skey.hcqt import VQT
from training_utils.gin import get_save_dict
from training_utils.callbacks import (
    EarlyStoppingCallback,
    NaNLoopCallback,
    restart_from_checkpoint,
    save_fn,
)
from training_utils.gin import parse_gin
from training_utils.scheduler import (
    get_learning_rate_scheduler,
    get_weights_decay_scheduler,
)
from training_utils.training import clip_gradients, get_optimizer, update_optimizer, cleanup

from skey.key_detection import load_checkpoint, load_model_components, infer_key

from madmom.evaluation.key import key_label_to_class, error_type

import torch
from tqdm import tqdm
import wandb

def create_save_dir(save_dir: str, name: str, train_type: str, circle_type: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        save_dir,
        "models",
        train_type,
        str(circle_type),
        name,
        timestamp
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print("PARAMETERS used for SAVING:")
    print("\t save_model_dir: {}".format(save_dir))
    print("\t exp_name: {}".format(name))
    return save_dir


class ModelCustomWrapper:
    def __init__(
        self,
        learning_rate: float,
        device: str,
        n_steps: int,
        n_epochs: int,
        train_type: str,
        circle_type: int,
        ckpt: Dict[str, Any]
    ) -> None:

        self.device = torch.device(device)
        self.lr = learning_rate
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.current_epoch = 0
        self.circle_type = circle_type
        self.train_type = train_type
        self.hcqt, self.chromanet, self.crop_fn = load_model_components(ckpt, self.device)

        # MODELS
        self.stone = Stone(VQT(), device=self.device).to(device) # type: ignore
        self.stone.load_state_dict(ckpt['stone'])

        # LOSS
        self.loss_fn = CrossPowerSpectralDensityLoss(self.circle_type, self.device).cuda(self.device)

        # OPTIMIZER
        self.optimizer = get_optimizer(self.stone)
        self.scaler = torch.cuda.amp.GradScaler()  # type: ignore

        # TRAINING STEPS
        self.step = (
            lambda self, batch: self.stone(batch)
            )

        # SCHEDULES
        self.lr_schedule = get_learning_rate_scheduler(self.lr, self.n_epochs, self.n_steps)  # comment to disable lr scheduler
        # self.lr_schedule = np.full(self.n_steps, self.lr, dtype=float)  # uncomment to disable lr scheduler
        self.wd_schedule = get_weights_decay_scheduler(self.n_epochs, self.n_steps)

    def training_step(
        self, batch: Any, current_global_step: int, current_epoch: int
    ) -> Any:
        self.current_global_step = current_global_step
        self.current_epoch = current_epoch
        lr_step = self.lr_schedule[self.current_global_step]  # comment to disable lr scheduler
        # lr_step = self.optimizer.param_groups[0]['lr']  # uncomment to disable lr scheduler
        wd_step = self.wd_schedule[self.current_global_step]

        # update weight decay and learning rate according to their schedule
        self.optimizer = update_optimizer(self.optimizer, lr_step, wd_step)
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # POWERFUL
            loss = self.loss_fn(self.step(self, batch))
        self.scaler.scale(loss["loss"]).backward()

        # unscale the gradients of optimizer's assigned params in-place
        self.scaler.unscale_(self.optimizer)
        self.stone = clip_gradients(self.stone)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # remove data
        torch.cuda.empty_cache()
        del batch

        return loss["loss_to_print"]

    def validation_step(self, predictions: List, batch: Any) -> Any:
        with torch.no_grad():
            logits = self.step(self, batch)
            loss = self.loss_fn(logits)

            try:
                for i in range(batch['audio'].shape[0]):
                    waveform = batch['audio'][i].to(self.device)
                    label = batch["keymode"][0][i]
                    pred_key = infer_key(self.hcqt, self.chromanet, self.crop_fn, waveform, self.device)
                    true_key = label.title().replace("Minor", "minor").replace("Major", "major")
                    pred_key = pred_key.title().replace("Minor", "minor").replace("Major", "major")
                    if pred_key == "Error":
                        print(f"Invalid prediction: {pred_key}")
                        return loss["loss"]
                    gt_class = key_label_to_class(true_key)
                    pred_class = key_label_to_class(pred_key)
                    score_category = error_type(pred_class, gt_class, strict_fifth=True)
                    if not isinstance(score_category, tuple) or len(score_category) != 2:
                        print(f"Invalid error_type output: {score_category}")
                        return loss["loss"]
                    score, category = score_category
                    predictions.append({
                        "true_key": true_key,
                        "pred_key": pred_key,
                        "category": category
                    })
            except Exception as e:
                print(f"Error: {e}")
            
        return loss["loss"]


def do_one_iter(
    model: ModelCustomWrapper,
    ds_train_iter: Any,
    ds_val_iter: Any,
    epoch: int,
    n_steps: int,
    val_steps: int,
    progress_bar: Any,
) -> float:
    # --- TRAINING ---
    model.stone.train()
    for i in range(n_steps):
        current_global_step = i + epoch * n_steps
        train_batch = next(ds_train_iter)
        train_loss = model.training_step(
            batch=train_batch,
            current_epoch=epoch,
            current_global_step=current_global_step,
        )
        wandb.log({
            "train/loss_total": train_loss["loss_total"].item(),
            "train/loss_pos": train_loss["loss_pos"].item(),
            "train/loss_equi": train_loss["loss_equi"].item(),
            "train/loss_mode": train_loss.get("loss_mode", torch.tensor(0.0)).item(),
            "epoch": epoch,
            "step": current_global_step
        })
        progress_bar.set_postfix({
            "train_loss": train_loss["loss_total"].item(),
            "train_loss_pos": train_loss["loss_pos"].item(),
            "train_loss_equi": train_loss["loss_equi"].item(),
            **({"train_loss_mode": train_loss["loss_mode"].item()} if model.train_type == "ks_mode" else {})
        })
        progress_bar.update(1)

    # --- VAL ---
    model.stone.eval()
    predictions = []
    val_losses = []

    for _ in range(val_steps):
        val_batch = next(ds_val_iter)
        val_loss = model.validation_step(predictions=predictions, batch=val_batch)
        val_losses.append(val_loss.item())

        wandb.log({
            "val/loss_per_step": val_loss,
        })
        
        progress_bar.set_postfix({"val_loss": val_loss.item()})
        progress_bar.update(1)

    avg_val_loss = float(np.mean(val_losses))

    weights = {'correct': 1.0, 'fifth': 0.5, 'relative': 0.3, 'parallel': 0.2, 'other': 0.0}
    counts = Counter(p['category'] for p in predictions)
    total = len(predictions)
    weighted = round(sum(weights[k] * counts.get(k, 0) * 100 / total for k in weights), 1)

    wandb.log({
            "val/weighted": weighted,
            "val/loss_per_epoch": avg_val_loss,
            "epoch": epoch,
    })
    
    val_loss_ckpt = avg_val_loss

    # Clean memory
    torch.cuda.empty_cache()
    _ = gc.collect()
    return val_loss_ckpt

@gin.configurable
def main_loop(
    n_epochs: int,
    n_steps: int,
    val_steps: int,
    learning_rate: float,
    gin_file: str,
    save_dir: str,
    name: str,
    train_type: str,
    circle_type: int,
    save_epochs: List = [25, 50, 75, 100],
    seed: int = 42,
    ) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Seeding ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    gin.parse_config_file(gin_file)
    save_dict = get_save_dict()
    save_dict["gin_info"] = parse_gin(gin_file)
    wandb.init(project="skey-finetune", name="t4", config=save_dict)
    # save_dir = create_save_dir(save_dir, name, train_type, circle_type,)
    # ckpt = load_checkpoint(os.path.join(save_dir, "skey.pt"))
    # where pretrained checkpoint lives (NO timestamp)
    base_ckpt_path = os.path.join(save_dir, "models", train_type, str(circle_type), name, "skey.pt")
    ckpt = load_checkpoint(base_ckpt_path)
    
    # where new fine-tuned models will be saved (WITH timestamp)
    save_dir = create_save_dir(save_dir, name, train_type, circle_type)

    hcqt, chromanet, crop_fn = load_model_components(ckpt, device)
    print(" ------- CREATING model -----------")
    model = ModelCustomWrapper(
        learning_rate=learning_rate,
        device=device,
        n_steps=n_steps,
        n_epochs=n_epochs,
        train_type=train_type,
        circle_type=circle_type,
        ckpt=ckpt
    )
    # DATALOADERS
    print(" ------- CREATING datasets ----------")
    ds_train, ds_val = get_datasets(device=device)
    save_dict["audio"] = {
        "dur": int(ds_train.duration),
        "sr": ds_train.sr,
    }
    model, save_dict = restart_from_checkpoint(model, save_dict, save_dir) # if an experiment of the same name was launched before
    early_stopping = EarlyStoppingCallback(save_dir, best_score=save_dict["val_loss"])
    nan_loop = NaNLoopCallback(save_dir)
    ds_train_iter = iter(ds_train)
    ds_val_iter = iter(ds_val)
    epoch, val_loss_ckpt = [save_dict["epoch"], save_dict["val_loss"]]
    
    while epoch < n_epochs:
        print("\nepoch {}/{}".format(epoch + 1, n_epochs))
        # Training loop
        val_loss_ckpt = do_one_iter(
            model=model,
            ds_train_iter=ds_train_iter,
            ds_val_iter=ds_val_iter,
            epoch=epoch,
            n_steps=n_steps,
            val_steps=val_steps,
            progress_bar=tqdm(total=n_steps + val_steps, desc=f"Epoch {epoch+1}"),
        )
        model, ds_train, save_dict, epoch = nan_loop(
            False, model, save_dict, ds_train, n_steps, epoch
        )
        # epoch is updated inside nan_loop
        save_dict["epoch"], save_dict["val_loss"] = [epoch, val_loss_ckpt]
        early_stopping(val_loss_ckpt, model, save_dict, epoch)
        if early_stopping.best_score == val_loss_ckpt:
            save_fn(save_dict, model, os.path.join(save_dir, "best_model.pt"))
        save_fn(save_dict, model, os.path.join(save_dir, "last_iter.pt"))
        if epoch in save_epochs:
            save_fn(
                save_dict, model, os.path.join(save_dir, "epoch_{}.pt".format(epoch))
            )
    cleanup()


    # Log final model checkpoint as W&B artifact
    final_checkpoint_path = os.path.join(save_dir, "last_iter.pt")
    if os.path.exists(final_checkpoint_path):
        artifact = wandb.Artifact(
            name=name,
            type="model"
        )
        artifact.add_file(final_checkpoint_path)
        wandb.log_artifact(artifact)

    wandb.finish()
    
    return
