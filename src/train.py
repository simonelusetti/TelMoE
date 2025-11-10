import os
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn.functional as F

from tqdm import tqdm

from dora import get_xp, hydra_main

from .data import initialize_dataloaders
from .models import ExpertModel
from .utils import get_logger, should_disable_tqdm
from .metrics import build_train_table, build_eval_table, evaluate_factor_metrics


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

class ExpertTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.weights = self._prepare_expert_weights(cfg)
        self.contrastive_tau = cfg.model.contrastive_tau
        max_epochs = cfg.train.max_epochs
        if max_epochs is not None:
            max_epochs = int(max_epochs)
            if max_epochs <= 0:
                raise ValueError("cfg.train.max_epochs must be > 0 when provided.")
        target_recall = cfg.train.target_recall
        if target_recall is not None:
            target_recall = float(target_recall)
            if not (0.0 < target_recall <= 1.0):
                raise ValueError("cfg.train.target_recall must be in (0, 1].")
        if max_epochs is None and target_recall is None:
            raise ValueError("Provide at least one of train.max_epochs or train.target_recall.")
        self.max_epochs = max_epochs
        self.target_recall = target_recall

    @staticmethod
    def _prepare_expert_weights(cfg):
        weights_cfg = cfg.model.loss_weights
        return {
            "sent": float(weights_cfg.sent),
            "token": float(weights_cfg.token),
            "entropy": float(weights_cfg.entropy),
            "overlap": float(weights_cfg.overlap),
            "diversity": float(weights_cfg.diversity),
            "balance": float(weights_cfg.balance),
        }

    @staticmethod
    def _factor_name_to_index(name: str):
        if not isinstance(name, str):
            return None
        if not name.startswith("factor_"):
            return None
        try:
            return int(name.split("_")[-1])
        except (ValueError, TypeError):
            return None

    def _select_factor(self, factor_metrics, model, *, maximize=True):
        if not factor_metrics:
            return None, None
        target_idx = None
        target_name = None
        target_stats = None
        best_val = float("-inf") if maximize else float("inf")
        for name, stats in factor_metrics.items():
            f1 = stats.get("f1")
            if f1 is None:
                continue
            idx = self._factor_name_to_index(name)
            if idx is None or model.is_expert_frozen(idx):
                continue
            val = float(f1)
            if maximize:
                if val > best_val:
                    best_val = val
                    target_idx = idx
                    target_name = name
                    target_stats = stats
            else:
                if val < best_val:
                    best_val = val
                    target_idx = idx
                    target_name = name
                    target_stats = stats
        if target_idx is None:
            return None, None
        return target_idx, {"name": target_name, "f1": best_val, "stats": target_stats}

    def _build_model(self, cfg, num_experts):
        model_cfg = deepcopy(cfg.model)
        model_cfg.expert.num_experts = num_experts
        return ExpertModel(model_cfg).to(cfg.device)

    @staticmethod
    def _build_optimizer(model, optim_cfg):
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(optim_cfg.lr),
            weight_decay=float(optim_cfg.weight_decay),
            betas=tuple(optim_cfg.betas),
        )

    def _loss(self, model, batch, device):
        embeddings = batch["embeddings"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        incoming = batch.get("incoming")
        outgoing = batch.get("outgoing")
        if incoming is not None:
            incoming = incoming.to(device, non_blocking=True)
        if outgoing is not None:
            outgoing = outgoing.to(device, non_blocking=True)

        outputs = model(embeddings, attention_mask, incoming, outgoing)
        anchor = outputs["anchor"]
        reconstruction = outputs["reconstruction"]

        anchor = F.normalize(anchor, dim=-1)
        reconstruction = F.normalize(reconstruction, dim=-1)

        logits = anchor @ reconstruction.t() / max(self.contrastive_tau, 1e-6)
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, targets)
        loss_ba = F.cross_entropy(logits.t(), targets)
        sent_loss = 0.5 * (loss_ab + loss_ba)

        token_reconstruction = outputs.get("token_reconstruction")
        if token_reconstruction is not None:
            mask = attention_mask.unsqueeze(-1).to(dtype=token_reconstruction.dtype)
            diff = token_reconstruction - embeddings
            token_loss = (diff.pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            token_loss = embeddings.new_tensor(0.0)

        entropy_loss = outputs["entropy"].mean()
        overlap_loss = outputs["overlap"].mean()
        diversity_loss = outputs["diversity"]
        balance_loss = outputs["balance"]

        loss_components = {
            "sent": sent_loss,
            "token": token_loss,
            "entropy": entropy_loss,
            "overlap": overlap_loss,
            "diversity": diversity_loss,
            "balance": balance_loss,
        }

        total_loss = sum(self.weights[key] * loss_components[key] for key in loss_components)
        metrics = {key: float(value.detach()) for key, value in loss_components.items()}
        metrics["total"] = float(total_loss.detach())
        return total_loss, metrics

    def _run_epoch(
        self,
        model,
        loader,
        device,
        *,
        train=False,
        optimizer=None,
        grad_clip=0.0,
        desc="",
        disable_progress=False,
    ):
        if train and optimizer is None:
            raise ValueError("Optimizer must be provided when train=True.")

        sums = defaultdict(float)
        counts = defaultdict(int)

        model.train() if train else model.eval()
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            iterator = tqdm(loader, desc=desc, disable=disable_progress)
            for batch in iterator:
                if train:
                    optimizer.zero_grad(set_to_none=True)
                total_loss, metrics = self._loss(model, batch, device)
                if train:
                    total_loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                for name, value in metrics.items():
                    sums[name] += value
                    counts[name] += 1

        return {name: (sums[name] / counts[name]) if counts[name] else 0.0 for name in sums}

    def _run_eval_step(
        self,
        model,
        loader,
        device,
        *,
        desc,
        logger,
        stage_id=None,
        stage_epoch=None,
        global_epoch=None,
        xp=None,
        record_epoch=False,
        disable_progress=False,
    ):
        eval_metrics = self._run_epoch(
            model,
            loader,
            device,
            train=False,
            desc=desc,
            disable_progress=disable_progress,
        )
        factor_metrics = evaluate_factor_metrics(model, loader, device, logger)

        eval_table = build_eval_table(factor_metrics)
        if stage_id is not None and stage_epoch is not None:
            logger.info(
                "\nEval factor metrics (stage %d epoch %d):\n%s",
                stage_id,
                stage_epoch,
                eval_table,
            )
        else:
            logger.info("\nEval factor metrics (%s):\n%s", desc, eval_table)

        if record_epoch and xp is not None and global_epoch is not None:
            xp.link.push_metrics({f"expert/eval_epoch/{global_epoch}": eval_metrics})
            if factor_metrics:
                xp.link.push_metrics({f"expert/factors_epoch/{global_epoch}": factor_metrics})

        return eval_metrics, factor_metrics

    def _should_advance_stage(self, stage_id, stage_epoch, factor_metrics, logger):
        best_recall = None
        if factor_metrics:
            best_recall = max(stats.get("recall", 0.0) for stats in factor_metrics.values())

        min_epochs = 2

        if stage_epoch >= min_epochs:
            if self.target_recall is not None and best_recall is not None:
                if best_recall >= self.target_recall:
                    logger.info(
                        "Reached recall %.4f (>= %.4f) at stage %d epoch %d; moving to next stage.",
                        best_recall,
                        self.target_recall,
                        stage_id,
                        stage_epoch,
                    )
                    return True

            if self.max_epochs is not None and stage_epoch >= self.max_epochs:
                if self.target_recall is None:
                    logger.info(
                        "Reached maximum of %d epochs at stage %d; moving to next stage.",
                        self.max_epochs,
                        stage_id,
                    )
                elif best_recall is not None:
                    logger.info(
                        "Stopping at stage %d after %d epochs; best recall %.4f < target %.4f.",
                        stage_id,
                        stage_epoch,
                        best_recall,
                        self.target_recall,
                    )
                else:
                    logger.info(
                        "Reached maximum of %d epochs without factor metrics at stage %d; moving on.",
                        self.max_epochs,
                        stage_id,
                    )
                return True

        if self.target_recall is not None and not factor_metrics and self.max_epochs is None and stage_epoch >= min_epochs:
            if self.target_recall is None:
                logger.info(
                    "Reached minimum epochs and recall target undefined; moving on (stage %d).",
                    stage_id,
                )
            return True

        return False

    def _save_checkpoint(self, model, path, logger):
        payload = {
            "state_dict": model.state_dict(),
            "num_experts": model.num_experts,
        }
        torch.save(payload, path, _use_new_zipfile_serialization=False)
        logger.info("Saved ExpertModel checkpoint (%d experts) to %s", model.num_experts, path)


    def _load_checkpoint_payload(self, path, device):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert checkpoint not found at {path}")
        payload = torch.load(path, map_location=device)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            ckpt_experts = payload.get("num_experts")
        else:
            state_dict = payload
            ckpt_experts = None
        return state_dict, ckpt_experts


    def train(self, cfg, logger, train_dl, eval_dl, xp):
        device = cfg.device
        target_experts = max(int(cfg.train.target_experts), 2)
        current_experts = min(int(cfg.model.expert.num_experts), target_experts)
        model = self._build_model(cfg, current_experts)

        optim_cfg = cfg.model.optim
        optimizer = self._build_optimizer(model, optim_cfg)
        grad_clip = cfg.train.grad_clip
        disable_progress = should_disable_tqdm()

        checkpoint_path = "expert_model.pth"
        best_record = {"f1": float("-inf"), "stage": None, "factor": None}
        total_stages = (target_experts - current_experts) + 1
        stage = 0
        global_epoch = 0
        training_active = True

        while training_active:
            stage_id = stage + 1
            logger.info(
                "Stage %d/%d: training with %d experts",
                stage_id,
                total_stages,
                model.num_experts,
            )
            latest_eval_metrics = None
            latest_factor_metrics = None
            stage_epoch = 0
            while True:
                stage_epoch += 1
                desc = f"Expert Train S{stage_id}E{stage_epoch}"
                train_metrics = self._run_epoch(
                    model,
                    train_dl,
                    device,
                    train=True,
                    optimizer=optimizer,
                    grad_clip=grad_clip,
                    desc=desc,
                    disable_progress=disable_progress,
                )

                train_table = build_train_table(train_metrics, model=model, weights=self.weights)
                logger.info(
                    "\nTrain metrics (stage %d epoch %d):\n%s",
                    stage_id,
                    stage_epoch,
                    train_table,
                )

                global_epoch += 1
                xp.link.push_metrics({f"expert/train/{global_epoch}": train_metrics})

                eval_desc = f"Expert Eval S{stage_id}E{stage_epoch}"
                eval_metrics, factor_metrics = self._run_eval_step(
                    model,
                    eval_dl,
                    device,
                    desc=eval_desc,
                    logger=logger,
                    stage_id=stage_id,
                    stage_epoch=stage_epoch,
                    global_epoch=global_epoch,
                    xp=xp,
                    record_epoch=True,
                    disable_progress=False,
                )
                latest_eval_metrics = eval_metrics
                latest_factor_metrics = factor_metrics

                if self._should_advance_stage(stage_id, stage_epoch, factor_metrics, logger):
                    break

            eval_metrics = latest_eval_metrics or {}
            factor_metrics = latest_factor_metrics or {}

            if latest_eval_metrics is not None:
                xp.link.push_metrics({f"expert/eval/{stage_id}": latest_eval_metrics})
            if latest_factor_metrics:
                xp.link.push_metrics({f"expert/factors/{stage_id}": latest_factor_metrics})

            split_idx, stage_best = self._select_factor(factor_metrics, model, maximize=True)
            freeze_idx, freeze_stats = self._select_factor(factor_metrics, model, maximize=False)
            if freeze_idx is not None and freeze_idx != split_idx:
                model.freeze_expert(freeze_idx)
                logger.info(
                    "Frozen expert %d (factor %s) after stage %d.",
                    freeze_idx,
                    freeze_stats["name"] if freeze_stats else f"factor_{freeze_idx}",
                    stage_id,
                )

            if stage_best is not None and stage_best["f1"] > best_record["f1"]:
                best_record = {
                    "f1": stage_best["f1"],
                    "stage": stage_id,
                    "factor": stage_best["name"],
                }
                self._save_checkpoint(model, checkpoint_path, logger)

            if model.num_experts >= target_experts:
                if best_record["stage"] is None:
                    self._save_checkpoint(model, checkpoint_path, logger)
                training_active = False
                continue

            if split_idx is None:
                logger.warning(
                    "No factor metrics available to continue telescoping; stopping at %d experts.",
                    model.num_experts,
                )
                if best_record["stage"] is None:
                    self._save_checkpoint(model, checkpoint_path, logger)
                    logger.info("Saved final model despite missing factor metrics.")
                training_active = False
                continue

            logger.info(
                "Splitting expert %d (factor %s) after stage %d.",
                split_idx,
                stage_best["name"] if stage_best else f"factor_{split_idx}",
                stage_id,
            )
            model.split_expert(split_idx)
            optimizer = self._build_optimizer(model, optim_cfg)
            stage += 1

        if best_record["stage"] is None:
            logger.info("No factor improvements detected; saved final model.")
        else:
            logger.info(
                "Best stage %d with factor %s achieving F1 %.4f",
                best_record["stage"],
                best_record["factor"],
                best_record["f1"],
            )
            xp.link.push_metrics(
                {
                    "expert/best_epoch": best_record["stage"],
                    "expert/best_factor": best_record["factor"],
                    "expert/best_f1": best_record["f1"],
                }
            )

        return model


    def evaluate(self, cfg, logger, eval_dl, xp):
        device = cfg.device
        checkpoint_path = "expert_model.pth"
        state_dict, ckpt_experts = self._load_checkpoint_payload(checkpoint_path, device)
        default_target = int(cfg.train.target_experts)
        num_experts = ckpt_experts if ckpt_experts is not None else default_target
        model = self._build_model(cfg, num_experts)
        model.load_state_dict(state_dict)
        logger.info("Loaded ExpertModel checkpoint with %d experts from %s", model.num_experts, checkpoint_path)

        eval_metrics, factor_metrics = self._run_eval_step(
            model,
            eval_dl,
            device,
            desc="Expert Evaluation",
            logger=logger,
            disable_progress=False,
        )

        loss_table = build_train_table(eval_metrics, model=model, weights=self.weights)
        logger.info("\nEval loss metrics:\n%s", loss_table)

        xp.link.push_metrics({"expert/eval": eval_metrics})
        if factor_metrics:
            xp.link.push_metrics({"expert/factors": factor_metrics})
        return eval_metrics, factor_metrics


@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg):
    logger = get_logger("train_expert.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")
    logger.info(f"Exec file: {__file__}")

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No GPU available, switching to CPU")
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    trainer = ExpertTrainer(cfg)

    train_dl, eval_dl = initialize_dataloaders(cfg, logger)

    if cfg.eval.eval_only:
        trainer.evaluate(cfg, logger, eval_dl, xp)
    else:
        trainer.train(cfg, logger, train_dl, eval_dl, xp)


if __name__ == "__main__":
    main()
