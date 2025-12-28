# -*- coding: utf-8 -*-
"""
Patched FSDP checkpoint manager.

Problem:
- transformers.dynamic_module_utils.custom_object_save may mutate the live model config in-place
  (e.g., injecting config.auto_map[None] = "..."), which breaks subsequent config.save_pretrained()
  because json.dumps(sort_keys=True) cannot compare NoneType and str keys.

Fix:
- Never pass the live config object into custom_object_save.
- Save HF config using a deep-copied config object.
- Remove auto_map[None] defensively before/after saving, to keep the live config clean.
- Provide a JSON-safe fallback when save_pretrained fails due to non-string dict keys.

This class overrides only save_checkpoint(); load_checkpoint() follows the upstream behavior.
"""

import json
import logging
import os
import warnings
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch.distributed.fsdp import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from transformers import GenerationConfig
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank

# Upstream imports (do not modify upstream source)
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager, FSDPConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _drop_none_key_in_auto_map(cfg) -> None:
    """Remove cfg.auto_map[None] if present."""
    if hasattr(cfg, "auto_map") and isinstance(getattr(cfg, "auto_map"), dict):
        cfg.auto_map.pop(None, None)


def _sanitize_dict_keys(obj: Any):
    """Recursively sanitize dict keys for JSON: drop None keys; cast non-str keys to str."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k is None:
                continue
            if not isinstance(k, str):
                k = str(k)
            out[k] = _sanitize_dict_keys(v)
        return out
    if isinstance(obj, list):
        return [_sanitize_dict_keys(x) for x in obj]
    if isinstance(obj, tuple):
        return [_sanitize_dict_keys(x) for x in obj]
    return obj


class PatchedFSDPCheckpointManager(FSDPCheckpointManager):
    """Drop-in replacement of FSDPCheckpointManager with a safe HF-config save path."""

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        if local_path is None:
            return

        self.previous_global_step = global_step

        # Checkpoint rotation (same policy as upstream)
        if (
            self.rank == 0
            and max_ckpt_to_keep
            and isinstance(max_ckpt_to_keep, int)
            and max_ckpt_to_keep > 0
            and len(self.previous_saved_paths) >= max_ckpt_to_keep
        ):
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        if self.should_save_model:
            assert self.model is not None, "model must be provided when saving checkpoint"
        if self.should_save_optimizer:
            assert self.optimizer is not None, "optimizer must be provided when saving checkpoint"

        # Per-rank sharded save (model/optim/extra), identical to upstream behavior
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.should_save_model:
                    torch.save(self.model.state_dict(), model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer:
                    torch.save(self.optimizer.state_dict(), optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        # Rank-0 HF artifacts save (config/tokenizer + remote-code files)
        if self.rank == 0:
            unwrap_model = self.model._fsdp_wrapped_module if fsdp_version(self.model) == 1 else self.model

            hf_dir = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_dir)

            model_config_live = unwrap_model.config

            # Keep live config clean
            _drop_none_key_in_auto_map(model_config_live)

            # Best-effort generation config
            if unwrap_model.can_generate() and getattr(model_config_live, "name_or_path", None):
                try:
                    generation_config = GenerationConfig.from_pretrained(model_config_live.name_or_path)
                    generation_config.save_pretrained(hf_dir)
                except Exception:
                    pass

            # Save config using a copy to avoid in-place mutation during serialization
            cfg_to_save = deepcopy(model_config_live)
            _drop_none_key_in_auto_map(cfg_to_save)

            try:
                cfg_to_save.save_pretrained(hf_dir)
            except TypeError as e:
                cfg_dict = _sanitize_dict_keys(cfg_to_save.to_dict())
                with open(os.path.join(hf_dir, "config.json"), "w") as f:
                    json.dump(cfg_dict, f, indent=2, sort_keys=True)
                log_with_rank(
                    f"[PatchedFSDPCheckpointManager] save_pretrained failed ({e}); wrote sanitized config.json",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )

            # Save tokenizer / processor artifacts
            self.processing_class.save_pretrained(hf_dir)
            log_with_rank(
                f"Saved model config and tokenizer/processor to {os.path.abspath(hf_dir)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # Save remote-code files using a copy of config to avoid mutating unwrap_model.config
            if hasattr(model_config_live, "auto_map"):
                cfg_for_custom = deepcopy(model_config_live)
                _drop_none_key_in_auto_map(cfg_for_custom)
                try:
                    custom_object_save(unwrap_model, hf_dir, config=cfg_for_custom)
                finally:
                    _drop_none_key_in_auto_map(model_config_live)

            # Save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_cfg = FSDPConfig(FSDP_version=fsdp_version(self.model), world_size=self.world_size)
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_cfg), f, indent=4)

        torch.distributed.barrier()

        # Optional full HF model save (kept consistent with upstream)
        if self.should_save_hf_model:
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_dir = os.path.join(local_path, "huggingface")
                os.makedirs(hf_dir, exist_ok=True)

                unwrap_model = self.model._fsdp_wrapped_module if fsdp_version(self.model) == 1 else self.model
                model_config_live = unwrap_model.config
                _drop_none_key_in_auto_map(model_config_live)

                arch0 = model_config_live.architectures[0]
                if "ForTokenClassification" in arch0:
                    from transformers import AutoModelForTokenClassification as auto_model_cls
                elif "ForCausalLM" in arch0:
                    from transformers import AutoModelForCausalLM as auto_model_cls
                elif "ForConditionalGeneration" in arch0:
                    from transformers import AutoModelForVision2Seq as auto_model_cls
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config_live.architectures}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config_live, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                save_model.save_pretrained(hf_dir, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_dir)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )

                del state_dict
                del save_model

            torch.distributed.barrier()

        self.previous_saved_paths.append(local_path)
