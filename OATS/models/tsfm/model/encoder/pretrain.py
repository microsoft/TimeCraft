# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution
import math
import time
import numpy as np
import random

from tsfm.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedNLLLoss,
)
from tsfm.module.norm import RMSNorm
from tsfm.module.position import (
    LearnedEmbedding,
    LearnedProjection,
)
from tsfm.optim import SchedulerType, get_scheduler
from tsfm.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    EvalCrop_AdaLength,
    EvalPad_AdaLength,
    EvalMaskedPrediction,
    DummyValueImputation,
    ExtendMask,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SampleDimension,
    SelectFields,
    SequencifyField,
    Transformation,
)

from .module import BasicModule
from tsfm.val.metrics import (
    MSE_mean,
    MAE_mean,
    MSE_median,
    MAE_median,
    MASE,
    MAPE,
    SMAPE,
    RMSE,
    NRMSE,
    ND,
    CRPS
)



class TransformerEncoderPretrain(L.LightningModule):
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
        "label",
        "label_observed_mask",
    )
    train_seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    )
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
    }
    
    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        num_training_steps: int,
        num_warmup_steps: int,
        max_dim: int = 1,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[BasicModule] = None,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = [
            MSE_mean() ,MAE_mean(), MSE_median(), MAE_median(), MASE(), MAPE(), SMAPE(), RMSE(), NRMSE(), ND(), CRPS()
            ],
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        num_low_influence_to_remove: int = 16,
        enable_influence_scoring: bool = False,
        enable_dataset_contribution_logging: bool = False,
        enable_reweighting: bool = False,
        add_noise: bool = False,
        influence_filter_ratio: float = 0.7,
        use_cosine_similarity: bool = False,
        select_from_generated: bool = False,
        generate_after_epoch: int = 0,
        mixup: bool = False,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = BasicModule(**module_kwargs) if module is None else module
        self.influence_scores = {}
        self.recommended_weights = {}  # Initialize for recommended weights-based filtering
        self.threshold = 4000
        self.generation_model = None  # Placeholder for generation model
        self.generation_data = None  # Placeholder for generation data
        self.cache_val_batch = None  # This is used to cache the validation batch for TS influence scoring
        
    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        output = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size,
        )
        return output
    
    def infer(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        distr = self.forward(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            patch_size=patch_size,
            prediction_mask=prediction_mask,
        )
        
        preds = distr.sample(torch.Size((self.hparams.num_samples, ))) # sample batch time features
        preds = preds.transpose(0, 1) # batch sample time features
        return distr, preds

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # Determine whether to use influence-based or random filtering
        current_step = self.global_step
        # use_influence_filtering = (current_step % self.hparams.influence_filter_frequency == 0) and self.hparams.enable_influence_scoring
        if self.hparams.enable_influence_scoring:
            # set use_influence_filtering = 1 by influence_filter_ratio possibility
            use_influence_filtering = random.random() < self.hparams.influence_filter_ratio
            if self.global_step == 0:
                use_influence_filtering = True
        else:
            use_influence_filtering = False
        
        # Always apply some form of filtering (influence-based every N steps, random otherwise)
        print(f"Step {current_step}: This batch originally has {len(batch['dataset_index'])} samples")
        
        if use_influence_filtering:
            print(f"Step {current_step}: Using influence-based filtering (p={self.hparams.influence_filter_ratio} steps)")
            batch, threshold = self._filter_low_influence_samples(
                batch, 
                num_to_remove=self.hparams.num_low_influence_to_remove, 
                use_influence_scores=True
            )
            # threshold = 0  # tmp

            if self.hparams.generate_after_epoch <= current_step:
                generated_batch = self._generated_similar_samples(batch)
                if self.hparams.select_from_generated:
                    gen_grad = self._compute_per_sample_gradients_with_indices(generated_batch, generated_batch["_dataset_idx"])
                    val_gradients = self.get_validation_gradients_from_trainer(max_val_samples=32)  # changed for case-study
                    # # calculate the influence score of the generated samples
                    influence_scores_gen = self.compute_influence_scores_batched_on_generated(val_gradients, gen_grad)

                    # Select only high-scoring generated samples (above threshold)
                    keep_indices = [i for i, score in enumerate(influence_scores_gen) if score > threshold]
                    print(f"Generated {len(influence_scores_gen)} samples, keeping {len(keep_indices)} samples above threshold {threshold}")
                    
                    # Filter generated batch to keep only high-scoring samples
                    if keep_indices:
                        filtered_generated_batch = {}
                        for key, value in generated_batch.items():
                            if torch.is_tensor(value) and value.shape[0] == len(influence_scores_gen):
                                filtered_generated_batch[key] = value[keep_indices]
                            else:
                                filtered_generated_batch[key] = value
                        batch = self._merge_batches(batch, filtered_generated_batch)
                    else:
                        print("No generated samples above threshold, skipping merge")
                else:
                    # Merge generated samples directly
                    batch = self._merge_batches(batch, generated_batch)
                    print(f"Step {current_step}: Merged generated samples, new batch size is {len(batch['dataset_index'])}")
        else:
            # Determine filtering strategy based on available recommended weights
            if hasattr(self, 'recommended_weights') and self.recommended_weights:
                print(f"Step {current_step}: Using recommended weights-based filtering")
            else:
                print(f"Step {current_step}: Using random filtering (no recommended weights available yet)")
            
            batch, threshold = self._filter_low_influence_samples(
                batch, 
                num_to_remove=self.hparams.num_low_influence_to_remove, 
                use_influence_scores=False
            )

            # add noise to the batch if self.hparams.add_noise is True
            if self.hparams.add_noise:
                batch = self._add_noise_to_batch(batch)

            if self.hparams.mixup:
                batch = self._mixup_batch(batch)

            if not self.hparams.add_noise and not self.hparams.mixup:
                generated_batch = self._generated_similar_samples(batch)
                batch = self._merge_batches(batch, generated_batch)
                print(f"Step {current_step}: Merged generated samples, new batch size is {len(batch['dataset_index'])}")

        
        print(f"Step {current_step}: This batch after filtering has {len(batch['dataset_index'])} samples")

        output = self(
            **{field: batch[field] for field in list(self.train_seq_fields) + ["sample_id"]}
        )
        loss = self.hparams.loss_func(
            pred=output,
            target=batch["label"],
            observed_mask=batch["label_observed_mask"],
            prediction_mask=batch["prediction_mask"],
            sample_id=batch["sample_id"],
            variate_id=batch["variate_id"],
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"train/{self.hparams.loss_func.__class__.__name__}",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss

    def _merge_batches(self, original_batch, generated_batch):
        """Merge the generated batch with the original batch"""
        for key in original_batch.keys():
            assert key in generated_batch, f"Key {key} not found in generated batch"
            original_batch[key] = torch.cat((original_batch[key], generated_batch[key]), dim=0)
        return original_batch

    def _add_noise_to_batch(self, batch):
        """Add small noise (0.01 STD) to batch['label'] and batch['target']"""
        # noise_std = 0.03
        
        # noise = torch.randn_like(batch["label"]) * noise_std
        bs = batch["label"].shape[0]
        noise = torch.randn_like(batch["label"]) * torch.std(batch["label"], dim=(1,2)).view(bs, 1, 1) * 0.2
        
        # Use the same cached noise for both fields
        if "label" in batch:
            batch["label"] = batch["label"] + noise
            
        if "target" in batch:
            batch["target"] = batch["target"] + noise
            
        return batch

    def _mixup_batch(self, batch):
        """Apply mixup augmentation to 50% of samples in the batch.
        
        For the replacing samples, change them to be 0.5*sample1 + 0.5*sample2.
        sample1/2 are samples randomly picked within the batch.
        For label and target, use weighted sum. For other values, use the sample 
        (from 1 and 2) with longer length.
        """
        batch_size = len(batch['dataset_index'])
        if batch_size < 2:
            return batch  # Need at least 2 samples for mixup
        
        # Select 50% of samples to replace
        num_to_replace = batch_size // 2
        replace_indices = torch.randperm(batch_size)[:num_to_replace]
        
        for idx in replace_indices:
            # Randomly pick two samples from the batch (excluding the current one)
            sample_indices = torch.randperm(batch_size)
            # Ensure we don't pick the same sample twice and not the current index
            candidate_indices = [i for i in sample_indices if i != idx][:2]
            if len(candidate_indices) < 2:
                # If we can't find 2 different samples, use any 2 samples
                candidate_indices = sample_indices[:2].tolist()
            
            idx1, idx2 = candidate_indices[0], candidate_indices[1]
            
            # For numerical fields (label, target), use weighted sum (0.5 + 0.5)
            numerical_fields = ['label', 'target']
            for field in numerical_fields:
                if field in batch:
                    batch[field][idx] = 0.5 * batch[field][idx1] + 0.5 * batch[field][idx2]
            
            # For other fields, choose the sample with longer sequence length
            # We'll use the sample_id field to determine which sample has more data
            other_fields = [k for k in batch.keys() if k not in numerical_fields + ['sample_id']]
            
            # Determine which sample has longer length by comparing sample_id max values
            if 'sample_id' in batch:
                len1 = batch['sample_id'][idx1].max().item()
                len2 = batch['sample_id'][idx2].max().item()
                longer_idx = idx1 if len1 >= len2 else idx2
            else:
                # Fallback: just use the first sample
                longer_idx = idx1
            
            # Copy other fields from the sample with longer length
            for field in other_fields:
                if field in batch:
                    batch[field][idx] = batch[field][longer_idx].clone()
        
        return batch

    def _generated_similar_samples(self, batch, num_samples=16):
        print("Generating similar samples for influence scoring...")
        prompt_list = self._crop_sample_from_patches(batch)

        # load the model and datamodule for generation
        import sys
        from generation.generate_batch_preview import generate_conditional_batch, load_model
        if self.generation_model is None or self.generation_data is None:
            ckpt_path = "000060-0.0853.ckpt"
            generation_model, self.generation_data = load_model(ckpt_path, seed=0)
            object.__setattr__(self, 'generation_model', generation_model)
        
        # print("prompt list length:", len(prompt_list))
        generated_samples_list = []
        dataset_names_list = []
        subset_id_list = []
        new_prompt_list = []
        for idx in range(len(prompt_list)):
            dataset_name = self._get_dataset_name_for_index(batch["_dataset_idx"][idx].item())
            print(dataset_name,
                  prompt_list[idx].shape if prompt_list is not None else None)
            if dataset_name not in self.generation_data.key_list:
                print(f"Dataset {dataset_name} not found in generation data, skipping...")
                continue
            dataset_names_list.append(dataset_name)
            new_prompt_list.append(prompt_list[idx])

            subset_id = self.generation_data.key_list.index(dataset_name)
            subset_id_list.append(subset_id)

        generated_samples = generate_conditional_batch(
            prompt=new_prompt_list,
            subset_id=subset_id_list,
            model=self.generation_model,
            dataset_name=dataset_names_list,
            data_module=self.generation_data,
            num_samples=1,  # generation 1:1
            ddim=True,
            ddim_steps=20,
            dataset_idx=(-batch["_dataset_idx"]).tolist(),
        )
        
        # mv generated_samples to the same device as the model
        device = "cuda"
        for key, value in generated_samples.items():
            if isinstance(value, torch.Tensor):
                generated_samples[key] = value.to(device)

        return generated_samples

    def _crop_sample_from_patches(self, batch, target_length=320, patch_size=32):
        """Crop a sample from the patches in the batch to match the target length"""

        # calculate the number of patches to crop
        num_patches = target_length // patch_size

        label = batch["label"]  # [16, 512, 32]
        batch_size, total_patches, patch_dim = label.shape

        # check number of available patches using label_observed_mask
        label_observed_mask = batch["label_observed_mask"]  # [16, 512, 32]
        
        # Get the availability mask by checking if any element in the patch dimension is observed
        # This assumes that if a patch is available, all elements in that patch should be 1
        patch_available_mask = label_observed_mask.any(dim=-1)  # [16, 512] - True where patches are available

        # random sample consecutive patches from the label with length=num_patches
        if num_patches > 0:
            cropped_samples_list = []
            
            for batch_idx in range(batch_size):
                # Find available patches for this batch item
                available_patches = patch_available_mask[batch_idx]  # [512]
                num_available = available_patches.sum().item()
                
                if num_available >= num_patches:
                    # Find the range of available patches (assuming they are consecutive from start)
                    available_indices = torch.where(available_patches)[0]  # Get indices where patches are available
                    
                    if len(available_indices) >= num_patches:
                        # Check if we can find consecutive patches
                        max_start_for_consecutive = available_indices[-num_patches].item() if len(available_indices) >= num_patches else 0
                        min_start = available_indices[0].item()
                        
                        # Randomly select a starting position that ensures num_patches consecutive available patches
                        if max_start_for_consecutive >= min_start:
                            start_idx = torch.randint(min_start, max_start_for_consecutive + 1, (1,), device=label.device).item()
                        else:
                            start_idx = min_start
                        
                        # Extract consecutive patches
                        end_idx = start_idx + num_patches
                        patch_indices = torch.arange(start_idx, end_idx, device=label.device)
                        
                        # Get the patches for this batch item
                        selected_patches = label[batch_idx, patch_indices]  # [num_patches, patch_dim]
                        cropped_sample = selected_patches.reshape(-1)  # [target_length]
                    else:
                        # Not enough available patches, return zeros
                        cropped_sample = None
                else:
                    # Not enough available patches, return zeros
                    cropped_sample = None
                
                cropped_samples_list.append(cropped_sample)
            
            # Stack all batch items
            # cropped_samples = torch.stack(cropped_samples_list, dim=0)  # [batch_size, target_length]
            
            # remove all the None in the cropped_samples_list
            cropped_samples_list = [sample for sample in cropped_samples_list if sample is not None]
            return cropped_samples_list
        else:
            # If num_patches is 0 or exceeds available patches, return zeros
            return None
            
    def on_train_batch_start(self, batch, batch_idx):
        """Calculate per-example gradients for influence function computation and filter low-influence samples"""

        # Only compute per-example gradients during training
        if not self.training:
            return
        
        # Only compute influence scores when influence-based filtering will be used
        current_step = self.global_step
        # use_influence_filtering = (current_step % self.hparams.influence_filter_frequency == 0)
        if self.hparams.enable_influence_scoring:
            # set use_influence_filtering = 1 by influence_filter_ratio possibility
            use_influence_filtering = random.random() < self.hparams.influence_filter_ratio
            if self.global_step == 0:
                use_influence_filtering = True
        else:
            use_influence_filtering = False
        
        
        if not use_influence_filtering and not self.hparams.enable_dataset_contribution_logging:  # case
            print(f"Step {current_step}: Skipping influence computation (random filtering step)")
            return
            
        # Skip influence scoring if disabled
        if not self.hparams.enable_influence_scoring:
            print(f"Step {current_step}: Influence scoring disabled - skipping gradient computation")
            return
        
        # Get original dataset indices from the new field
        dataset_indices = batch.get("dataset_index", None)
        if dataset_indices is None:
            print(f"Warning: No dataset_index found in batch {batch_idx}")
            print(f"Make sure to use PadCollateWithDatasetIndex and TimeSeriesDatasetWithIndex")
            return
        
        # Get unique dataset indices (filter out padding with -1)
        unique_indices = dataset_indices.flatten().unique()
        unique_indices = unique_indices[unique_indices >= 0]  # Remove padding (-1)
        
        # print(f"Unique dataset indices in batch {batch_idx}: {unique_indices.tolist()}")
        print(f"Unique dataset indices in batch {batch_idx} length: {len(unique_indices)}")
        
        # Initialize per-example gradient storage if not exists
        if not hasattr(self, 'per_example_gradients'):
            self.per_example_gradients = {}
            
        # Initialize influence score history if not exists
        if not hasattr(self, 'influence_score_history'):
            self.influence_score_history = {}
        
        # Calculate per-sample gradients using original dataset indices
        per_sample_grads = self._compute_per_sample_gradients_with_indices(batch, unique_indices)
        
        # Store gradients indexed by original dataset index
        for i, dataset_idx in enumerate(unique_indices):
            dataset_idx_item = dataset_idx.item()
            
            # Extract gradients for this specific sample
            sample_grads = {}
            for param_name, grad_batch in per_sample_grads.items():
                if grad_batch is not None and i < len(grad_batch):
                    sample_grads[param_name] = grad_batch[i].clone().detach()
            
            # Store in per-example gradient dict
            if dataset_idx_item not in self.per_example_gradients:
                self.per_example_gradients[dataset_idx_item] = []
            
            self.per_example_gradients[dataset_idx_item].append({
                'gradients': sample_grads,
                'step': self.global_step,
                'epoch': self.current_epoch,
                'loss': None  # We don't have outputs yet in on_train_batch_start
            })

        # directly calculate the influence scores on validation gradients
        val_gradients = self.get_validation_gradients_from_trainer(max_val_samples=32)  # changed for case-study
        influence_scores = self.compute_influence_scores_batched(val_gradients)

        # update the influence scores, but append the new scores to the end, don't overwrite the existing scores
        for sample_idx, scores in influence_scores.items():
            if sample_idx in self.influence_scores:
                self.influence_scores[sample_idx].extend(scores)
            else:
                self.influence_scores[sample_idx] = scores

        # Clean up old influence scores to keep only recent 4000 steps
        self._cleanup_old_influence_scores(keep_recent_steps=4000)  # 4000 for case-study

        # Only compute and log dataset contributions if dataset contribution logging is enabled
        if self.hparams.enable_dataset_contribution_logging:
            # aggregate the influence scores by dataset name
            dataset_influence_scores = {}
            count_dataset_scores = {}
            for sample_idx, scores in self.influence_scores.items():
                for score in scores:
                    dataset_name = score.get('dataset_name', 'Unknown')
                    if dataset_name not in dataset_influence_scores:
                        dataset_influence_scores[dataset_name] = 0
                        count_dataset_scores[dataset_name] = 0
                    # use running average to aggregate the influence scores
                    dataset_influence_scores[dataset_name] = (dataset_influence_scores[dataset_name] * count_dataset_scores[dataset_name] + score['influence_score']) / (count_dataset_scores[dataset_name] + 1)
                    count_dataset_scores[dataset_name] += 1
            
            # sort the dataset_influence_scores by the score
            dataset_influence_scores = sorted(dataset_influence_scores.items(), key=lambda x: x[1], reverse=True)
            
            # print the dataset_influence_scores in order
            print("=" * 80)
            print("DATASET INFLUENCE SCORES:")
            for dataset_name, score in dataset_influence_scores:
                print(f"  {dataset_name:25} | Score: {score:10.4f} | Step: {self.global_step:6d} | Epoch: {self.current_epoch:6d} | Count: {count_dataset_scores[dataset_name]:6d}")
            print("=" * 80)
        else:
            print("Dataset contribution logging disabled (enable_dataset_contribution_logging=False)")

        if self.hparams.enable_reweighting:
            # Calculate updated sampling ratios based on dataset influence scores

            scores = np.array([score for _, score in dataset_influence_scores])
            dataset_names = [name for name, _ in dataset_influence_scores]
            
            # Method 1: Linear scaling (more conservative)
            # Normalize to [0.1, 2.0] range to avoid extreme ratios
            if scores.max() > scores.min():
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
                linear_ratios = 0.1 + 1.9 * normalized_scores  # Scale to [0.1, 2.0]
                linear_ratios = linear_ratios / linear_ratios.mean()  # Normalize so mean = 1.0
            else:
                linear_ratios = np.ones(len(scores))  # All equal if no variation
            
            # Create sampling ratio dictionary
            sampling_ratios = {}
            for i, dataset_name in enumerate(dataset_names):
                sampling_ratios[dataset_name] = {
                    'influence_score': scores[i],
                    'linear_ratio': float(linear_ratios[i]),
                    'count': count_dataset_scores[dataset_name]
                }
            
            # Store sampling ratios for potential use by external components
            self.latest_sampling_ratios = sampling_ratios
            
            # Option: Use linear ratios as the recommended sampling weights
            recommended_weights = {name: info['linear_ratio'] for name, info in sampling_ratios.items()}
            print(f"Recommended sampling weights: {recommended_weights}")
            
            # Save recommended weights as member variable for future filtering rounds
            self.recommended_weights = recommended_weights
            
            # Dataset weights calculation completed - weights not applied to maintain decoupling
        else:
            print("Dataset reweighting disabled (enable_reweighting=False)")

        # Update influence score history for future filtering
        self._update_influence_score_history(influence_scores)

        # clear the per-example gradients
        self.clear_per_example_gradients()
        
    def _filter_low_influence_samples(self, batch, num_to_remove=16, use_influence_scores=True):
        """Filter out samples from the current batch using influence scores or random selection"""

        # use_influence_scores = True, then we use influence score filtering
        # use_influence_scores = False, then we use recommended weights-based filtering or random filtering
        
        if "dataset_index" not in batch:
            print("Warning: No dataset_index found in batch, skipping filtering")
            return batch
        
        dataset_indices = batch["dataset_index"]
        batch_size, seq_len = dataset_indices.shape
        
        # Get unique dataset indices in this batch
        unique_indices = dataset_indices.flatten().unique()
        unique_indices = unique_indices[unique_indices >= 0]  # Remove padding (-1)
        
        if len(unique_indices) <= num_to_remove:
            print(f"Batch has only {len(unique_indices)} unique samples, not removing any")
            return batch
        
        threshold = 0
        if use_influence_scores:
            # Use influence-based filtering (original behavior)
            if not self.hparams.enable_influence_scoring:
                print("Warning: Influence scoring disabled but influence-based filtering requested, using random filtering instead")
                use_influence_scores = False
            else:
                # Get influence scores for samples in this batch
                sample_scores = []
                for idx in unique_indices:
                    idx_item = idx.item()
                    
                    # Get latest influence score for this sample
                    if (hasattr(self, 'influence_score_history') and 
                        idx_item in self.influence_score_history):
                        latest_score = self.influence_score_history[idx_item]
                        sample_scores.append((idx_item, latest_score))
                    else:
                        # If no history, assign neutral score (0.0)
                        sample_scores.append((idx_item, 0.0))
                        print(f"No influence score history found for sample {idx_item}")
                
                # Sort by influence score (ascending) and get the lowest scoring samples
                sample_scores.sort(key=lambda x: x[1])
                samples_to_remove = [idx for idx, score in sample_scores[:num_to_remove]]

                # save the batch["target"] of the samples that has top-3 influence scores and lowest-3 influence scores
                if self.global_step % 500 == 0:
                    import os
                    
                    # Get top-3 and lowest-3 influence score samples
                    # sample_scores is already sorted by influence score (ascending)
                    lowest_3_samples = sample_scores[:3] if len(sample_scores) >= 3 else sample_scores
                    top_3_samples = sample_scores[-3:] if len(sample_scores) >= 3 else sample_scores
                    
                    # Create directory for saving targets if it doesn't exist
                    # save_dir = os.path.join(os.getcwd(), "influence_target_analysis")
                    save_dir = "./influence_target_analysis"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Function to extract target data for specific indices
                    def extract_targets_for_indices(batch, target_indices):
                        extracted_targets = []
                        dataset_indices = batch["dataset_index"]
                        targets = batch["target"] if "target" in batch else None
                        
                        if targets is None:
                            print("Warning: No 'target' found in batch")
                            return []
                        
                        for target_idx in target_indices:
                            # Find positions in batch where this dataset index appears
                            mask = (dataset_indices == target_idx)
                            if mask.any():
                                # Extract target for this sample
                                target_data = targets[mask]
                                extracted_targets.append({
                                    'dataset_index': target_idx,
                                    'target_data': target_data.cpu(),
                                    'positions': mask.nonzero().cpu()
                                })
                        return extracted_targets
                    
                    # Extract targets for lowest-3 and top-3 samples
                    lowest_indices = [idx for idx, score in lowest_3_samples]
                    top_indices = [idx for idx, score in top_3_samples]
                    
                    lowest_targets = extract_targets_for_indices(batch, lowest_indices)
                    top_targets = extract_targets_for_indices(batch, top_indices)
                    
                    # Get dataset names for the samples
                    lowest_dataset_names = [self._get_dataset_name_for_index(idx) for idx, score in lowest_3_samples]
                    top_dataset_names = [self._get_dataset_name_for_index(idx) for idx, score in top_3_samples]
                    
                    # Save the data
                    save_data = {
                        'step': self.global_step,
                        'lowest_3_influence': {
                            'samples': lowest_3_samples,
                            'targets': lowest_targets,
                            'dataset_names': lowest_dataset_names
                        },
                        'top_3_influence': {
                            'samples': top_3_samples,
                            'targets': top_targets,
                            'dataset_names': top_dataset_names
                        }
                    }
                    
                    # Save to file
                    save_path = os.path.join(save_dir, f"influence_targets_step_{self.global_step}.pt")
                    torch.save(save_data, save_path)
                    print(f"Saved influence target analysis to {save_path}")
                    print(f"Lowest 3 influence scores: {[score for _, score in lowest_3_samples]}")
                    print(f"Top 3 influence scores: {[score for _, score in top_3_samples]}")
                
                
                print(f"Influence-based filtering: removing {num_to_remove} batch positions from low-influence samples: {samples_to_remove}")
                print(f"Average influence score: {sum(score for _, score in sample_scores) / len(sample_scores)}")
                print(f"Median influence score: {sample_scores[len(sample_scores)//2][1]}")
                threshold = sample_scores[len(sample_scores)//2][1]
                print(f"Samples to remove average influence score: {sum(score for _, score in sample_scores[:num_to_remove]) / num_to_remove}")
        
        if not use_influence_scores:
            # Use recommended weights-based filtering if available, otherwise random filtering
            if hasattr(self, 'recommended_weights') and self.recommended_weights:
                # Get dataset names and weights for samples in this batch
                sample_weights = []
                for idx in unique_indices:
                    idx_item = idx.item()
                    dataset_name = self._get_dataset_name_for_index(idx_item)
                    weight = self.recommended_weights.get(dataset_name, 1.0)  # Default to 1.0 if not found
                    sample_weights.append((idx_item, weight, dataset_name))
                
                # Use probabilistic sampling based on inverse weights to maintain diversity
                # Lower weights = higher probability of removal, but still maintains randomness
                import numpy as np
                
                # Extract weights and indices
                indices = [idx for idx, weight, dataset_name in sample_weights]
                weights = np.array([weight for idx, weight, dataset_name in sample_weights])
                dataset_names = [dataset_name for idx, weight, dataset_name in sample_weights]
                
                # Calculate removal probabilities (inverse of weights, normalized)
                # Add small epsilon to avoid division by zero
                epsilon = 1e-6
                inverse_weights = 1.0 / (weights + epsilon)
                removal_probs = inverse_weights / inverse_weights.sum()
                
                # Sample indices to remove based on probabilities
                try:
                    selected_indices = np.random.choice(
                        len(indices), 
                        size=min(num_to_remove, len(indices)), 
                        replace=False, 
                        p=removal_probs
                    )
                    samples_to_remove = [indices[i] for i in selected_indices]
                    
                    print(f"Recommended weights-based filtering: probabilistically removing {len(samples_to_remove)} batch positions")
                    for i in selected_indices:
                        idx, weight, dataset_name = indices[i], weights[i], dataset_names[i]
                        prob = removal_probs[i]
                        print(f"  Removing sample {idx} from {dataset_name} (weight: {weight:.3f}, removal_prob: {prob:.3f})")
                        
                except Exception as e:
                    print(f"Error in probabilistic sampling: {e}, falling back to random selection")
                    import random
                    samples_to_remove = random.sample(indices, min(num_to_remove, len(indices)))
            else:
                # Fallback to random filtering if no recommended weights available
                import random
                # unique_indices_list = unique_indices.cpu().numpy().tolist()
                # samples_to_remove = random.sample(unique_indices_list, min(num_to_remove, len(unique_indices_list)))
                samples_to_remove = []
                
                print(f"Random filtering (no recommended weights available): removing {len(samples_to_remove)} batch positions from randomly selected samples: {samples_to_remove}")
        
        if len(samples_to_remove) == 0:
            return batch, threshold

        # Create mask for samples to keep
        keep_mask = torch.ones(batch_size, dtype=torch.bool, device=dataset_indices.device)
        
        # Track how many batch positions we've removed
        removed_count = 0
        
        for sample_idx in samples_to_remove:
            if removed_count >= num_to_remove:
                break
                
            # Find batch positions that contain this sample
            sample_mask = (dataset_indices == sample_idx).any(dim=1)
            sample_positions = sample_mask.nonzero().squeeze(1)
            
            # Remove only one instance of this sample (or fewer if we're at the limit)
            positions_to_remove = min(len(sample_positions), num_to_remove - removed_count)
            if positions_to_remove > 0:
                keep_mask[sample_positions[:positions_to_remove]] = False
                removed_count += positions_to_remove
        
        # Filter all tensors in the batch
        filtered_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                if value.dim() > 0 and value.shape[0] == batch_size:
                    # This tensor has batch dimension, filter it
                    filtered_batch[key] = value[keep_mask]
                else:
                    # This tensor doesn't have batch dimension, keep as is
                    filtered_batch[key] = value
            else:
                # Non-tensor values, keep as is
                filtered_batch[key] = value
        
        original_batch_size = batch_size
        new_batch_size = keep_mask.sum().item()
        actual_removed = original_batch_size - new_batch_size
        
        # Determine filter type for logging
        if use_influence_scores:
            filter_type = "influence-based"
        elif hasattr(self, 'recommended_weights') and self.recommended_weights:
            filter_type = "recommended weights-based"
        else:
            filter_type = "random"
        
        print(f"Filtered batch size ({filter_type}): {original_batch_size} -> {new_batch_size} (actually removed {actual_removed} positions)")
        
        return filtered_batch, threshold
    
    def _update_influence_score_history(self, influence_scores):
        """Update the influence score history for batch filtering"""
        
        if not hasattr(self, 'influence_score_history'):
            self.influence_score_history = {}
        
        # Process each sample's influence scores
        for sample_idx, score_entries in influence_scores.items():
            if len(score_entries) > 0:
                # Use the most recent influence score for this sample
                latest_score_entry = score_entries[-1]
                influence_score = latest_score_entry['influence_score']
                
                # Store the latest influence score
                self.influence_score_history[sample_idx] = influence_score
        
        print(f"Updated influence score history for {len(influence_scores)} samples")

    def _cleanup_old_influence_scores(self, keep_recent_steps=1600):
        """Clean up old influence scores to keep only recent steps for memory management"""
        
        if not hasattr(self, 'influence_scores') or not self.influence_scores:
            return
        
        current_step = self.global_step
        cutoff_step = current_step - keep_recent_steps
        
        # Count entries before cleanup
        total_entries_before = sum(len(scores) for scores in self.influence_scores.values())
        samples_before = len(self.influence_scores)
        
        # Clean up old entries from each sample's influence scores
        samples_to_remove = []
        
        for sample_idx, score_entries in self.influence_scores.items():
            # Filter out old entries based on step
            recent_entries = [
                entry for entry in score_entries 
                if entry.get('step', 0) > cutoff_step
            ]
            
            if recent_entries:
                # Keep only recent entries
                self.influence_scores[sample_idx] = recent_entries
            else:
                # Mark sample for removal if no recent entries
                samples_to_remove.append(sample_idx)
        
        # Remove samples with no recent entries
        for sample_idx in samples_to_remove:
            del self.influence_scores[sample_idx]
        
        # Count entries after cleanup
        total_entries_after = sum(len(scores) for scores in self.influence_scores.values())
        samples_after = len(self.influence_scores)
        
        entries_removed = total_entries_before - total_entries_after
        samples_removed = samples_before - samples_after
        
        if entries_removed > 0 or samples_removed > 0:
            print(f"Cleaned up influence scores: removed {entries_removed} old entries from {samples_removed} samples")
            print(f"Kept influence scores from step {cutoff_step + 1} onwards (recent {keep_recent_steps} steps)")
            print(f"Remaining: {total_entries_after} entries across {samples_after} samples")

    def _compute_per_sample_gradients_with_indices(self, batch, unique_indices):
        """Compute per-sample gradients for samples with specific dataset indices."""
        per_sample_grads = {}
        
        # Initialize gradient storage
        for name, param in self.named_parameters():
            if param.requires_grad:
                per_sample_grads[name] = []
        
        # Get dataset indices tensor
        dataset_indices = batch["dataset_index"]
        
        # Compute gradient for each unique dataset index
        for dataset_idx in unique_indices:
            # Zero gradients
            self.zero_grad()
            
            # Create mask for this dataset index
            mask = (dataset_indices == dataset_idx)
            
            # Find positions where this dataset index appears
            batch_indices, seq_indices = torch.where(mask)
            
            if len(batch_indices) == 0:
                # No data for this index, store None
                print(f"WARNING: No data for dataset index {dataset_idx}, skipping...")
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        per_sample_grads[name].append(None)
                continue
            
            # Get unique batch indices (samples in the batch containing this dataset index)
            unique_batch_indices = batch_indices.unique()
            
            # Create a mini-batch with only the relevant samples
            single_batch = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    if value.dim() > 1:
                        # Take only the samples that contain this dataset index
                        single_batch[key] = value[unique_batch_indices]
                    else:
                        single_batch[key] = value
                else:
                    single_batch[key] = value
            
            # Forward pass for this dataset index
            try:
                output = self(**{
                    field: single_batch[field] 
                    for field in list(self.train_seq_fields) + ["sample_id"]
                    if field in single_batch
                })
                
                # Compute loss for this dataset index
                loss = self.hparams.loss_func(
                    pred=output,
                    target=single_batch.get("label"),
                    observed_mask=single_batch.get("label_observed_mask"),
                    prediction_mask=single_batch.get("prediction_mask"),
                    sample_id=single_batch.get("sample_id"),
                    variate_id=single_batch.get("variate_id"),
                )
                
                # Scale loss by the proportion of data from this dataset index
                total_elements = mask.sum().item()
                loss = loss * total_elements / mask.numel()
                
                # Backward pass
                loss.backward(retain_graph=True)
                
                # Store gradients
                for name, param in self.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        per_sample_grads[name].append(param.grad.clone().detach())
                    else:
                        per_sample_grads[name].append(None)
                    
            except Exception as e:
                print(f"Error computing gradient for dataset index {dataset_idx}: {e}")
                # Store None for this sample
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        per_sample_grads[name].append(None)
        
        # Convert lists to tensors
        for name in per_sample_grads:
            valid_grads = [g for g in per_sample_grads[name] if g is not None]
            if valid_grads:
                per_sample_grads[name] = torch.stack(valid_grads, dim=0)
            else:
                per_sample_grads[name] = None
        
        # Clear gradients
        self.zero_grad()
        
        return per_sample_grads

    def compute_influence_scores(self, val_gradients):
        """Compute influence scores by inner product with validation gradients"""
        if not hasattr(self, 'per_example_gradients'):
            print("No per-example gradients stored")
            return {}
        
        influence_scores = {}
        
        # Try to get dataset metadata for mapping global indices to dataset names
        dataset_metadata = None
        try:
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
                train_dataset = self.trainer.datamodule.train_dataset
                # Navigate to the ConcatDatasetBuilderWithGlobalIndex if it exists
                # This assumes the train_dataset was created by instantiating a config with a ConcatDatasetBuilderWithGlobalIndex
                # We need to trace back to find the dataset builder that created this dataset
                if hasattr(train_dataset, 'datasets'):  # ConcatDataset
                    # Look for global index metadata in any of the sub-datasets
                    for sub_dataset in train_dataset.datasets:
                        if hasattr(sub_dataset, 'global_offset'):
                            # This indicates we're using the enhanced datasets with global indexing
                            # We need to find the builder that created the dataset hierarchy
                            pass
        except Exception as e:
            print(f"Warning: Could not access dataset metadata for sub-dataset names: {e}")
        
        for sample_idx, grad_history in self.per_example_gradients.items():
            sample_scores = []
            
            for grad_entry in grad_history:
                train_grads = grad_entry['gradients']
                
                # Compute inner product or cosine similarity with validation gradients
                similarity_score = 0.0
                param_count = 0
                
                if self.hparams.use_cosine_similarity:
                    # Compute cosine similarity for each parameter separately and average
                    for param_name in train_grads:
                        if param_name in val_gradients and train_grads[param_name] is not None:
                            train_grad = train_grads[param_name].flatten()
                            val_grad = val_gradients[param_name].flatten()
                            
                            # Ensure same size
                            if train_grad.shape == val_grad.shape:
                                # Compute cosine similarity: cos(θ) = (A·B) / (||A|| * ||B||)
                                train_norm = torch.norm(train_grad)
                                val_norm = torch.norm(val_grad)
                                
                                if train_norm > 0 and val_norm > 0:
                                    cosine_sim = torch.dot(train_grad, val_grad) / (train_norm * val_norm)
                                    similarity_score += cosine_sim.item()
                                    param_count += 1
                else:
                    # Original dot product computation
                    for param_name in train_grads:
                        if param_name in val_gradients and train_grads[param_name] is not None:
                            train_grad = train_grads[param_name].flatten()
                            val_grad = val_gradients[param_name].flatten()
                            
                            # Ensure same size
                            if train_grad.shape == val_grad.shape:
                                similarity_score += torch.dot(train_grad, val_grad).item()
                                param_count += 1
                
                if param_count > 0:
                    score_entry = {
                        'influence_score': similarity_score,
                        'step': grad_entry['step'],
                        'epoch': grad_entry['epoch'],
                        'loss': None,  # We don't have outputs yet in on_train_batch_start
                        'global_dataset_index': sample_idx  # Store the global index
                    }
                    
                    # Try to add dataset name if we can map global index to dataset
                    if dataset_metadata and sample_idx in dataset_metadata:
                        score_entry['dataset_name'] = dataset_metadata[sample_idx]
                    else:
                        # Fallback: try to map using a simpler approach
                        score_entry['dataset_name'] = self._get_dataset_name_for_index(sample_idx)
                    
                    sample_scores.append(score_entry)
            
            influence_scores[sample_idx] = sample_scores
        
        return influence_scores
    
    def compute_influence_scores_batched(self, val_gradients):
        """Compute influence scores using batched operations for major speedup"""
        if not hasattr(self, 'per_example_gradients'):
            print("No per-example gradients stored")
            return {}
        
        if not val_gradients:
            print("No validation gradients provided")
            return {}
        
        # Pre-process validation gradients - get consistent parameter order
        param_names = sorted([n for n in val_gradients.keys() if val_gradients[n] is not None])
        if not param_names:
            print("No valid validation gradients found")
            return {}
        
        val_grad_flat = torch.cat([val_gradients[name].flatten() for name in param_names])
        device = val_grad_flat.device
        
        # Collect all training gradients into batches
        all_sample_indices = []
        all_train_grads = []
        all_metadata = []
        
        for sample_idx, grad_history in self.per_example_gradients.items():
            for grad_entry in grad_history:
                train_grads = grad_entry['gradients']
                
                # Check if all required gradients exist and are valid
                valid_grads = []
                valid = True
                
                for name in param_names:
                    if name in train_grads and train_grads[name] is not None:
                        valid_grads.append(train_grads[name].flatten())
                    else:
                        valid = False
                        break
                
                if valid and len(valid_grads) == len(param_names):
                    try:
                        train_grad_flat = torch.cat(valid_grads)
                        # Ensure same device
                        train_grad_flat = train_grad_flat.to(device)
                        
                        all_sample_indices.append(sample_idx)
                        all_train_grads.append(train_grad_flat)
                        all_metadata.append(grad_entry)
                    except Exception as e:
                        print(f"Warning: Could not process gradients for sample {sample_idx}: {e}")
                        continue
        
        if not all_train_grads:
            print("No valid training gradients found for batch processing")
            return {}
        
        print(f"Batch processing influence scores for {len(all_train_grads)} gradient entries...")
        
        try:
            # Stack into matrix: [num_samples, num_params]
            train_grad_matrix = torch.stack(all_train_grads)
            
            if self.hparams.use_cosine_similarity:
                # Normalize training gradients (each row)
                train_grad_norms = torch.norm(train_grad_matrix, dim=1, keepdim=True)
                # Avoid division by zero
                train_grad_norms = torch.clamp(train_grad_norms, min=1e-8)
                train_grad_matrix_normalized = train_grad_matrix / train_grad_norms
                
                # Normalize validation gradients
                val_grad_norm = torch.norm(val_grad_flat)
                val_grad_norm = torch.clamp(val_grad_norm, min=1e-8)
                val_grad_flat_normalized = val_grad_flat / val_grad_norm
                
                # Compute cosine similarity: [num_samples]
                influence_scores_flat = torch.matmul(train_grad_matrix_normalized, val_grad_flat_normalized)
            else:
                # Original dot product computation: [num_samples]
                influence_scores_flat = torch.matmul(train_grad_matrix, val_grad_flat)
            
            # Process results back into the expected format
            influence_scores = {}
            for i, (sample_idx, metadata) in enumerate(zip(all_sample_indices, all_metadata)):
                score_entry = {
                    'influence_score': influence_scores_flat[i].item(),
                    'step': metadata['step'],
                    'epoch': metadata['epoch'],
                    'loss': metadata.get('loss', None),
                    'global_dataset_index': sample_idx
                }
                
                # Try to add dataset name
                try:
                    score_entry['dataset_name'] = self._get_dataset_name_for_index(sample_idx)
                except Exception as e:
                    score_entry['dataset_name'] = f"Unknown_Idx_{sample_idx}"
                
                if sample_idx not in influence_scores:
                    influence_scores[sample_idx] = []
                influence_scores[sample_idx].append(score_entry)
            
            print(f"Successfully computed influence scores for {len(influence_scores)} unique samples")
            return influence_scores
            
        except Exception as e:
            print(f"Error in batched influence computation: {e}")
            print("Falling back to original method...")
            return self.compute_influence_scores(val_gradients)
    
    def compute_influence_scores_batched_on_generated(self, val_gradients, gen_grad):
        """Compute influence scores for generated samples using batched operations"""
        if not val_gradients:
            print("No validation gradients provided")
            return {}
        
        if not gen_grad:
            print("No generated gradients provided")
            return {}
        
        # Pre-process validation gradients - get consistent parameter order
        param_names = sorted([n for n in val_gradients.keys() if val_gradients[n] is not None])
        if not param_names:
            print("No valid validation gradients found")
            return {}
        
        val_grad_flat = torch.cat([val_gradients[name].flatten() for name in param_names])
        device = val_grad_flat.device
        
        # Collect generated gradients into batches
        all_train_grads = []
        
        # Process generated gradients
        for name in param_names:
            if name in gen_grad and gen_grad[name] is not None:
                # gen_grad[name] should be [num_generated_samples, param_shape...]
                gen_grad_param = gen_grad[name]
                if gen_grad_param.dim() > 1:
                    # Flatten each sample's gradient for this parameter
                    flattened_grads = gen_grad_param.view(gen_grad_param.shape[0], -1)
                    if len(all_train_grads) == 0:
                        # Initialize with the first parameter's gradients
                        all_train_grads = [[] for _ in range(gen_grad_param.shape[0])]
                    
                    for i in range(gen_grad_param.shape[0]):
                        all_train_grads[i].append(flattened_grads[i])
                else:
                    print(f"Warning: Unexpected gradient shape for {name}: {gen_grad_param.shape}")
                    continue
            else:
                print(f"Warning: Parameter {name} not found in generated gradients")
                return {}
        
        if not all_train_grads:
            print("No valid generated gradients found for batch processing")
            return {}
        
        # Concatenate gradients for each sample
        try:
            processed_grads = []
            for sample_grads in all_train_grads:
                if len(sample_grads) == len(param_names):
                    sample_grad_flat = torch.cat(sample_grads)
                    sample_grad_flat = sample_grad_flat.to(device)
                    processed_grads.append(sample_grad_flat)
            
            if not processed_grads:
                print("No valid processed gradients found")
                return {}
            
            print(f"Batch processing influence scores for {len(processed_grads)} generated samples...")
            
            # Stack into matrix: [num_generated_samples, num_params]
            train_grad_matrix = torch.stack(processed_grads)
            
            # Original dot product computation: [num_generated_samples]
            influence_scores_flat = torch.matmul(train_grad_matrix, val_grad_flat)
            
            # Return the influence scores as a simple list or tensor
            influence_scores_list = influence_scores_flat.detach().cpu().numpy().tolist()
            
            print(f"Successfully computed influence scores for {len(influence_scores_list)} generated samples")
            print(f"Generated samples influence scores: {influence_scores_list}")
            
            return influence_scores_list
            
        except Exception as e:
            print(f"Error in generated samples influence computation: {e}")
            return {}
    
    def _get_dataset_name_for_index(self, global_idx: int) -> str:
        """Helper method to get dataset name for a global index."""
        try:
            # BOUNDS CHECK: Validate global_idx is within reasonable range
            dataset_size = len(self.trainer.datamodule.train_dataset)  # Reasonable upper bound
            if global_idx > dataset_size:
                print(f"WARNING: Global index {global_idx} exceeds reasonable bounds (>{dataset_size}). This might indicate stale influence scores.")
                return f"OutOfBounds_Idx_{global_idx}"
            
            # Method 1: Try to use ConcatDatasetBuilderWithGlobalIndex metadata (preferred)
            if (hasattr(self.trainer, 'datamodule') and 
                hasattr(self.trainer.datamodule, 'data_builder') and
                hasattr(self.trainer.datamodule.data_builder, 'get_dataset_name_for_global_index')):
                
                # Additional bounds check using actual dataset size
                if hasattr(self.trainer.datamodule, 'train_dataset'):
                    dataset_size = len(self.trainer.datamodule.train_dataset)
                    if global_idx >= dataset_size:
                        print(f"WARNING: Global index {global_idx} >= dataset size {dataset_size}. Clearing stale influence scores.")
                        # Clear stale per_example_gradients to prevent future issues
                        if hasattr(self, 'per_example_gradients'):
                            # Remove any indices beyond the current dataset size
                            stale_indices = [idx for idx in self.per_example_gradients.keys() if idx >= dataset_size]
                            for idx in stale_indices:
                                del self.per_example_gradients[idx]
                            if stale_indices:
                                print(f"Cleared {len(stale_indices)} stale influence scores with indices: {stale_indices[:10]}{'...' if len(stale_indices) > 10 else ''}")
                        return f"Stale_Idx_{global_idx}_Cleared"
                
                return self.trainer.datamodule.data_builder.get_dataset_name_for_global_index(global_idx)
            
            # Method 2: Fallback to manual traversal of ConcatDataset
            elif hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
                train_dataset = self.trainer.datamodule.train_dataset
                
                # Bounds check against actual dataset
                if global_idx >= len(train_dataset):
                    print(f"WARNING: Global index {global_idx} >= actual dataset size {len(train_dataset)}")
                    return f"OutOfRange_Idx_{global_idx}"
                
                # Check if it's a ConcatDataset with sub-datasets that have global_offset
                if hasattr(train_dataset, 'datasets'):
                    cumulative_size = 0
                    for i, sub_dataset in enumerate(train_dataset.datasets):
                        dataset_size = len(sub_dataset)
                        if global_idx < cumulative_size + dataset_size:
                            # This global index belongs to this sub-dataset
                            # Try to get the dataset name from various sources
                            
                            # Method 2a: Check if dataset has indexer with dataset info
                            if hasattr(sub_dataset, 'indexer') and hasattr(sub_dataset.indexer, 'dataset'):
                                if hasattr(sub_dataset.indexer.dataset, 'info') and hasattr(sub_dataset.indexer.dataset.info, 'dataset_name'):
                                    return sub_dataset.indexer.dataset.info.dataset_name
                            
                            # Method 2b: Check if dataset itself has info
                            if hasattr(sub_dataset, 'info') and hasattr(sub_dataset.info, 'dataset_name'):
                                return sub_dataset.info.dataset_name
                            
                            # Method 2c: Return a descriptive name based on position
                            return f"SubDataset_{i}"
                        
                        cumulative_size += dataset_size
            
            # Final fallback
            return f"Dataset_GlobalIdx_{global_idx}"
            
        except Exception as e:
            return f"Unknown_Idx_{global_idx}_Error_{str(e)[:30]}"

    def get_validation_gradients_from_trainer(self, max_val_samples=32):  # None
        """Compute gradients using trainer's validation dataloader
        
        Args:
            max_val_samples: Maximum number of validation samples to use. If None, uses all samples.
        """
        
        if not hasattr(self.trainer, 'val_dataloaders') or not self.trainer.val_dataloaders:
            print("Warning: No validation dataloader found in trainer")
            return {}
        
        # Get validation dataloader from trainer
        val_dataloader = self.trainer.val_dataloaders[0]  # Use first validation dataloader
        
        # Check sampler for shuffle as well
        if hasattr(val_dataloader, 'sampler') and hasattr(val_dataloader.sampler, 'shuffle'):
            if val_dataloader.sampler.shuffle:
                print("WARNING: Validation dataloader sampler has shuffle=True!")
                print("This may still cause non-deterministic ordering even if shuffle=false in config.")
            print("Val dataloader sampler shuffle: ", val_dataloader.sampler.shuffle)

        return self.get_validation_gradients(val_dataloader, max_val_samples)

    def get_validation_gradients(self, dataloader_or_dataset, max_val_samples=None):
        """Compute gradients on validation dataset for influence computation
        
        Args:
            dataloader_or_dataset: Validation dataloader or dataset
            max_val_samples: Maximum number of validation samples to use. If None, uses all samples.
        """
        
        # Save current training state
        was_training = self.training
        self.eval()
        self.zero_grad()
        
        # Handle both dataset and dataloader inputs
        val_dataloader = dataloader_or_dataset
        
        total_samples = 0
        samples_processed = 0
        
        print(f"Computing validation gradients over {len(val_dataloader)} batches...")
        if max_val_samples is not None:
            print(f"Limiting to maximum {max_val_samples} validation samples")
        print(f"NOTE: Validation dataset indices will NOT interfere with training influence scores")
        
        # Collect all validation data first for single backward pass
        all_batches = []
        
        for batch_idx, val_batch in enumerate(val_dataloader):
            try:
                # IMPORTANT: Remove dataset_index from validation batch to prevent contamination
                if self.cache_val_batch is not None:
                    val_batch_clean = self.cache_val_batch
                    print("Using cached val batch --------------------------------")
                else:
                    val_batch_clean = {k: v for k, v in val_batch.items() if k != 'dataset_index'}
                    self.cache_val_batch = val_batch_clean
                # Move tensors to device
                val_batch_clean = {
                    k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in val_batch_clean.items()
                }
                
                # Get batch size for this batch
                batch_size = (
                    val_batch_clean["sample_id"].max(dim=1).values.sum().item() 
                    if "sample_id" in val_batch_clean and val_batch_clean["sample_id"].dim() > 1
                    else val_batch_clean["target"].shape[0]
                )
                
                # Check if we've reached the limit
                if max_val_samples is not None and samples_processed + batch_size > max_val_samples:
                    # Only take what we need from this batch
                    remaining_samples = max_val_samples - samples_processed
                    if remaining_samples <= 0:
                        break
                    
                    # Truncate the batch to only take remaining_samples
                    for key in val_batch_clean:
                        if torch.is_tensor(val_batch_clean[key]) and val_batch_clean[key].dim() > 0:
                            val_batch_clean[key] = val_batch_clean[key][:remaining_samples]
                    
                    batch_size = remaining_samples
                
                # Store the batch for later processing
                all_batches.append(val_batch_clean)
                
                total_samples += batch_size
                samples_processed += batch_size
                
                # Clear intermediate variables to free memory
                del val_batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(val_dataloader)} validation batches")
                
                # Check if we've reached the limit
                if max_val_samples is not None and samples_processed >= max_val_samples:
                    print(f"Reached maximum validation samples limit: {max_val_samples}")
                    break
                    
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
        
        if total_samples == 0:
            print("Warning: No validation samples processed")
            return {}
        
        print(f"Collected {total_samples} validation samples, computing gradients...")
        
        # Now compute the total loss and do a single backward pass
        try:
            # Compute total loss across all collected data with gradients enabled
            total_loss = 0.0
            num_batches = len(all_batches)
            
            for i, val_batch_clean in enumerate(all_batches):
                # Forward pass with gradient computation enabled
                with torch.enable_grad():
                    output = self(**{
                        field: val_batch_clean[field] 
                        for field in list(self.train_seq_fields) + ["sample_id"]
                        if field in val_batch_clean
                    })
                    
                    # Compute loss for this batch
                    batch_loss = self.hparams.loss_func(
                        pred=output,
                        target=val_batch_clean["label"],
                        observed_mask=val_batch_clean["label_observed_mask"],
                        prediction_mask=val_batch_clean["prediction_mask"],
                        sample_id=val_batch_clean["sample_id"],
                        variate_id=val_batch_clean["variate_id"],
                    )
                    
                    # Add to total loss
                    total_loss += batch_loss
            
            # Average the loss
            total_loss = total_loss / num_batches
            
            # Single backward pass
            total_loss.backward()
            
            # Extract gradients
            val_gradients = {}
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    val_gradients[name] = param.grad.clone().detach()
            
            # Clear gradients and restore training state
            self.zero_grad()
            if was_training:
                self.train()
            
            print(f"Computed validation gradients for {len(val_gradients)} parameters over {total_samples} samples")
            
            return val_gradients
            
        except Exception as e:
            print(f"Error in single backward pass: {e}")
            print("Falling back to original method...")
            
            # Fallback: Clear gradients and restore training state
            self.zero_grad()
            if was_training:
                self.train()
            
            # Return empty dict to indicate failure
            return {}


    def clear_per_example_gradients(self, keep_recent_steps=None):
        """Clear old per-example gradients to manage memory"""
        if not hasattr(self, 'per_example_gradients'):
            return
        
        self.per_example_gradients = {}

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        distr, preds = self.infer(
            **{field: batch[field] for field in list(self.train_seq_fields) + ["sample_id"]}
        )
        val_loss = self.hparams.loss_func(
            pred=distr,
            target=batch["label"],
            observed_mask=batch["label_observed_mask"],
            **{
                field: batch[field]
                for field in [
                    "prediction_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"val/{self.hparams.loss_func.__class__.__name__}",
            val_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
            add_dataloader_idx=True,
        )

        if self.hparams.val_metric is not None:
            val_metrics = (
                self.hparams.val_metric
                if isinstance(self.hparams.val_metric, list)
                else [self.hparams.val_metric]
            )
            for metric_func in val_metrics:
                metric = metric_func(
                    pred=preds,
                    target=batch["label"],
                    observed_mask=batch["label_observed_mask"],
                    **{
                        field: batch[field]
                        for field in [
                            "prediction_mask",
                            "sample_id",
                            "variate_id",
                        ]
                    },
                )

                self.log(
                    f"val/{metric_func.__class__.__name__}",
                    metric,
                    on_step=self.hparams.log_on_step,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    rank_zero_only=True,
                    add_dataloader_idx=True,
                )

        return val_loss
    
    def configure_optimizers(self) -> dict:
        decay = set()
        no_decay = set()

        whitelist_params = (
            LearnedProjection,
            nn.Linear,
        )
        blacklist_params = (
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }
        
    @property
    def train_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        def default_train_transform():
            return (
                SampleDimension(
                    max_dim=self.hparams.max_dim,
                    fields=("target",),
                    optional_fields=(),
                )
                + GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_size=self.module.patch_size,
                    patch_size_constraints=None,
                    offset=True,
                )
                + PatchCrop(
                    min_time_patches=self.hparams.min_patches,
                    max_patches=self.module.max_seq_len,
                    will_flatten=True,
                    offset=True,
                    fields=("target",),
                    optional_fields=(),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                    feat=False,
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=(),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=(),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=self.module.patch_size,
                    fields=("target", "observed_mask"),
                    optional_fields=(),
                )
                + MaskedPrediction(
                    min_mask_ratio=self.hparams.min_mask_ratio,
                    max_mask_ratio=self.hparams.max_mask_ratio,
                    target_field="target",
                    truncate_fields=(),
                    optional_truncate_fields=(),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=(),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=False,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=(),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackCollection(
                    field="label_observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="label",
                    fields=("label",),
                    optional_fields=(),
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=(),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields) + ["_dataset_idx"])
            )

        return defaultdict(lambda: default_train_transform)
    
    @property
    def val_transform_map(
        self,
    ) -> dict[str | type, Callable[..., Transformation]]:
        def default_val_transform(
            offset: int,
            distance: int,
            prediction_length: int,
            context_length: int,
            patch_size: int,
        ):
            return (
                SampleDimension(
                    max_dim=1,
                    fields=("target",),
                    optional_fields=(),
                )
                + GetPatchSize(
                    min_time_patches=2,
                    target_field="target",
                    patch_size=self.module.patch_size,
                    patch_size_constraints=None,
                    offset=True,
                )
                + EvalCrop_AdaLength(
                    offset,
                    distance,
                    prediction_length,
                    context_length,
                    fields=("target",),
                    optional_fields=(),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                )
                + EvalPad_AdaLength(
                    prediction_length=prediction_length,
                    context_length=context_length,
                    patch_size=self.module.patch_size,
                    fields=("target",),
                    optional_fields=()
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=(),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=(),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=self.module.patch_size,
                    fields=("target", "observed_mask"),
                    optional_fields=(),
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=(),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=False,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=(),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + EvalMaskedPrediction(
                    mask_length=math.ceil(prediction_length / patch_size),
                    target_field="target",
                    truncate_fields=(),
                    optional_truncate_fields=(),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + ExtendMask(
                    fields=tuple(),
                    optional_fields=(),
                    mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=(),
                    feat=True,
                )
                + FlatPackCollection(
                    field="label_observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="label",
                    fields=("label",),
                    optional_fields=(),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields) + ["_dataset_idx"])
            )

        return defaultdict(lambda: default_val_transform)