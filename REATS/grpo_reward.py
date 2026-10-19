import os
import re
import logging
import numpy as np
from typing import List
from swift.plugin import ORM, orms

logger = logging.getLogger(__name__)


class ForecastMixedReward(ORM):
    """
    GRPO mixed reward: Logarithmic reward mapping.
    reward = 1 / (1 + k * delta) where delta = max(pred_mse - oracle_mse, 0)
    Log-like decay: slower than exp, faster than linear. No saturation at small delta.

    k=20: delta=0.01 -> 0.83, delta=0.03 -> 0.63, delta=0.05 -> 0.50, delta=0.1 -> 0.33
    """

    BASE_PATH = './data/checkpoints_forecast_UTS'

    W_REL_MSE = 0.8
    W_AGG = 0
    W_ORACLE = 0.2
    FORMAT_PENALTY = -0.5

    LOG_K = 20.0

    def __init__(self):
        self._pred_cache = {}
        self._gt_cache = {}
        self._oracle_cache = {}

    def _load_predictions(self, dataset, group_dir, split, model_name, in_len, pred_len):
        key = (dataset, group_dir, split, model_name, in_len, pred_len)
        if key not in self._pred_cache:
            base = os.path.join(self.BASE_PATH, dataset, group_dir)
            path = os.path.join(base, f'{split}_pred_{model_name}_{in_len}_{pred_len}.npy')
            arr = np.load(path)
            if arr.ndim == 3:
                arr = arr.squeeze(-1)
            if split == 'train':
                val_path = os.path.join(base, f'val_pred_{model_name}_{in_len}_{pred_len}.npy')
                if os.path.exists(val_path):
                    val_arr = np.load(val_path)
                    if val_arr.ndim == 3:
                        val_arr = val_arr.squeeze(-1)
                    arr = np.concatenate([arr, val_arr], axis=0)
            self._pred_cache[key] = arr
        return self._pred_cache[key]

    def _load_ground_truth(self, dataset, group_dir, split, in_len, pred_len):
        key = (dataset, group_dir, split, in_len, pred_len)
        if key not in self._gt_cache:
            base = os.path.join(self.BASE_PATH, dataset, group_dir)
            path = os.path.join(base, f'{split}_batch_y_{in_len}_{pred_len}.npy')
            arr = np.load(path)
            if arr.ndim == 3:
                arr = arr.squeeze(-1)
            if split == 'train':
                val_path = os.path.join(base, f'val_batch_y_{in_len}_{pred_len}.npy')
                if os.path.exists(val_path):
                    val_arr = np.load(val_path)
                    if val_arr.ndim == 3:
                        val_arr = val_arr.squeeze(-1)
                    arr = np.concatenate([arr, val_arr], axis=0)
            self._gt_cache[key] = arr
        return self._gt_cache[key]

    def _load_oracle_weights(self, dataset, group_dir, split, in_len, pred_len):
        key = (dataset, group_dir, split, in_len, pred_len)
        if key not in self._oracle_cache:
            base = os.path.join(self.BASE_PATH, dataset, group_dir)
            path = os.path.join(base, f'{split}_weights_{in_len}_{pred_len}.npy')
            arr = np.load(path)
            if split == 'train':
                val_path = os.path.join(base, f'val_weights_{in_len}_{pred_len}.npy')
                if os.path.exists(val_path):
                    val_arr = np.load(val_path)
                    arr = np.concatenate([arr, val_arr], axis=0)
            self._oracle_cache[key] = arr
        return self._oracle_cache[key]

    def _load_tool_preds_for_sample(self, model_names, dataset, group_dir,
                                     split, in_len, pred_len, sid):
        preds = []
        for name in model_names:
            try:
                all_preds = self._load_predictions(
                    dataset, group_dir, split, name, in_len, pred_len)
                preds.append(all_preds[sid])
            except Exception as e:
                logger.warning(f'Failed to load pred for {name}: {e}')
                return None
        return np.array(preds)

    def parse_completion(self, text):
        if not text or not text.strip():
            return [], []
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        anchor = None
        for i, ln in enumerate(lines):
            if 'Optimal ensemble weights(%)' in ln:
                anchor = i
                break
        if anchor is None or anchor + 2 >= len(lines):
            return [], []
        tool_line = lines[anchor + 1]
        model_names = [name.strip() for name in tool_line.split(',')]
        n_tools = len(model_names)
        rows = []
        for line in lines[anchor + 2:]:
            try:
                pcts = [int(x.strip()) for x in line.split(',')]
                if len(pcts) != n_tools:
                    continue
                w = []
                for x in pcts:
                    abs_digits = len(str(abs(x)))
                    if abs_digits > 2:
                        w.append(x / (10 ** abs_digits))
                    else:
                        w.append(x / 100)
                rows.append(np.array(w))
            except Exception:
                continue
        return model_names, rows

    def _normalize_group(self, g):
        g_clip = np.clip(g, 0, None)
        s = g_clip.sum()
        if s > 0:
            return g_clip / s
        return None

    def compute_format_penalty(self, groups):
        if not groups:
            return self.FORMAT_PENALTY
        violations = 0
        for g in groups:
            if np.any(g < -0.05):
                violations += 1
            if abs(g.sum() - 1.0) >= 0.2:
                violations += 1
        if violations == 0:
            return 0.0
        ratio = violations / (len(groups) * 2)
        return self.FORMAT_PENALTY * ratio

    def _log_reward(self, delta):
        """Reciprocal (log-like): 1 / (1 + k * delta)"""
        return 1.0 / (1.0 + self.LOG_K * delta)

    def compute_relative_mse_reward(self, groups, tool_preds, gt, oracle_weights):
        results = []
        for i, g in enumerate(groups):
            w = self._normalize_group(g)
            if w is None:
                continue
            pred_ens = np.dot(w, tool_preds)
            pred_mse = float(np.mean((pred_ens - gt) ** 2))
            if i < len(oracle_weights):
                ow = oracle_weights[i]
                ow_s = ow.sum()
                if ow_s > 0:
                    ow_norm = ow / ow_s
                else:
                    ow_norm = np.full(len(ow), 1.0 / len(ow))
                oracle_ens = np.dot(ow_norm, tool_preds)
                oracle_mse = float(np.mean((oracle_ens - gt) ** 2))
            else:
                oracle_mse = 0.0
            delta = max(pred_mse - oracle_mse, 0.0)
            results.append(self._log_reward(delta))
        if not results:
            return 0.0
        return sum(results) / len(results)

    def compute_aggregated_mse_reward(self, groups, tool_preds, gt, oracle_weights):
        valid_weights = []
        for g in groups:
            w = self._normalize_group(g)
            if w is not None:
                valid_weights.append(w)
        if not valid_weights:
            return 0.0
        agg_w = np.mean(valid_weights, axis=0)
        agg_s = agg_w.sum()
        if agg_s > 0:
            agg_w = agg_w / agg_s
        pred_ens = np.dot(agg_w, tool_preds)
        pred_mse = float(np.mean((pred_ens - gt) ** 2))
        if oracle_weights is not None and len(oracle_weights) > 0:
            ow = oracle_weights[0]
            ow_s = ow.sum()
            if ow_s > 0:
                ow_norm = ow / ow_s
                oracle_ens = np.dot(ow_norm, tool_preds)
                oracle_mse = float(np.mean((oracle_ens - gt) ** 2))
                delta = max(pred_mse - oracle_mse, 0.0)
                return self._log_reward(delta)
        return self._log_reward(pred_mse)

    def compute_oracle_proximity_reward(self, groups, oracle_weights):
        if not groups or oracle_weights is None:
            return 0.0
        oracle_norm = []
        for ow in oracle_weights:
            s = ow.sum()
            if s > 0:
                oracle_norm.append(ow / s)
            else:
                oracle_norm.append(ow)
        oracle_norm = np.array(oracle_norm)
        rewards = []
        for i, g in enumerate(groups):
            if i >= len(oracle_norm):
                break
            g_n = self._normalize_group(g)
            if g_n is None:
                continue
            ow = oracle_norm[i]
            weight_mse = float(np.mean((g_n - ow) ** 2))
            rewards.append(self._log_reward(weight_mse))
        if not rewards:
            return 0.0
        return float(np.mean(rewards))

    def __call__(self, completions, sample_id,
                 dataset_name, group_dir, split, input_len, pred_len, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            try:
                sid = int(sample_id[i])
                ds = str(dataset_name[i])
                gdir = str(group_dir[i])
                sp = str(split[i])
                in_len = int(input_len[i])
                p_len = int(pred_len[i])

                model_names, groups = self.parse_completion(completion)
                format_p = self.compute_format_penalty(groups)

                rel_mse_r = 0.0
                agg_r = 0.0
                oracle_r = 0.0

                if model_names and groups:
                    tool_preds = self._load_tool_preds_for_sample(
                        model_names, ds, gdir, sp, in_len, p_len, sid)
                    if tool_preds is not None:
                        gt = self._load_ground_truth(ds, gdir, sp, in_len, p_len)[sid]
                        oracle_w = None
                        try:
                            all_oracle = self._load_oracle_weights(ds, gdir, sp, in_len, p_len)
                            oracle_w = all_oracle[sid]
                        except Exception as e:
                            logger.warning(f'[comp {i}] oracle load failed: {e}')

                        if oracle_w is not None:
                            rel_mse_r = self.compute_relative_mse_reward(
                                groups, tool_preds, gt, oracle_w)
                        else:
                            abs_rewards = []
                            for g in groups:
                                w = self._normalize_group(g)
                                if w is not None:
                                    ens = np.dot(w, tool_preds)
                                    mse = float(np.mean((ens - gt) ** 2))
                                    abs_rewards.append(self._log_reward(mse))
                            rel_mse_r = np.mean(abs_rewards) if abs_rewards else 0.0

                        agg_r = self.compute_aggregated_mse_reward(
                            groups, tool_preds, gt, oracle_w)

                        if oracle_w is not None:
                            oracle_r = self.compute_oracle_proximity_reward(
                                groups, oracle_w)

                reward = (self.W_REL_MSE * rel_mse_r +
                          self.W_AGG * agg_r +
                          self.W_ORACLE * oracle_r +
                          format_p)

            except Exception as e:
                logger.warning(f'Reward computation failed for completion {i}: {e}')
                reward = 0.0

            rewards.append(reward)
        return rewards


orms['forecast_mixed'] = ForecastMixedReward
