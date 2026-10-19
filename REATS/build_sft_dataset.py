
import os
import sys
import json
import random
import argparse
import numpy as np
from glob import glob
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from time_analysis_generate import generate_analysis


DATA_PATH = './data/checkpoints_forecast_UTS'
OUTPUT_BASE = './datasets'
SEQ_LEN = 96
PRED_LEN = 96

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather', 'electricity', 'traffic']

DATASET_DESCRIPTIONS = {
    "ETTh1": "ETTh1 (hourly electricity transformer temperature of 7 features)",
    "ETTh2": "ETTh2 (hourly electricity transformer temperature of 7 features)",
    "ETTm1": "ETTm1 (15-minute electricity transformer temperature of 7 features)",
    "ETTm2": "ETTm2 (15-minute electricity transformer temperature of 7 features)",
    "electricity": "Electricity (15-minute electricity consumption of 321 clients)",
    "exchange_rate": "Exchange (daily exchange rates of 8 countries)",
    "weather": "Weather (10-minute weather observations of 21 meteorological indicators)",
    "traffic": "Traffic (hourly road occupancy rates from 862 sensors)",
}

TOOLS = {
    "TimeXer": "TimeXer is a decoder-only Transformer that uses patch tokenization and causal masking to capture intra- and inter-series dependencies for multivariate forecasting.",
    "CARD": "CARD is a channel-dependent Transformer that models cross-channel correlations and multi-scale temporal patterns with channel-aligned attention and token blending.",
    "TimeMixer": "TimeMixer is a multiscale mixing model that decomposes series into trend and seasonal components at multiple scales and ensembles predictions across scales.",
    "CATS": "CATS is a cross-attention-only Transformer that uses future horizons as learnable queries and past observations as keys/values with query-adaptive masking.",
    "ModernTCN": "ModernTCN is a convolutional model using large-kernel depthwise convolutions on patches to capture temporal and cross-variable dependencies without attention.",
    "DLinear": "DLinear is a simple linear model that decomposes series into trend and seasonal components and applies linear layers for direct multi-step prediction.",
    "PDF": "PDF uses FFT-based periodicity analysis to decouple multi-periodic patterns and models them with parallel Transformer and convolutional branches.",
    "PatchTST": "PatchTST segments series into patches and processes each channel independently with shared Transformer weights to capture local semantics for forecasting.",
    "LSINet": "LSINet is a lightweight linear model that learns sparse temporal connections via a multihead Bernoulli-based interaction mechanism with shared matrices across variables.",
    "SEMixer": "SEMixer is an MLP-Mixer-based multiscale model that uses random attention for patch interaction and progressively mixes adjacent scales from fine to coarse.",
    "MLF": "MLF is a multi-period framework that uses adaptive patching across periods, inter-period redundancy filtering, and learnable weighted fusion for financial forecasting.",
    "MOMENT": "MOMENT is a pre-trained time series foundation model that uses masked patching for self-supervised learning and generalizes to forecasting via zero-shot or fine-tuning.",
    "Sundial": "Sundial is a universal time series foundation model based on a generative diffusion framework that produces probabilistic forecasts via iterative denoising.",
    "Timer": "Timer is a GPT-style generative Transformer pre-trained on large-scale time series corpora, using next-token prediction with patched tokenization for unified forecasting.",
    "TimeMoE": "TimeMoE is a mixture-of-experts foundation model for time series that uses sparse expert routing to scale capacity while maintaining efficient inference.",
    "TimesFM": "TimesFM is a decoder-only foundation model for time series forecasting pre-trained on a large corpus of real-world and synthetic data with patched input tokens.",
    "MOIRAI": "MOIRAI is a masked-encoder foundation model that uses any-variate attention and multiple input/output projection layers to handle diverse frequencies and variable counts.",
    "TimerXL": "TimerXL is a decoder-only causal Transformer that extends next-token prediction to multivariate settings with TimeAttention for fine-grained intra- and inter-series dependencies across flattened patch tokens.",
    "Chronos_small": "Chronos_small is a T5-based foundation model that tokenizes time series via scaling and quantization into a fixed vocabulary, trained with cross-entropy loss to produce probabilistic forecasts through token sampling.",
}

SYSTEM_PROMPT = (
    "You are a time series ensemble expert. "
    "Generate a weight table (N rows × M tools) for ensemble. "
    "Each row: all weights non-negative and sum to 1."
)

TOOL_TRAITS = {
    "TimeXer":   {"cat": "transformer", "strengths": ["exogenous variable fusion", "causal temporal patterns"],
                  "good_for": ["multi-variable", "cross-channel"], "weak_for": ["simple trend", "univariate"]},
    "CARD":      {"cat": "transformer", "strengths": ["cross-channel correlation", "multi-scale temporal modeling"],
                  "good_for": ["multi-variable", "complex patterns"], "weak_for": ["simple linear"]},
    "TimeMixer": {"cat": "decomposition", "strengths": ["multi-scale decomposition", "trend-seasonal separation"],
                  "good_for": ["seasonality", "multi-scale"], "weak_for": ["pure noise", "no periodicity"]},
    "CATS":      {"cat": "transformer", "strengths": ["horizon-specific query attention", "long-range dependency"],
                  "good_for": ["long dependency", "complex horizon"], "weak_for": ["simple short-term"]},
    "ModernTCN": {"cat": "cnn", "strengths": ["local temporal convolution", "cross-variable mixing"],
                  "good_for": ["local pattern", "multi-variable"], "weak_for": ["long-range", "strong trend"]},
    "DLinear":   {"cat": "linear", "strengths": ["trend-seasonality decomposition", "linear projection"],
                  "good_for": ["strong trend", "low noise", "linear"], "weak_for": ["nonlinear", "high noise"]},
    "PDF":       {"cat": "fft", "strengths": ["FFT periodicity decoupling", "parallel multi-period modeling"],
                  "good_for": ["seasonality", "multi-period", "periodic"], "weak_for": ["no periodicity", "pure trend"]},
    "PatchTST":  {"cat": "transformer", "strengths": ["patch-based local semantics", "channel-independent modeling"],
                  "good_for": ["local patch", "channel-independent"], "weak_for": ["cross-channel dependency"]},
    "LSINet":    {"cat": "linear", "strengths": ["sparse temporal interaction", "lightweight linear mapping"],
                  "good_for": ["strong autocorrelation", "low noise", "sparse dependency"], "weak_for": ["complex nonlinear"]},
    "SEMixer":   {"cat": "mlp", "strengths": ["multiscale patch mixing", "coarse-to-fine aggregation"],
                  "good_for": ["multiscale", "moderate complexity"], "weak_for": ["simple linear"]},
    "MLF":       {"cat": "multi-period", "strengths": ["adaptive multi-period patching", "redundancy scaling"],
                  "good_for": ["multi-period", "financial"], "weak_for": ["non-periodic"]},
    "MOMENT":    {"cat": "foundation", "strengths": ["pre-trained general representation", "zero-shot transfer"],
                  "good_for": ["diverse pattern", "zero-shot"], "weak_for": ["domain-specific optimization"]},
    "Sundial":   {"cat": "foundation", "strengths": ["diffusion-based probabilistic generation", "uncertainty modeling"],
                  "good_for": ["noisy series", "uncertainty"], "weak_for": ["clean deterministic"]},
    "Timer":     {"cat": "foundation", "strengths": ["autoregressive token generation", "pre-trained causal knowledge"],
                  "good_for": ["sequential generation", "causal"], "weak_for": ["non-causal patterns"]},
    "TimeMoE":   {"cat": "foundation", "strengths": ["sparse expert routing", "diverse pattern specialization"],
                  "good_for": ["heterogeneous pattern", "mixed behavior"], "weak_for": ["uniform simple pattern"]},
    "TimesFM":   {"cat": "foundation", "strengths": ["large-scale pre-trained forecasting", "general temporal patterns"],
                  "good_for": ["general forecasting", "cross-domain"], "weak_for": ["narrow domain-specific"]},
    "MOIRAI":    {"cat": "foundation", "strengths": ["any-variate universal forecasting", "multi-frequency support"],
                  "good_for": ["multi-frequency", "cross-variate"], "weak_for": ["single simple pattern"]},
    "TimerXL":   {"cat": "foundation", "strengths": ["multivariate next-token prediction", "fine-grained intra/inter-series attention"],
                  "good_for": ["multivariate causal", "long-context"], "weak_for": ["simple univariate"]},
    "Chronos_small": {"cat": "foundation", "strengths": ["quantization-based tokenization", "probabilistic sampling"],
                  "good_for": ["probabilistic forecasting", "zero-shot transfer"], "weak_for": ["deterministic precision"]},
}


# ============================================================
# Formatting functions
# ============================================================
def format_analysis_result(analysis_result: Dict, shuffle=False, seed=None) -> str:
    items = list(analysis_result.items())
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    lines = ["Historical series analysis:\n"]
    for feature, result in items:
        lines.append(f"[{feature}]")
        for key, value in result.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    return "\n".join(lines)


def format_analysis_result_knowledge(analysis_result: Dict, shuffle=False, seed=None) -> str:
    items = list(analysis_result.items())
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    lines = ["Similar sample analysis:\n"]
    for feature, result in items:
        lines.append(f"[{feature}]")
        for key, value in result.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    return "\n".join(lines)


def build_tools_description(tool_names: List[str], shuffle: bool = True) -> str:
    tool_items = [(n, TOOLS[n]) for n in tool_names if n in TOOLS]
    if shuffle:
        random.shuffle(tool_items)
    lines = ["Available forecasting tools:\n"]
    for name, desc in tool_items:
        lines.append(f"{name}: {desc}\n")
    return "\n".join(lines)


def _weights_to_pct(w, n_tools):
    pcts = [round(w[j] * 100) for j in range(n_tools)]
    diff = 100 - sum(pcts)
    if diff != 0:
        idx = max(range(len(pcts)), key=lambda k: w[k])
        pcts[idx] += diff
    return pcts


def format_weights_text(weights_2d, tool_names, header_prefix="Optimal ensemble weights(%) on the historical time series:"):
    tool_header = ",".join(tool_names)
    lines = [header_prefix, tool_header]
    for w in weights_2d:
        pcts = _weights_to_pct(w, len(tool_names))
        lines.append(",".join(str(p) for p in pcts))
    return "\n".join(lines)


# ============================================================
# RAG: find top-k similar samples from pool
# ============================================================
def find_topk_euclidean(query, pool, k=5):
    dists = np.linalg.norm(pool - query[None, :], axis=1)
    topk_indices = np.argpartition(dists, k)[:k]
    topk_indices = topk_indices[np.argsort(dists[topk_indices])]
    return topk_indices, pool[topk_indices], dists[topk_indices]


# ============================================================
# Rule-based think
# ============================================================
_TREND_TEMPLATES = {
    ("strong", "up"):   ["dominated by a strong upward trend", "exhibits pronounced upward drift", "clear upward momentum"],
    ("strong", "down"): ["driven by a strong downward trend", "marked downward trajectory"],
    ("weak", "up"):     ["mild upward tendency", "slight positive drift with weak trend strength"],
    ("weak", "down"):   ["slight downward drift", "marginal negative trend"],
    ("none", ""):       ["no discernible trend", "essentially flat with negligible trend component"],
}
_NOISE_TEMPLATES = {
    "low":    ["low noise level", "clean signal with little noise contamination", "very little noise"],
    "medium": ["moderate noise level", "noticeable but manageable noise", "medium noise contamination"],
    "high":   ["high noise level", "substantial noise contamination", "noisy profile"],
}
_ACF_TEMPLATES = {
    "very_high": ["extremely strong short-lag persistence", "near-unit-root autocorrelation", "very high short-lag dependence"],
    "moderate": ["moderate autocorrelation", "notable but not extreme lag dependence"],
    "weak": ["weak autocorrelation", "limited serial dependence"],
}
_TOOL_FIT_HIGH = [
    "{tool} fits best because {reason}", "{tool} is well-suited here as {reason}",
    "{tool} aligns strongly with these dynamics since {reason}", "{tool} is the natural choice given that {reason}",
]
_TOOL_FIT_MID = [
    "{tool} adds complementary value via {strength}", "{tool} can help modestly through {strength}",
    "{tool} contributes through its {strength}",
]
_TOOL_FIT_LOW = [
    "{tool} is less critical given {reason}", "{tool} has limited relevance here because {reason}",
    "{tool} plays a minor role as {reason}",
]
_CONCLUSION_TEMPLATES = [
    "allocate heavily to {dom}, keep moderate weight on {mid_str}, and minimal weight on {low_str}",
    "concentrate on {dom} for primary coverage, supplement with {mid_str}, residual on {low_str}",
    "prioritize {dom} given its fit, hedge with {mid_str}, and only token allocation to {low_str}",
    "{dom} dominates due to strong alignment, {mid_str} provide secondary coverage, {low_str} serve as minor hedges",
]


def _parse_features(analysis_result: Dict) -> Dict:
    f = {}
    st = analysis_result.get('stationarity', {})
    f['stat_judgment'] = str(st.get('Overall judgment', 'unknown'))
    f['stationary'] = 'non' not in f['stat_judgment'].lower() and 'stationary' in f['stat_judgment'].lower()
    tr = analysis_result.get('trend', {})
    f['trend_type'] = str(tr.get('Trend type', 'none')).lower()
    f['trend_strength'] = str(tr.get('Trend strength', 'none')).lower()
    ns = analysis_result.get('noise', {})
    f['noise_level'] = str(ns.get('Noise strength', 'unknown')).lower().strip()
    ac = analysis_result.get('autocorrelation', {})
    f['acf_judgment'] = str(ac.get('Autocorrelation judgment', ''))
    f['acf_val'] = ac.get('Dominant ACF value', None)
    f['has_acf'] = 'significant' in f['acf_judgment'].lower()
    se = analysis_result.get('seasonality', {})
    f['has_season'] = len(se) > 0
    di = analysis_result.get('distribution', {})
    f['normal'] = 'true' in str(di.get('Normal distribution', 'False')).lower()
    f['dist_type'] = str(di.get('Distribution type', 'unknown'))
    ol = analysis_result.get('iqr_outliers', {})
    f['outlier_count'] = ol.get('Outlier count', 0)
    f['has_outliers'] = f['outlier_count'] not in (0, '0')
    return f


def _step1_patterns(f: Dict, rng: random.Random) -> str:
    parts = []
    parts.append("non-stationary series" if not f['stationary'] else "stationary series")
    strength_key = "strong" if "strong" in f['trend_strength'] else ("weak" if "weak" in f['trend_strength'] else "none")
    dir_key = "up" if "up" in f['trend_type'] else ("down" if "down" in f['trend_type'] else "")
    templates = _TREND_TEMPLATES.get((strength_key, dir_key), _TREND_TEMPLATES[("none", "")])
    parts.append(rng.choice(templates))
    nl = f['noise_level']
    nl_key = "low" if "low" in nl else ("high" if "high" in nl else "medium")
    parts.append(rng.choice(_NOISE_TEMPLATES[nl_key]))
    if f['has_acf'] and f['acf_val'] is not None:
        try:
            acf_v = float(f['acf_val'])
            acf_key = "very_high" if acf_v > 0.9 else ("moderate" if acf_v > 0.7 else "weak")
            parts.append(rng.choice(_ACF_TEMPLATES[acf_key]))
        except (ValueError, TypeError):
            pass
    if f['has_season']:
        parts.append("seasonal component detected")
    if f['has_outliers']:
        parts.append("outliers present")
    else:
        parts.append("no outliers")
    if not f['normal']:
        parts.append(f"{f['dist_type'].lower()} distribution")
    return "Key patterns: " + ", ".join(parts) + "."


def _step2_tool_match(f: Dict, tool_names: List[str], w_avg: np.ndarray, rng: random.Random) -> str:
    sorted_idx = np.argsort(-w_avg)
    sentences = []
    for rank, idx in enumerate(sorted_idx):
        tool = tool_names[idx]
        w = w_avg[idx]
        traits = TOOL_TRAITS.get(tool, {"cat": "unknown", "strengths": ["general"], "good_for": [], "weak_for": []})
        cat = traits["cat"]
        strengths = traits["strengths"]
        match_reasons, mismatch_reasons = [], []
        if cat == "linear":
            if "strong" in f['trend_strength']: match_reasons.append("strong trend favors linear decomposition")
            if "low" in f['noise_level']: match_reasons.append("clean signal suits direct linear projection")
            if f['has_acf']: match_reasons.append("persistent autocorrelation aligns with sparse linear mapping")
            if "high" in f['noise_level']: mismatch_reasons.append("high noise limits simple linear approaches")
            if f['has_season']: mismatch_reasons.append("multi-periodic patterns need more than linear capacity")
        elif cat == "transformer":
            if f['has_acf']: match_reasons.append("attention can exploit autocorrelation structure")
            if f['has_outliers']: match_reasons.append("attention weighting handles sporadic outliers")
            if not f['has_acf'] and "low" in f['noise_level']: mismatch_reasons.append("simple clean dynamics may not need attention overhead")
        elif cat == "fft":
            if f['has_season']: match_reasons.append("FFT decomposition captures periodic structure")
            if not f['has_season'] and "strong" in f['trend_strength']: mismatch_reasons.append("trend-dominated profile lacks the periodicity FFT targets")
            if not f['has_season']: mismatch_reasons.append("no clear periodic component for frequency-domain analysis")
        elif cat == "decomposition":
            if f['has_season']: match_reasons.append("multi-scale decomposition separates seasonal from residual")
            if "strong" in f['trend_strength']: match_reasons.append("explicit trend-season separation benefits from strong trend")
            if not f['has_season'] and "weak" in f['trend_strength']: mismatch_reasons.append("no clear multi-scale structure to decompose")
        elif cat == "mlp":
            if f['has_season'] or "medium" in f['noise_level'] or "moderate" in f['noise_level']: match_reasons.append("multiscale mixing captures cross-resolution interactions")
            if "low" in f['noise_level'] and not f['has_season']: mismatch_reasons.append("simple clean patterns may not benefit from multiscale complexity")
        elif cat == "foundation":
            match_reasons.append(f"pre-trained knowledge enables {strengths[0]}")
            if "low" in f['noise_level'] and "strong" in f['trend_strength']: mismatch_reasons.append("clean trending data may favor specialized small models over zero-shot")
        elif cat == "multi-period":
            if f['has_season']: match_reasons.append("adaptive patching can align with periodic structure")
            if not f['has_season']: mismatch_reasons.append("no multi-period structure for adaptive patching to exploit")
        elif cat == "cnn":
            if f['has_acf']: match_reasons.append("local convolution captures short-range lag patterns")
            if "strong" in f['trend_strength']: mismatch_reasons.append("global trend may exceed local receptive field")
        if w > 0.35:
            reason = match_reasons[0] if match_reasons else f"{strengths[0]} aligns with series characteristics"
            sentences.append(rng.choice(_TOOL_FIT_HIGH).format(tool=tool, reason=reason))
        elif w > 0.15:
            strength = match_reasons[0] if match_reasons else strengths[0]
            sentences.append(rng.choice(_TOOL_FIT_MID).format(tool=tool, strength=strength))
        elif w > 0.05:
            reason = mismatch_reasons[0] if mismatch_reasons else f"the series profile doesn't strongly favor {cat} approaches"
            sentences.append(rng.choice(_TOOL_FIT_LOW).format(tool=tool, reason=reason))
    return "Tool match: " + "; ".join(sentences) + "."


def _step3_reference(f: Dict, f_know: Optional[Dict], knowledge_w: Optional[np.ndarray],
                     tool_names: List[str], w_avg: np.ndarray, rng: random.Random) -> str:
    if knowledge_w is None:
        return "Reference: no similar sample available for comparison."
    k_avg = knowledge_w.mean(axis=0)
    k_sorted = np.argsort(-k_avg)
    k_dom = tool_names[k_sorted[0]]
    c_sorted = np.argsort(-w_avg)
    c_dom = tool_names[c_sorted[0]]
    parts = [f"similar-series ensembles lean toward {k_dom}"]
    sec = [tool_names[j] for j in k_sorted[1:3] if k_avg[j] > 0.1]
    if sec:
        parts.append(f"with moderate {', '.join(sec)}")
    if c_dom != k_dom:
        diff_reasons = []
        if f_know is not None:
            if f['has_season'] != f_know.get('has_season', f['has_season']): diff_reasons.append("different periodicity profile")
            if ("strong" in f['trend_strength']) != ("strong" in f_know.get('trend_strength', '')): diff_reasons.append("different trend intensity")
            if f['noise_level'] != f_know.get('noise_level', f['noise_level']): diff_reasons.append("different noise characteristics")
        if diff_reasons:
            parts.append(f"but the current series differs in {', '.join(diff_reasons[:2])}, shifting preference toward {c_dom}")
        else:
            parts.append(f"but current characteristics better suit {c_dom}")
    else:
        parts.append("consistent with the current allocation direction")
    return "Reference: " + ", ".join(parts) + "."


def _step4_conclusion(tool_names: List[str], w_avg: np.ndarray, rng: random.Random) -> str:
    sorted_idx = np.argsort(-w_avg)
    dom = tool_names[sorted_idx[0]]
    mid_tools = [tool_names[j] for j in sorted_idx[1:] if w_avg[j] > 0.12]
    low_tools = [tool_names[j] for j in sorted_idx[1:] if 0.03 < w_avg[j] <= 0.12]
    mid_str = ", ".join(mid_tools) if mid_tools else "minor hedges"
    low_str = ", ".join(low_tools) if low_tools else "the rest"
    return "Conclusion: " + rng.choice(_CONCLUSION_TEMPLATES).format(dom=dom, mid_str=mid_str, low_str=low_str) + "."


def generate_thinking_rule_textonly(analysis_result: Dict, tool_names: List[str],
                                    sample_weights: np.ndarray,
                                    knowledge_weights: Optional[np.ndarray] = None,
                                    knowledge_analysis: Optional[Dict] = None,
                                    seed: Optional[int] = None) -> str:
    rng = random.Random(seed)
    f = _parse_features(analysis_result)
    f_know = _parse_features(knowledge_analysis) if knowledge_analysis is not None else None
    w_avg = sample_weights.mean(axis=0)
    s1 = _step1_patterns(f, rng)
    s2 = _step2_tool_match(f, tool_names, w_avg, rng)
    s3 = _step3_reference(f, f_know, knowledge_weights, tool_names, w_avg, rng)
    s4 = _step4_conclusion(tool_names, w_avg, rng)
    return f"{s1}\n{s2}\n{s3}\n{s4}"


# ============================================================
# Build samples for a split (no think)
# ============================================================
def build_eval_samples(data_bx, data_weights, tool_names, dataset_description,
                       knowledge_pool_X, knowledge_pool_W,
                       shuffle_analysis=False, seed_offset=0, topk=25):
    """Build samples for val/test splits (no thinking)."""
    N = data_bx.shape[0]
    samples = []
    for i in range(N):
        data_x = data_bx[i].squeeze(-1)
        sample_weights = data_weights[i]

        analysis_result = generate_analysis(data_x)
        analysis_text = format_analysis_result(analysis_result, shuffle=shuffle_analysis, seed=i + seed_offset)
        tools_text = build_tools_description(tool_names, shuffle=True)

        topk_idx, _, _ = find_topk_euclidean(data_x, knowledge_pool_X, k=topk)
        knowledge_weights = knowledge_pool_W[topk_idx].mean(axis=0)
        analysis_knowledge = generate_analysis(knowledge_pool_X[topk_idx[0]])
        analysis_text_knowledge = format_analysis_result_knowledge(
            analysis_knowledge, shuffle=shuffle_analysis, seed=i + seed_offset + 10000)
        knowledge_weights_text = format_weights_text(
            knowledge_weights, tool_names,
            header_prefix="Optimal ensemble weights(%) on the most similar time series:")
        label_weights_text = format_weights_text(
            sample_weights, tool_names,
            header_prefix="Optimal ensemble weights(%) on the historical time series:")

        human_prompt = (
            "Assign ensemble weights to the forecasting tools based on the information below.\n\n"
            f"{dataset_description}\n\n"
            f"{analysis_text}\n"
            f"{tools_text}\n"
            f"{analysis_text_knowledge}\n\n"
            f"{knowledge_weights_text}\n\n"
        )

        samples.append({
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "gpt", "value": label_weights_text}
            ],
            "system": SYSTEM_PROMPT,
        })

        if (i + 1) % 500 == 0:
            print(f"    Built {i+1}/{N} samples")

    return samples


# ============================================================
# Main pipeline
# ============================================================
def process_group(data_path, dataset, group_dir_name, think_mode='none',
                  shuffle_analysis=False, max_samples=0, num_groups=10, topk=25):
    src_dir = os.path.join(data_path, dataset, group_dir_name)
    if not os.path.isdir(src_dir):
        print(f"[SKIP] {src_dir} not found")
        return

    # Parse tool names from directory name
    parts = group_dir_name.split('_')
    tool_start = None
    for i, p in enumerate(parts):
        if p.startswith('top'):
            tool_start = i + 1
            break
    if tool_start is None:
        print(f"[ERROR] Cannot parse tool names from {group_dir_name}")
        return
    tool_names = parts[tool_start:]
    n_tools = len(tool_names)
    print(f"\nProcessing: {dataset} / {group_dir_name}")
    print(f"  Tools: {tool_names}")

    suffix = f'{SEQ_LEN}_{PRED_LEN}.npy'

    # Load train, val, test SEPARATELY
    train_bx = np.load(os.path.join(src_dir, f'train_batch_x_{suffix}'))
    train_by = np.load(os.path.join(src_dir, f'train_batch_y_{suffix}'))
    train_weights = np.load(os.path.join(src_dir, f'train_weights_{suffix}'))

    val_bx = np.load(os.path.join(src_dir, f'val_batch_x_{suffix}'))
    val_by = np.load(os.path.join(src_dir, f'val_batch_y_{suffix}'))
    val_weights = np.load(os.path.join(src_dir, f'val_weights_{suffix}'))

    test_bx = np.load(os.path.join(src_dir, f'test_batch_x_{suffix}'))
    test_by = np.load(os.path.join(src_dir, f'test_batch_y_{suffix}'))
    test_weights = np.load(os.path.join(src_dir, f'test_weights_{suffix}'))

    # Truncate weight groups
    num_groups = min(num_groups, train_weights.shape[1])
    train_weights = train_weights[:, :num_groups, :]
    val_weights = val_weights[:, :num_groups, :]
    test_weights = test_weights[:, :num_groups, :]

    N_train = train_bx.shape[0]
    N_val = val_bx.shape[0]
    N_test = test_bx.shape[0]

    if max_samples > 0:
        N_train = min(N_train, max_samples)
        train_bx = train_bx[:N_train]
        train_by = train_by[:N_train]
        train_weights = train_weights[:N_train]

    print(f"  train: {N_train}, val: {N_val}, test: {N_test}")

    # RAG knowledge pool = train only
    knowledge_pool_X = train_bx.squeeze(-1)
    knowledge_pool_W = train_weights

    ds_desc = DATASET_DESCRIPTIONS.get(dataset, dataset)
    dataset_description = f"Dataset: {ds_desc}. Task: predict the next {PRED_LEN} values from the historical data."

    # ---- Build TRAIN samples (with think) ----
    print("  Building train samples...")
    train_samples_raw = []
    for i in range(N_train):
        data_x = train_bx[i].squeeze(-1)
        sample_weights = train_weights[i]

        analysis_result = generate_analysis(data_x)
        analysis_text = format_analysis_result(analysis_result, shuffle=shuffle_analysis, seed=i)
        tools_text = build_tools_description(tool_names, shuffle=True)

        # RAG: top-k similar (exclude self)
        topk_idx, _, _ = find_topk_euclidean(data_x, knowledge_pool_X, k=topk+1)
        topk_idx = topk_idx[topk_idx != i][:topk]
        knowledge_x = knowledge_pool_X[topk_idx].mean(axis=0)
        knowledge_weights_rag = knowledge_pool_W[topk_idx].mean(axis=0)

        analysis_knowledge = generate_analysis(knowledge_x)
        analysis_text_knowledge = format_analysis_result_knowledge(
            analysis_knowledge, shuffle=shuffle_analysis, seed=i + 10000)
        knowledge_weights_text = format_weights_text(
            knowledge_weights_rag, tool_names,
            header_prefix="Optimal ensemble weights(%) on the most similar time series:")
        label_weights_text = format_weights_text(
            sample_weights, tool_names,
            header_prefix="Optimal ensemble weights(%) on the historical time series:")

        human_prompt = (
            "Assign ensemble weights to the forecasting tools based on the information below.\n\n"
            f"{dataset_description}\n\n"
            f"{analysis_text}\n\n"
            f"{tools_text}\n"
            f"{analysis_text_knowledge}\n\n"
            f"{knowledge_weights_text}\n\n"
        )

        train_samples_raw.append({
            'human_prompt': human_prompt,
            'label_weights_text': label_weights_text,
            'analysis_result': analysis_result,
            'sample_weights': sample_weights,
            'knowledge_weights': knowledge_weights_rag,
            'knowledge_analysis': analysis_knowledge,
        })

        if (i + 1) % 500 == 0:
            print(f"    Built {i+1}/{N_train} train samples")

    # Generate think for train
    if think_mode == 'rule':
        print(f"  Generating rule-based think for {N_train} train samples...")
        think_texts = []
        for i in range(N_train):
            s = train_samples_raw[i]
            t = generate_thinking_rule_textonly(
                s['analysis_result'], tool_names,
                s['sample_weights'], s['knowledge_weights'],
                knowledge_analysis=s.get('knowledge_analysis'), seed=i)
            think_texts.append(t)
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{N_train} think done")
    else:
        think_texts = [None] * N_train

    # Assemble train JSONL
    train_final = []
    for i in range(N_train):
        lw = train_samples_raw[i]['label_weights_text']
        if think_texts[i]:
            gpt_value = f"<think>\n{think_texts[i]}\n</think>\n{lw}"
        else:
            gpt_value = lw
        train_final.append({
            "conversations": [
                {"from": "human", "value": train_samples_raw[i]['human_prompt']},
                {"from": "gpt", "value": gpt_value}
            ],
            "system": SYSTEM_PROMPT,
        })

    # ---- Build VAL samples (no think) ----
    print("  Building val samples...")
    val_final = build_eval_samples(
        val_bx, val_weights, tool_names, dataset_description,
        knowledge_pool_X, knowledge_pool_W,
        shuffle_analysis=shuffle_analysis, seed_offset=100000, topk=topk)

    # ---- Build TEST samples (no think) ----
    print("  Building test samples...")
    test_final = build_eval_samples(
        test_bx, test_weights, tool_names, dataset_description,
        knowledge_pool_X, knowledge_pool_W,
        shuffle_analysis=shuffle_analysis, seed_offset=200000, topk=topk)

    # ---- Save ----
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    think_tag = f"_{think_mode}" if think_mode != 'none' else "_nothink"

    def save_jsonl(samples, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("[\n")
            for i, sample in enumerate(samples):
                json.dump(sample, f, ensure_ascii=False)
                if i != len(samples) - 1:
                    f.write(",\n")
            f.write("\n]")

    train_path = os.path.join(OUTPUT_BASE, f"{dataset}_{group_dir_name}{think_tag}_top{topk}_train.jsonl")
    val_path = os.path.join(OUTPUT_BASE, f"{dataset}_{group_dir_name}_top{topk}_val.jsonl")
    test_path = os.path.join(OUTPUT_BASE, f"{dataset}_{group_dir_name}_top{topk}_test.jsonl")

    save_jsonl(train_final, train_path)
    print(f"  Saved {len(train_final)} train -> {train_path}")
    save_jsonl(val_final, val_path)
    print(f"  Saved {len(val_final)} val -> {val_path}")
    save_jsonl(test_final, test_path)
    print(f"  Saved {len(test_final)} test -> {test_path}")


def register_dataset_info():
    """Scan all JSONL files in OUTPUT_BASE and write dataset_info.json."""
    info = {}
    jsonl_files = sorted(glob(os.path.join(OUTPUT_BASE, '*.jsonl')))
    for fpath in jsonl_files:
        fname = os.path.basename(fpath)
        key = fname.replace('.jsonl', '')
        info[key] = {
            'file_name': fname,
            'formatting': 'sharegpt',
            'columns': {
                'messages': 'conversations',
                'system': 'system'
            }
        }
    info_path = os.path.join(OUTPUT_BASE, 'dataset_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\nRegistered {len(info)} entries in {info_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name or "all"')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                        help='Root directory containing all dataset directories')
    parser.add_argument('--group', type=str, default=None, help='Specific group dir name')
    parser.add_argument('--group_pattern', type=str, default=None, help='Glob pattern for group dirs')
    parser.add_argument('--think_mode', type=str, default='none', choices=['none', 'rule'],
                        help='Think generation mode: none or rule (text-only)')
    parser.add_argument('--shuffle_analysis', action='store_true', default=False)
    parser.add_argument('--max_samples', type=int, default=0, help='Limit train samples (0=all)')
    parser.add_argument('--num_groups', type=int, default=10)
    parser.add_argument('--topk', type=int, default=25, help='Top-k similar samples for RAG')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        ds_dir = os.path.join(args.data_path, ds)
        if not os.path.isdir(ds_dir):
            print(f"[SKIP] {ds_dir} not found")
            continue

        if args.group:
            groups = [args.group]
        elif args.group_pattern:
            import fnmatch
            groups = sorted([d for d in os.listdir(ds_dir)
                           if os.path.isdir(os.path.join(ds_dir, d))
                           and fnmatch.fnmatch(d, args.group_pattern)])
        else:
            groups = sorted([d for d in os.listdir(ds_dir)
                           if os.path.isdir(os.path.join(ds_dir, d))])

        print(f"\nDataset: {ds}, groups to process: {len(groups)}")
        for g in groups:
            process_group(args.data_path, ds, g, think_mode=args.think_mode,
                         shuffle_analysis=args.shuffle_analysis,
                         max_samples=args.max_samples,
                         num_groups=args.num_groups,
                         topk=args.topk)

    register_dataset_info()
    print("\nAll done.")


if __name__ == '__main__':
    main()
