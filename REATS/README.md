# REATS: LLM Reasoning-based Ensemble Learning for Adaptive Time Series Forecasting

REATS uses a lightweight 1.7B LLM as an interpretable ensemble router for time series forecasting. It combines textual temporal descriptions, numerical features, and retrieved similar-sample priors to generate sample-adaptive weights, using rule-based chain-of-thought supervision and SFT followed by GRPO with reciprocal reward mapping. Experiments on eight benchmarks demonstrate improved performance over competitive ensemble baselines across foundation-model and small-model candidate groups, alongside natural-language explanations, cross-dataset transfer, and generalization to unseen candidate models.

The paper is available at [[Paper]](https://arxiv.org/pdf/2608.10149).

## 1. Generate Base Model Predictions

Run inference for all small models (11 models) on each dataset to produce prediction `.npy` files.

```bash
python generate_base_preds.py --models_root /path/to/model-repo --dataset ETTh1 --gpu 0
python generate_base_preds.py --models_root /path/to/model-repo --dataset ETTh1 --models SEMixer TimeXer --gpu 0
```

`--models_root` specifies the external model source directory containing `models/`; replace it with your actual path.

Output: `{split}_pred_x_all_96_96.npy` for each model/dataset/split.

## 2. Build SFT Dataset

Construct SFT training data in ShareGPT format. Each sample contains time series analysis, RAG-retrieved similar samples, and oracle ensemble weights as labels. The `<think>` reasoning in train labels can be generated via rule-based templates (`--think_mode rule`) or LLM distillation (`--think_mode llm`).

```bash
python build_sft_dataset.py --dataset ETTh1 --think_mode rule
python build_sft_dataset.py --dataset all --think_mode rule --topk 3
python build_sft_dataset.py --dataset weather --data_path /path/to/checkpoints_forecast_UTS
```

Key arguments:
- `--data_path`: root directory containing all dataset directories, including `weather`
- `--think_mode rule`: generate rule-based `<think>` reasoning in train labels
- `--topk 3`: number of similar samples for RAG retrieval
- `--dataset all`: process all 8 datasets

Output: `./datasets/{dataset}_{group}_rule_top{k}_{split}.jsonl`

## 3. SFT Training

**Prerequisites**: Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) following its official documentation.

Run SFT (full fine-tuning):

```bash
bash train_sft.sh
```

Or specify datasets:

```bash
bash train_sft.sh "ETTh1_mixed_top1_SEMixer_TimeMixer_TimeXer_Timer_rule_top3_train ETTh2_mixed_top1_SEMixer_TimeMixer_TimeXer_Timer_rule_top3_train"
```

The default SFT learning rate is `1e-5` and the per-device training batch size is `2`.
Override them with `LEARNING_RATE` and `BATCH_SIZE`, respectively.

## 4. Build GRPO Dataset

Convert SFT data to GRPO format (prompt-only, no labels).

```bash
python build_grpo_dataset.py
```

Output: `./datasets/{dataset}_{group}_rule_grpo_train_top3.jsonl`

## 5. GRPO Training

**Prerequisites**: Install [ms-swift](https://github.com/modelscope/ms-swift) and vLLM following its official documentation.

Run GRPO:

```bash
MODEL=./output/sft/ETTh1_xxx_sft/checkpoint-xxx \
GRPO_DATA=./datasets/ETTh1_xxx_rule_grpo_train_top3.jsonl \
bash train_grpo.sh
```

The default GRPO learning rate is `5e-6`. Override it with `LEARNING_RATE`.

## File Structure

```
REATS/
  generate_base_preds.py   # Step 1: base model inference
  build_sft_dataset.py             # Step 2: SFT data construction
  train_sft.sh                     # Step 3: SFT training script
  build_grpo_dataset.py            # Step 4: GRPO data construction
  grpo_reward.py                   # Step 5: GRPO reward function
  train_grpo.sh                    # Step 5: GRPO training script
```
