
# BRIDGE: Pipeline

This module implements a multi-agent, LLM-driven pipeline for refining textual descriptions of time series data. It encompasses candidate collection, template extraction, description generation, and iterative refinement.

## File Overview

- `self_refine_main.py`: Main entry point for the self-refinement pipeline.
- `ts_to_text.py`: Converts time series data into textual descriptions.
- `multi_agent_refiner.py`: Implements the multi-agent refinement process.
- `template_extractor.py`: Extracts templates from textual data.
- `llm_agents/`: Contains LLM agent implementations and tools.
- `self_refine/`: Contains modules for task initialization, iteration, feedback, and prompt building.

## Setup Instructions

1. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set Up API Keys:**

Ensure you have an OpenAI API key. You can set it as an environment variable or pass it as a command-line argument.

## Usage Examples

### 1. Generate Descriptions from Time Series Data

```bash
python self_refine_main.py \
  --openai_key YOUR_API_KEY \
  --ts_file path/to/your_timeseries.csv \
  --dataset_name your_dataset_name \
  --ts_to_text
```

### 2. Collect Textual Candidates from the Web

```bash
python self_refine_main.py \
  --openai_key YOUR_API_KEY \
  --collect_candidate
```

### 3. Extract Templates from Documents

```bash
python self_refine_main.py \
  --openai_key YOUR_API_KEY \
  --extract_template \
  --template_input_file path/to/your_input_file.json
```

### 4. Run Multi-Agent Self-Refinement

```bash
python self_refine_main.py \
  --openai_key YOUR_API_KEY \
  --ts_file path/to/your_timeseries.csv \
  --dataset_name your_dataset_name \
  --refine \
  --team_iterations 3 \
  --global_iterations 2
```

## Command-Line Arguments

| Argument                      | Description                                                                |
|-------------------------------|----------------------------------------------------------------------------|
| `--openai_key`                | OpenAI API key.                                                            |
| `--openai_model`              | OpenAI model name (default: `gpt-4o-2024-05-13`).                          |
| `--llm_optimize`              | Use LLM to optimize text descriptions.                                     |
| `--ts_file`                   | Path to the CSV file containing time series data.                          |
| `--dataset_name`              | Name/ID of the dataset for description templates.                          |
| `--prediction_length`         | Prediction length for time series windows (default: 168).                  |
| `--dataset_template_file`     | Path to dataset template JSON file.                                        |
| `--description_template_file` | Path to description template JSON file.                                    |
| `--json_file`                 | Dataset description JSON file (default: `dataset_description_bank.json`). |
| `--output_file`               | Output file for descriptions or refinement results.                        |
| `--output_template_file`      | Path to save extracted templates.                                          |
| `--collect_candidate`         | Collect text candidates from the web.                                      |
| `--extract_template`          | Extract templates from documents.                                          |
| `--template_input_file`       | Input file path for template extraction.                                   |
| `--feedback_example_file`     | Path to the feedback examples JSON file.                                   |
| `--refine`                    | Run the multi-agent self-refinement process.                               |
| `--refine_iterations`         | Max attempts for legacy refinement iterations (default: 10).               |
| `--tests`                     | Number of refinement tests (default: 5).                                   |
| `--team_iterations`           | Max iterations per team before convergence (default: 3).                   |
| `--global_iterations`         | Max global refinement rounds (default: 2).                                 |
| `--predictor_method`          | Prediction method for generating new time series during refinement.        |