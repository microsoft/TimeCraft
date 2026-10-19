
import json
import os
import re
import glob
import argparse

DATASET_DIR = './datasets'
INPUT_LEN = 96
PRED_LEN = 96


def parse_filename(filepath):
    """Parse dataset, group_dir, and split from an SFT filename.

    Filename pattern: {dataset}_{group_dir}_rule_train.jsonl
    Example: exchange_rate_mixed_top1_CARD_CATS_DLinear_MOMENT_rule_train.jsonl
        -> dataset='exchange_rate', group_dir='mixed_top1_CARD_CATS_DLinear_MOMENT', split='train'
    """
    basename = os.path.basename(filepath)
    # Remove extension
    name = basename.replace('.jsonl', '')

    # Determine split and suffix
    if name.endswith('_rule_train_top3'):
        split = 'train'
        name = name[:-len('_rule_train_top3')]
    elif name.endswith('_rule_train'):
        split = 'train'
        name = name[:-len('_rule_train')]
    elif name.endswith('_test'):
        split = 'test'
        name = name[:-len('_test')]
    else:
        return None, None, None

    # Split into dataset and group_dir.
    for category in ['pure_small', 'pure_zeroshot', 'mixed']:
        pattern_top = f'_{category}_top'
        idx = name.find(pattern_top)
        if idx != -1:
            dataset = name[:idx]
            group_dir = name[idx + 1:]
            return dataset, group_dir, split

        pattern_cross = f'_cross_{category}_'
        idx = name.find(pattern_cross)
        if idx != -1:
            dataset = name[:idx]
            group_dir = name[idx + 1:]  # "cross_mixed_DLinear_..."
            return dataset, group_dir, split

    return None, None, None


def convert_sft_to_grpo(input_path, output_path, dataset, group_dir, split):
    """Convert a single SFT file to GRPO format."""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        if not isinstance(data, list):
            data = [data]
    except json.JSONDecodeError:
        data = []
        for line in content.splitlines():
            line = line.strip()
            if line and line not in ('[', ']'):
                line = line.rstrip(',')
                if line:
                    data.append(json.loads(line))

    grpo_samples = []
    for idx, item in enumerate(data):
        messages = []

        system = item.get('system', '')
        if system:
            messages.append({'role': 'system', 'content': system})

        for turn in item.get('conversations', []):
            if turn['from'] == 'human':
                messages.append({'role': 'user', 'content': turn['value']})

        grpo_sample = {
            'messages': messages,
            'sample_id': idx,
            'dataset_name': dataset,
            'group_dir': group_dir,
            'split': split,
            'input_len': INPUT_LEN,
            'pred_len': PRED_LEN,
        }
        grpo_samples.append(grpo_sample)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in grpo_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f'  {len(grpo_samples)} samples -> {os.path.basename(output_path)}')
    return len(grpo_samples)


def main():
    parser = argparse.ArgumentParser(description='Convert SFT data to GRPO format')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name, e.g. exchange_rate. If not set, process all.')
    parser.add_argument('--group', type=str, default=None,
                        help='Group dir name, e.g. mixed_top1_CARD_CATS_DLinear_MOMENT. If not set, process all groups for the dataset.')
    parser.add_argument('--data_dir', type=str, default=DATASET_DIR,
                        help='SFT data directory')
    parser.add_argument('--suffix', type=str, default=None,
                        help='Only process files with this suffix, e.g. top3')
    args = parser.parse_args()

    if args.dataset and args.group:
        # Single file mode
        input_path = os.path.join(args.data_dir, f'{args.dataset}_{args.group}_rule_train.jsonl')
        if not os.path.exists(input_path):
            print(f'ERROR: {input_path} not found')
            return
        output_path = input_path.replace('_rule_train.jsonl', '_rule_grpo_train.jsonl')
        print(f'[{args.dataset}] {args.group}:')
        convert_sft_to_grpo(input_path, output_path, args.dataset, args.group, 'train')

    elif args.dataset:
        # All groups for one dataset
        pattern = os.path.join(args.data_dir, f'{args.dataset}_*_rule_train.jsonl')
        train_files = sorted(glob.glob(pattern))
        print(f'Found {len(train_files)} train files for {args.dataset}')
        for input_path in train_files:
            dataset, group_dir, split = parse_filename(input_path)
            if dataset is None:
                continue
            output_path = input_path.replace('_rule_train.jsonl', '_rule_grpo_train.jsonl')
            print(f'[{dataset}] {group_dir}:')
            convert_sft_to_grpo(input_path, output_path, dataset, group_dir, 'train')

    else:
        # Batch: all files
        if args.suffix:
            train_files = sorted(glob.glob(os.path.join(args.data_dir, f'*_rule_train_{args.suffix}.jsonl')))
        else:
            train_files = sorted(
                glob.glob(os.path.join(args.data_dir, '*_rule_train.jsonl'))
                + glob.glob(os.path.join(args.data_dir, '*_rule_train_top3.jsonl'))
            )
        print(f'Found {len(train_files)} train files in {args.data_dir}')
        total = 0
        for input_path in train_files:
            dataset, group_dir, split = parse_filename(input_path)
            if dataset is None:
                continue
            if input_path.endswith('_rule_train_top3.jsonl'):
                output_path = input_path.replace('_rule_train_top3.jsonl', '_rule_grpo_train_top3.jsonl')
            else:
                output_path = input_path.replace('_rule_train.jsonl', '_rule_grpo_train.jsonl')
            print(f'[{dataset}] {group_dir}:')
            n = convert_sft_to_grpo(input_path, output_path, dataset, group_dir, 'train')
            total += n
        print(f'\nTotal: {total} GRPO samples from {len(train_files)} files')


if __name__ == '__main__':
    main()
