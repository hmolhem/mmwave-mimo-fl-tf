import os
import json
import csv
from typing import List, Dict, Any

# Script to aggregate centralized and federated cross-day metrics into flat CSV/JSON exports.
# Output directory: exports/

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CROSS_DAY_FED_DIR = os.path.join(ROOT, 'outputs', 'cross_day_fed')
CROSS_DAY_CENTRAL_DIR = os.path.join(ROOT, 'outputs', 'cross_day')
EXPORT_DIR = os.path.join(ROOT, 'exports')

os.makedirs(EXPORT_DIR, exist_ok=True)

Record = Dict[str, Any]

def collect_federated() -> List[Record]:
    records: List[Record] = []
    if not os.path.isdir(CROSS_DAY_FED_DIR):
        return records
    for entry in os.listdir(CROSS_DAY_FED_DIR):
        path = os.path.join(CROSS_DAY_FED_DIR, entry)
        if not os.path.isdir(path):
            continue
        # Expect folder pattern: model_dayX e.g. baseline_day1
        parts = entry.split('_day')
        if len(parts) != 2:
            continue
        model = parts[0]
        train_day_str = parts[1]
        if not train_day_str.isdigit():
            continue
        train_day = int(train_day_str)
        # inside: train_day{d}_{model}
        train_folder = f'train_day{train_day}_{model}'
        inner = os.path.join(path, train_folder)
        if not os.path.isdir(inner):
            continue
        cross_day_results = os.path.join(inner, 'cross_day_results.json')
        if os.path.isfile(cross_day_results):
            with open(cross_day_results, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k, v in data.get('results', {}).items():
                records.append({
                    'mode': 'federated',
                    'model': model,
                    'train_day': train_day,
                    'test_day': v['test_day'],
                    'accuracy': v['test_accuracy'],
                    'loss': v['test_loss'],
                    'num_samples': v.get('num_samples', None),
                    'rounds': data.get('rounds'),
                    'local_epochs': data.get('local_epochs'),
                    'normalize': data.get('normalize'),
                })
        else:
            # fallback: individual test_day*_metrics.json files
            for fname in os.listdir(inner):
                if fname.startswith('test_day') and fname.endswith('_metrics.json'):
                    metrics_path = os.path.join(inner, fname)
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        v = json.load(f)
                    records.append({
                        'mode': 'federated',
                        'model': model,
                        'train_day': train_day,
                        'test_day': v['test_day'],
                        'accuracy': v['test_accuracy'],
                        'loss': v['test_loss'],
                        'num_samples': v.get('num_samples', None),
                        'rounds': None,
                        'local_epochs': None,
                        'normalize': None,
                    })
    return records

def collect_centralized() -> List[Record]:
    records: List[Record] = []
    if not os.path.isdir(CROSS_DAY_CENTRAL_DIR):
        return records
    # Folders may be like baseline_day0 or improved_day0? Inspect similar to federated? Use pattern search.
    for entry in os.listdir(CROSS_DAY_CENTRAL_DIR):
        path = os.path.join(CROSS_DAY_CENTRAL_DIR, entry)
        if not os.path.isdir(path):
            continue
        # Expect like baseline_day0 or improved_day0
        parts = entry.split('_day')
        if len(parts) != 2:
            continue
        model = parts[0]
        train_day_str = parts[1]
        if not train_day_str.isdigit():
            continue
        train_day = int(train_day_str)
        # inside: train_day{d}_{model}
        inner = os.path.join(path, f'train_day{train_day}_{model}')
        if not os.path.isdir(inner):
            continue
        for fname in os.listdir(inner):
            if fname.startswith('test_day') and fname.endswith('_metrics.json'):
                metrics_path = os.path.join(inner, fname)
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    v = json.load(f)
                records.append({
                    'mode': 'centralized',
                    'model': model,
                    'train_day': train_day,
                    'test_day': v['test_day'],
                    'accuracy': v['test_accuracy'],
                    'loss': v['test_loss'],
                    'num_samples': v.get('num_samples', None),
                    'rounds': None,
                    'local_epochs': None,
                    'normalize': None,
                })
    return records

def write_flat_csv(records: List[Record], out_path: str):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def write_json(records: List[Record], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)


def pivot_accuracy(records: List[Record], mode: str, model: str) -> Dict[str, Dict[str, float]]:
    # Returns nested dict: train_day -> test_day -> accuracy
    pivot: Dict[str, Dict[str, float]] = {}
    for r in records:
        if r['mode'] != mode or r['model'] != model:
            continue
        td = str(r['train_day'])
        if td not in pivot:
            pivot[td] = {}
        pivot[td][str(r['test_day'])] = r['accuracy'] * 100.0  # percent
    return pivot


def write_pivot_csv(pivot: Dict[str, Dict[str, float]], out_path: str):
    # Collect all test_days
    test_days = sorted({int(k2) for v in pivot.values() for k2 in v.keys()})
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['TrainDay'] + [f'TestDay{d}' for d in test_days]
        writer.writerow(header)
        for train_day in sorted(pivot.keys(), key=lambda x: int(x)):
            row = [train_day]
            for d in test_days:
                row.append(f'{pivot[train_day].get(str(d), ""):.2f}')
            writer.writerow(row)


def main():
    fed_records = collect_federated()
    cent_records = collect_centralized()
    all_records = fed_records + cent_records

    if not all_records:
        print('No records found. Ensure experiments were run.')
        return

    flat_csv = os.path.join(EXPORT_DIR, 'metrics_flat.csv')
    flat_json = os.path.join(EXPORT_DIR, 'metrics_flat.json')
    write_flat_csv(all_records, flat_csv)
    write_json(all_records, flat_json)

    for mode in ['federated', 'centralized']:
        for model in ['baseline', 'improved']:
            pivot = pivot_accuracy(all_records, mode, model)
            if pivot:
                pivot_csv = os.path.join(EXPORT_DIR, f'{mode}_{model}_accuracy_matrix.csv')
                write_pivot_csv(pivot, pivot_csv)

    print('Export complete:')
    print(f'  Flat CSV: {flat_csv}')
    print(f'  Flat JSON: {flat_json}')
    print('  Matrices:')
    for mode in ['federated', 'centralized']:
        for model in ['baseline', 'improved']:
            path = os.path.join(EXPORT_DIR, f'{mode}_{model}_accuracy_matrix.csv')
            if os.path.isfile(path):
                print(f'    {path}')

if __name__ == '__main__':
    main()
