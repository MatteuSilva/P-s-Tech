import json
import pandas as pd

def load_json(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as f:
        return pd.json_normalize(json.load(f))

def load_all(data_dir: str) -> dict:
    return {
        'vagas': load_json(f'{data_dir}/vagas.json'),
        'prospects': load_json(f'{data_dir}/prospects.json'),
        'applicants': load_json(f'{data_dir}/applicants.json'),
    }

if __name__ == '__main__':
    dfs = load_all('../data')
    for name, df in dfs.items():
        print(f"{name}: {df.shape}")
