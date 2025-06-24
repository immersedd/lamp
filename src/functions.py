import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split

def split_workloads(workloads: List[str], vali_rate: float, random_seed: int = 42):
    train_wls, temp_wls = train_test_split(workloads, test_size=vali_rate, random_state=random_seed)
    vali_wls, test_wls = train_test_split(temp_wls, test_size=0.5, random_state=random_seed)
    return train_wls, vali_wls, test_wls

def build_workload_feature_dict(csv_file):
    df = pd.read_csv(csv_file)
    features = df.drop(['WORKLOAD', 'MEM_mb'], axis=1).values
    workloads = df['WORKLOAD'].values
    workload_feature_dict = {workload: features[i] for i, workload in enumerate(workloads)}

    return workload_feature_dict

def get_ops_metrics(csv_path):
    metrics = [
        'Bitmap Index Scan_COUNT',
        'Bitmap Index Scan_PLANROWS',
        'Hash Join_COUNT',
        'Hash Join_PLANROWS',
        'Index Only Scan_COUNT',
        'Index Only Scan_PLANROWS',
        'Index Scan_COUNT',
        'Index Scan_PLANROWS',
        'Merge Join_COUNT',
        'Merge Join_PLANROWS',
        'Nested Loop_COUNT',
        'Nested Loop_PLANROWS',
        'Seq Scan_COUNT',
        'Seq Scan_PLANROWS'
    ]
    df = pd.read_csv(csv_path, encoding='utf-8')
    result = {}
    for _, row in df.iterrows():
        key = row['WORKLOAD_SQLNAME']
        values = [row[col] for col in metrics]
        result[key] = values

    return result


import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_ops_vector_dict(ops_metrics, scaler=None):
    df = pd.DataFrame.from_dict(ops_metrics, orient='index') \
        .fillna(0)

    if scaler is None:
        scaler = StandardScaler().fit(df.values)
    X = scaler.transform(df.values)

    vector_dict = {k: X[i] for i, k in enumerate(df.index)}
    import torch
    ops_tensor_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in vector_dict.items()}
    return ops_tensor_dict, scaler
