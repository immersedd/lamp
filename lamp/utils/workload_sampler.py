from configparser import ConfigParser
import os
import random
import json

def load_sqls(sql_path):
    all_queries = []
    for root, _, files in os.walk(sql_path):
        for file in sorted(files):
            full_path = os.path.join(root, file)
            with open(full_path, encoding = 'latin1') as f:
                query_content = f.read()
                all_queries.append({"name": file, "sql": query_content})
    return all_queries

def generate_fixed_workload(sql_path, n, num_groups, replace=False):
    all_queries = load_sqls(sql_path)
    workloads = []
    for i in range(num_groups):
        if replace:
            group_queries = random.choices(all_queries, k=n)
        else:
            group_queries = random.sample(all_queries, n)
        group = {
            "name": f"w{i+1}",
            "queries": group_queries
        }
        workloads.append(group)

    sql_folder_name = os.path.basename(os.path.normpath(sql_path))
    replace_flag = "1" if replace else "0"
    file_name = f"F_{sql_folder_name}_N{n}_W{num_groups}_R{replace_flag}.json"

    output_dir = os.path.join(f"{sql_folder_name}_workloads_json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, "w") as f:
        json.dump(workloads, f, indent=4, ensure_ascii=False)

    print(f"Saved workloads to {file_path}")
    return workloads


def generate_fixed_workload_unique(sql_path, n, num_groups):

    all_queries = load_sqls(sql_path)
    total_needed = n * num_groups

    if total_needed > len(all_queries):
        raise ValueError(
            f"{total_needed} > {len(all_queries)}"
        )

    selected = random.sample(all_queries, total_needed)

    workloads = []
    for i in range(num_groups):
        group_queries = selected[i * n : (i + 1) * n]
        workloads.append({
            "name": f"w{i+1}",
            "queries": group_queries
        })

    sql_folder_name = os.path.basename(os.path.normpath(sql_path))
    file_name = f"F_Uni_{sql_folder_name}_N{n}_W{num_groups}.json"
    output_dir = f"{sql_folder_name}_workloads_json"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(workloads, f, indent=4, ensure_ascii=False)

    print(f"Saved workloads to {file_path}")
    return workloads

def generate_variable_workload(sql_path, min_n, max_n, num_groups, replace=False):

    all_queries = load_sqls(sql_path)
    workloads = []
    for i in range(num_groups):
        n = random.randint(min_n, max_n)
        if replace:
            group_queries = random.choices(all_queries, k=n)
        else:
            group_queries = random.sample(all_queries, n)
        group = {
            "name": f"w{i+1}",
            "queries": group_queries
        }
        workloads.append(group)

    sql_folder_name = os.path.basename(os.path.normpath(sql_path))
    replace_flag = "1" if replace else "0"
    file_name = f"V_{sql_folder_name}_N{min_n}-{max_n}_W{num_groups}_R{replace_flag}.json"

    output_dir = os.path.join(f"{sql_folder_name}_workloads_json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, "w") as f:
        json.dump(workloads, f, indent=4, ensure_ascii=False)

    print(f"Saved variable workloads to {file_path}")
    return workloads

def generate_variable_workload_unique(sql_path, min_n, max_n, num_groups):

    all_queries = load_sqls(sql_path)

    group_sizes = [random.randint(min_n, max_n) for _ in range(num_groups)]
    total_needed = sum(group_sizes)

    if total_needed > len(all_queries):
        raise ValueError(
            f"{total_needed} > {len(all_queries)}"
        )

    selected = random.sample(all_queries, total_needed)

    workloads = []
    idx = 0
    for i, size in enumerate(group_sizes):
        group_queries = selected[idx : idx + size]
        workloads.append({
            "name": f"w{i+1}",
            "queries": group_queries
        })
        idx += size

    sql_folder_name = os.path.basename(os.path.normpath(sql_path))
    file_name = f"V_Uni_{sql_folder_name}_N{min_n}-{max_n}_W{num_groups}.json"
    output_dir = f"{sql_folder_name}_workloads_json"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(workloads, f, indent=4, ensure_ascii=False)

    print(f"Saved unique variable workloads to {file_path}")
    return workloads

def load_workload(file_path):
    with open(file_path, 'r') as f:
        workloads = json.load(f)
    return workloads

if __name__ == '__main__':
    cfg = ConfigParser(inline_comment_prefixes=(';', '#'))
    cfg.read("../conf.ini")

    all_queries_path = cfg.get("WorkloadSampling", "sql")
    seed = cfg.get("WorkloadSampling", "random_seed")
    random.seed(seed)

    if cfg.get("WorkloadSampling", "mode") == "fixed":
        fixed_n = cfg.getint("WorkloadSampling", "fixed_n")
        fixed_num_groups = cfg.getint("WorkloadSampling", "fixed_num_groups")
        fixed_replace = eval(cfg.get("WorkloadSampling", "fixed_replace"))
        if cfg.get("WorkloadSampling", "sampling_mode") == "duplicate":
            generate_fixed_workload(all_queries_path, fixed_n, fixed_num_groups, fixed_replace)
        elif cfg.get("WorkloadSampling", "sampling_mode") == "unique":
            generate_fixed_workload_unique(all_queries_path, fixed_n, fixed_num_groups)

    elif cfg.get("WorkloadSampling", "mode") == "variable":
        variable_min_n = cfg.getint("WorkloadSampling", "variable_min_n")
        variable_max_n = cfg.getint("WorkloadSampling", "variable_max_n")
        variable_num_groups = cfg.getint("WorkloadSampling", "variable_num_groups")
        sampling_mode = cfg.get("WorkloadSampling", "sampling_mode")

        if sampling_mode == "duplicate":
            variable_replace = eval(cfg.get("WorkloadSampling", "variable_replace"))
            generate_variable_workload(all_queries_path,
                                       variable_min_n,
                                       variable_max_n,
                                       variable_num_groups,
                                       variable_replace)
        elif sampling_mode == "unique":
            generate_variable_workload_unique(all_queries_path,
                                              variable_min_n,
                                              variable_max_n,
                                              variable_num_groups)


