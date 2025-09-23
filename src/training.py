import os
from configparser import ConfigParser
from transformers import set_seed
from Utils import *
from model_utils import *
from models.featurize import TreeFeaturizer
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import time
import csv
from plot_utils import *
from utils.functions import *
from models._TCNN_deepsets import *
import sys
import numpy as np
from models._ftrl import *
torch.backends.cudnn.enabled = False
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True)

class LoadMemNet(object):
    def __init__(self, cfg):
        pass
        self.cfg = cfg
        self.dataset_json_path = cfg.get("Train", "dataset_json_path")
        self.batch_size = cfg.getint("Train", "batch_size")
        self.n_epochs = cfg.getint("Train", "n_epochs")
        self.vali_rate = cfg.getfloat("Train", "vali_rate")
        self.buffer = cfg.getint("Train", "buffer")
        self.buffer_k_list = eval(cfg.get("Train", "buffer_k_list"))
        self.test_note = cfg.get("Train", "note")
        self.model_name = cfg.get("Train", "model")
        self.wmp_clusters_num = cfg.getint("Train", "wmp_clusters_num")
        self.loss_type = cfg.get("Train", "loss_type")
        self.buffer_flag = False
        self.phi_hidden_dims = eval(cfg.get("Train", "phi_hidden_dims"))
        self.phi_output_dim = cfg.getint("Train", "phi_output_dim")

        self.wl_to_scaled_memory = dict()
        self.scale = get_scaler("std")
        self.dataset_json_name = os.path.splitext(os.path.basename(self.dataset_json_path))[0][len("run_"):]

        json_name = self.dataset_json_name
        if "_mod_" in json_name:
            self.dataset_name = "mod"
        else:
            raise

        today_ymd = get_today_ymd_compact()
        if self.model_name in ["WMP"]:
            self.results_path = f"./RESULT/{self.dataset_name}/{today_ymd}/{self.dataset_json_name}_e{self.n_epochs}_bz{self.batch_size}_{self.model_name}{self.wmp_clusters_num}_{self.loss_type}_{self.test_note}/"
        else:
            self.results_path = f"./RESULT/{self.dataset_name}/{today_ymd}/{self.dataset_json_name}_e{self.n_epochs}_bz{self.batch_size}_{self.model_name}_{self.loss_type}_{self.test_note}/"
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path, exist_ok=True)

    def prepare_data(self):
        self.dataset_json = load_json(self.dataset_json_path)

        self.wl_to_sqls = {}
        self.wl_to_plans = {}
        self.wl_to_memory = {}
        self.wl_to_planscost = {}


        for workload in self.dataset_json:
            wname = workload.get("workload_name")
            sql_names = []
            plans = []
            planscost = []
            for query in workload.get("queries", []):
                sql_names.append(query.get("sql_name"))
                plans.append(query.get("plan"))
                planscost.append(query.get("plan")["Plan"]["Total Cost"])
            self.wl_to_sqls[wname] = sql_names
            self.wl_to_plans[wname] = plans
            self.wl_to_memory[wname] = workload.get("peak_memory_mb")
            self.wl_to_planscost[wname] = sum(planscost)

        self.workloads_list = list(self.wl_to_sqls.keys())
        self.X_train, self.X_vali, self.X_test = split_workloads(self.workloads_list, self.vali_rate)

        mem_list = [v for k,v in self.wl_to_memory.items() if k in self.X_train]
        self.scale.fit(mem_list)
        for wl, mem in self.wl_to_memory.items():
            self.wl_to_scaled_memory[wl] = self.scale.transform(mem)

        self.wl_to_vectorized_plans = {}
        featurizer = TreeFeaturizer()
        all_train_plans = []
        for wl, plans in self.wl_to_plans.items():
            if wl in self.X_train:
                all_train_plans.extend(plans)
        featurizer.fit(all_train_plans)

        for wl, plans in self.wl_to_plans.items():
            vectorized_plans = [np.array(feature) for feature in featurizer.transform(plans)]
            self.wl_to_vectorized_plans[wl] = vectorized_plans

        self.wmp_num_clusters = self.wmp_clusters_num
        train_file = f'./WMP_plan/cluster_data/train_workloads_final_{str(self.wmp_num_clusters)}_clusters.csv'
        test_file = f'./WMP_plan/cluster_data/test_workloads_final_{str(self.wmp_num_clusters)}_clusters.csv'
        train_dict = build_workload_feature_dict(train_file)
        test_dict = build_workload_feature_dict(test_file)
        self.wl_to_vectorized_plans_wmp = train_dict.copy()
        self.wl_to_vectorized_plans_wmp.update(test_dict)

        dump_json("/path/to/artifacts/","wl_to_sqls.json", self.wl_to_sqls)
        dump_json("/path/to/artifacts/","wl_to_memory.json", self.wl_to_memory)
        save_as_pkl("/path/to/artifacts/wl_to_vectorized_plans.pkl", self.wl_to_vectorized_plans)

    def collate(self, x):
        vectorized_plans_orig = [self.wl_to_vectorized_plans[w] for w in x]

        vectorized_plans = list(zip(*vectorized_plans_orig))
        vectorized_plans = [list(x) for x in vectorized_plans]

        scaled_memory = [self.wl_to_scaled_memory[w] for w in x]
        return vectorized_plans, torch.tensor(scaled_memory, device=device, dtype=torch.float32), x

    def train(self):
        train_set =DataLoader(self.X_train, self.batch_size, shuffle=True, drop_last=False, collate_fn=self.collate)
        vali_set = DataLoader(self.X_vali, self.batch_size, shuffle=True, drop_last=False, collate_fn=self.collate)

        if self.model_name == "DS":
            self.tcnn = TCNNDS(9,phi_hidden_dims= self.phi_hidden_dims,phi_output_dim=self.phi_output_dim).to(device)

        optimizer = optim.Adam(self.tcnn.parameters(), lr=0.001)

        criterion = get_loss_function(self.loss_type)

        best_val_loss = float('inf')
        train_losses, val_losses = [], []
        for epoch in range(self.n_epochs):
            start_time = time.time()
            self.tcnn.train()
            train_loss = 0
            for vectorized_plans, scaled_memory, _ in train_set:
                optimizer.zero_grad()
                mean, log_var = self.tcnn(*vectorized_plans,)
                if self.loss_type == 'gs':
                    y_pred = torch.stack((mean, log_var), dim=1)
                    loss = criterion(scaled_memory, y_pred)
                else:
                    y_pred = mean
                    loss = criterion(y_pred, scaled_memory)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_set)

            self.tcnn.eval()
            val_loss = 0
            with torch.no_grad():
                for vectorized_plans, scaled_memory, _ in vali_set:
                    mean, log_var = self.tcnn(*vectorized_plans, )
                    if self.loss_type == 'gs':
                        y_pred = torch.stack((mean, log_var), dim=1)
                        loss = criterion(scaled_memory, y_pred)
                    else:
                        y_pred = mean
                        loss = criterion(y_pred, scaled_memory)
                    val_loss += loss.item()
            val_loss /= len(vali_set)

            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{self.n_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.tcnn.state_dict())
                print(f"New best model found at epoch {epoch+1} with validation loss {val_loss:.4f}")

        best_model_path = self.results_path + "best_model.pth"
        torch.save(best_model_state, best_model_path)

    def test_model(self):
        if not hasattr(self, 'X_test'):
            if self.model_name == "DS":
                self.tcnn = TCNNDS(9, phi_hidden_dims=self.phi_hidden_dims, phi_output_dim=self.phi_output_dim).to(
                    device)
            _, _, self.X_test = split_workloads(self.workloads_list, self.vali_rate)

        test_set = DataLoader(self.X_test, self.batch_size, shuffle=False, drop_last=False, collate_fn=self.collate)

        self.tcnn.load_state_dict(torch.load(self.results_path + "best_model.pth"))
        self.tcnn.eval()

        buffer_flag = getattr(self, "buffer_flag", 0)
        if buffer_flag:
            buffer_k_list = getattr(self, "buffer_k_list", [1.0, 2.0])
        else:
            buffer_k_list = [0.0]

        all_pred_dict = {k: [] for k in buffer_k_list}
        all_workload_names = []
        all_true = []
        total_test_time = 0

        with torch.no_grad():
            for vectorized_plans, scaled_memory, workload_names in test_set:
                batch_start_time = time.time()

                mean, log_var = self.tcnn(*vectorized_plans)
                std = torch.exp(0.5 * log_var)

                mean_np = mean.cpu().numpy()
                std_np = std.cpu().numpy()
                all_workload_names.extend(workload_names)
                all_true.append(self.scale.detransform(scaled_memory.cpu().numpy()))

                for k in buffer_k_list:
                    pred_buffered = mean_np + k * std_np
                    pred_buffered_np = self.scale.detransform(pred_buffered)
                    all_pred_dict[k].append(pred_buffered_np)

                batch_time = time.time() - batch_start_time
                total_test_time += batch_time

                print(f"Batch processed in {batch_time:.2f} seconds")

        w_true_mem_all = np.concatenate(all_true, axis=0)
        wl_names = all_workload_names


        num_samples = len(w_true_mem_all)

        for k in buffer_k_list:
            w_pred_mem_all = np.concatenate(all_pred_dict[k], axis=0)
            mem_mae = mae(w_true_mem_all, w_pred_mem_all)
            mem_mse = np.mean((w_true_mem_all - w_pred_mem_all) ** 2)
            mem_mape = mape(w_true_mem_all, w_pred_mem_all)
            mem_rmse = rmse(w_true_mem_all, w_pred_mem_all)

            num_oom = np.sum(w_pred_mem_all < w_true_mem_all)
            oom_ratio = num_oom / num_samples

            if k == 0.0:
                filename = self.results_path + f"results.csv"
            else:
                filename = self.results_path + f"results_buffer_k_{k}.csv"

            csv_result_list = [['workload', 'sqls', 'true_memory', 'pred_memory']]
            for idx, w in enumerate(wl_names):
                pred_value = w_pred_mem_all[idx, 0] if w_pred_mem_all.ndim > 1 else w_pred_mem_all[idx]

                csv_result_list.append([
                    w,
                    self.wl_to_sqls[w],
                    self.wl_to_memory[w],
                    pred_value,
                ])

            with open(filename, "w", newline='') as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerows(csv_result_list)
                csv_writer.writerow(['============================================================='])
                csv_writer.writerow(['mem_MAPE: {0}%'.format(mem_mape)])
                csv_writer.writerow(['mem_MAE: {0}'.format(mem_mae)])
                csv_writer.writerow(['mem_RMSE: {0}'.format(mem_rmse)])
                csv_writer.writerow(['mem_MSE: {0}'.format(mem_mse)])
                csv_writer.writerow(['num_samples: {0}'.format(num_samples)])
                csv_writer.writerow(['OOM_count (pred<true): {0}'.format(num_oom)])
                csv_writer.writerow(['OOM_ratio: {0:.4f}'.format(oom_ratio)])

            scatter_path = self.results_path + (f"pred_scatter_buffer_k_{k}.png" if k != 0.0 else "pred_scatter.png")
            scatter_plot(w_true_mem_all, w_pred_mem_all, save_path=scatter_path)

            print(f"Test samples: {num_samples}, OOM count: {num_oom}, OOM ratio: {oom_ratio:.4f}")

        print(f"Total testing time: {total_test_time:.2f} seconds")

    def do(self):
        self.prepare_data()
        model = self.cfg.get("Train", "model")
        if model in ["DS"]:
            self.train()
            self.test_model()

    def run_task(self, run_task):
        if run_task in ["normal"]:
            random_seed = self.cfg.getint("Train", "random_seed")
            set_seed(random_seed)
            self.do()

if __name__ == '__main__':
    cfg = ConfigParser(inline_comment_prefixes=(';', '#'))
    cfg.read("./conf.ini")

    args = sys.argv
    run_task = cfg.get("Train", "task")
    model = args[1]
    dataset_json_path = args[2]
    batch_size = args[3]

    cfg.set("Train", "model", model)
    cfg.set("Train", "dataset_json_path", dataset_json_path)
    cfg.set("Train", "batch_size", batch_size)


    a = LoadMemNet(cfg)
    res = a.run_task(run_task)
