import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import csv
import statistics
import glob

stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
BATCH_SIZES = ['64', '128', '256', '512']

gpu_map = {
        "p2.xlarge" : "K80-1",
        "chameleon.xlarge" : "gpus-11",
        "p2.8xlarge" : "K80-8",
        "p2.8xlarge_2" : "K80-8_2",
        "p2.16xlarge" : "K80-16",
        "p3.2xlarge" : "V100-1",
        "p3.8xlarge" : "V100-4",
        "p3.16xlarge" : "V100-8"}

cost_map = {
    "p2.xlarge" : 0.9,
    "p2.8xlarge" : 7.2,
    "p2.8xlarge_2" : 14.4,
    "p2.16xlarge" : 14.4,
    "p3.2xlarge" : 3.06,
    "p3.8xlarge" : 12.24,
    "p3.8xlarge_2" : 24.48,
    "p3.16xlarge" : 24.48}

instances = []

def process_json(model, gpu, batch, json_path):

    with open(json_path) as fd:
        dagJson = json.load(fd)

    stats[model][gpu][batch]["TRAIN_SPEED_INGESTION"] = dagJson["SPEED_INGESTION"]
    stats[model][gpu][batch]["TRAIN_SPEED_DISK"]  = dagJson["SPEED_DISK"]
    stats[model][gpu][batch]["TRAIN_SPEED_CACHED"] = dagJson["SPEED_CACHED"]
    stats[model][gpu][batch]["DISK_THR"] = dagJson["DISK_THR"]
#    stats[model][gpu][batch]["MEM_THR"] = dagJson["MEM_THR"]
    stats[model][gpu][batch]["TRAIN_TIME_INGESTION"] = dagJson["RUN1"]["TRAIN"]
    stats[model][gpu][batch]["TRAIN_TIME_DISK"] = dagJson["RUN2"]["TRAIN"]
    stats[model][gpu][batch]["TRAIN_TIME_CACHED"] = dagJson["RUN3"]["TRAIN"]
    stats[model][gpu][batch]["CPU_UTIL_DISK_PCT"] = dagJson["RUN2"]["CPU"]
    stats[model][gpu][batch]["CPU_UTIL_CACHED_PCT"] = dagJson["RUN3"]["CPU"]
    stats[model][gpu][batch]["GPU_UTIL_DISK_PCT"] = dagJson["RUN2"]["GPU_UTIL"]
    stats[model][gpu][batch]["GPU_UTIL_CACHED_PCT"] = dagJson["RUN3"]["GPU_UTIL"]
    stats[model][gpu][batch]["GPU_MEM_UTIL_DISK_PCT"] = dagJson["RUN2"]["GPU_MEM_UTIL"]
    stats[model][gpu][batch]["GPU_MEM_UTIL_CACHED_PCT"] = dagJson["RUN3"]["GPU_MEM_UTIL"]
    stats[model][gpu][batch]["MEMCPY_TIME"] = dagJson["RUN1"]["MEMCPY"]
    stats[model][gpu][batch]["COMPUTE_TIME"] = dagJson["RUN3"]["COMPUTE"]
    stats[model][gpu][batch]["COMPUTE_BWD_TIME"] = dagJson["RUN3"]["COMPUTE_BWD"]
    stats[model][gpu][batch]["COMPUTE_FWD_TIME"] = stats[model][gpu][batch]["COMPUTE_TIME"] - stats[model][gpu][batch]["COMPUTE_BWD_TIME"]

    stats[model][gpu][batch]["PREP_STALL_TIME"] = dagJson["RUN3"]["TRAIN"] - dagJson["RUN1"]["TRAIN"]
    stats[model][gpu][batch]["FETCH_STALL_TIME"] = dagJson["RUN2"]["TRAIN"] - stats[model][gpu][batch]["PREP_STALL_TIME"]
    if "RUN0" in dagJson:
        stats[model][gpu][batch]["INTERCONNECT_STALL_TIME"] = dagJson["RUN1"]["TRAIN"] - dagJson["RUN0"]["TRAIN"]

    stats[model][gpu][batch]["PREP_STALL_PCT"] = stats[model][gpu][batch]["PREP_STALL_TIME"] / stats[model][gpu][batch]["TRAIN_TIME_CACHED"] * 100
    stats[model][gpu][batch]["FETCH_STALL_PCT"] = stats[model][gpu][batch]["FETCH_STALL_TIME"] / stats[model][gpu][batch]["TRAIN_TIME_DISK"] * 100
    stats[model][gpu][batch]["INTERCONNECT_STALL_PCT"] = stats[model][gpu][batch]["INTERCONNECT_STALL_TIME"] / stats[model][gpu][batch]["TRAIN_TIME_INGESTION"] * 100



def process_json2(model, instance, batch, json_path):

    gpu = gpu_map[instance]
    with open(json_path) as fd:
        dagJson = json.load(fd)

    stats[model][gpu][batch]["TRAIN_SPEED_INGESTION"] = dagJson["SPEED_INGESTION"]
    stats[model][gpu][batch]["TRAIN_SPEED_DISK"] = dagJson["SPEED_DISK"]
    stats[model][gpu][batch]["TRAIN_SPEED_CACHED"] = dagJson["SPEED_CACHED"]
    stats[model][gpu][batch]["DISK_THR"] = dagJson["DISK_THR"]
    stats[model][gpu][batch]["TRAIN_TIME_INGESTION"] = dagJson["RUN1"]["TRAIN"]
    stats[model][gpu][batch]["TRAIN_TIME_DISK"] = dagJson["RUN2"]["TRAIN"]
    stats[model][gpu][batch]["TRAIN_TIME_CACHED"] = dagJson["RUN3"]["TRAIN"]
    stats[model][gpu][batch]["MEM_DISK"] = dagJson["RUN2"]["MEM"]
    stats[model][gpu][batch]["PCACHE_DISK"] = dagJson["RUN2"]["PCACHE"]
    stats[model][gpu][batch]["MEM_CACHED"] = dagJson["RUN3"]["MEM"]
    stats[model][gpu][batch]["PCACHE_CACHED"] = dagJson["RUN3"]["PCACHE"]
    stats[model][gpu][batch]["READ_WRITE_DISK"] = dagJson["RUN2"]["READ"] + dagJson["RUN2"]["WRITE"]
    stats[model][gpu][batch]["IO_WAIT_DISK"] = dagJson["RUN2"]["IO_WAIT"]
    stats[model][gpu][batch]["READ_WRITE_CACHED"] = dagJson["RUN3"]["READ"] + dagJson["RUN3"]["WRITE"]
    stats[model][gpu][batch]["IO_WAIT_CACHED"] = dagJson["RUN3"]["IO_WAIT"]

    stats[model][gpu][batch]["CPU_UTIL_DISK_PCT"] = dagJson["RUN2"]["CPU"]
    stats[model][gpu][batch]["CPU_UTIL_CACHED_PCT"] = dagJson["RUN3"]["CPU"]
    stats[model][gpu][batch]["GPU_UTIL_DISK_PCT"] = dagJson["RUN2"]["GPU_UTIL"]
    stats[model][gpu][batch]["GPU_UTIL_CACHED_PCT"] = dagJson["RUN3"]["GPU_UTIL"]
    stats[model][gpu][batch]["GPU_MEM_UTIL_DISK_PCT"] = dagJson["RUN2"]["GPU_MEM_UTIL"]
    stats[model][gpu][batch]["GPU_MEM_UTIL_CACHED_PCT"] = dagJson["RUN3"]["GPU_MEM_UTIL"]

    stats[model][gpu][batch]["MEMCPY_TIME"] = dagJson["RUN1"]["MEMCPY"]
    stats[model][gpu][batch]["COMPUTE_TIME"] = dagJson["RUN3"]["COMPUTE"]
    stats[model][gpu][batch]["COMPUTE_BWD_TIME"] = dagJson["RUN3"]["COMPUTE_BWD"]
    stats[model][gpu][batch]["COMPUTE_FWD_TIME"] = stats[model][gpu][batch]["COMPUTE_TIME"] - stats[model][gpu][batch]["COMPUTE_BWD_TIME"]

    stats[model][gpu][batch]["CPU_UTIL_DISK_LIST"] = dagJson["RUN2"]["CPU_LIST"]
    stats[model][gpu][batch]["CPU_UTIL_CACHED_LIST"] = dagJson["RUN3"]["CPU_LIST"]
    stats[model][gpu][batch]["GPU_UTIL_DISK_LIST"] = dagJson["RUN2"]["GPU_UTIL_LIST"]
    stats[model][gpu][batch]["GPU_UTIL_CACHED_LIST"] = dagJson["RUN3"]["GPU_UTIL_LIST"]
    stats[model][gpu][batch]["GPU_MEM_UTIL_DISK_LIST"] = dagJson["RUN2"]["GPU_MEM_UTIL_LIST"]
    stats[model][gpu][batch]["GPU_MEM_UTIL_CACHED_LIST"] = dagJson["RUN3"]["GPU_MEM_UTIL_LIST"]
    stats[model][gpu][batch]["READ_WRITE_LIST_DISK"] = dagJson["RUN2"]["READ_LIST"] + dagJson["RUN2"]["WRITE_LIST"]
    stats[model][gpu][batch]["READ_WRITE_LIST_CACHED"] = dagJson["RUN3"]["READ_LIST"] + dagJson["RUN3"]["WRITE_LIST"]
    stats[model][gpu][batch]["IO_WAIT_LIST_DISK"] = dagJson["RUN2"]["IO_WAIT_LIST"]
    stats[model][gpu][batch]["IO_WAIT_LIST_CACHED"] = dagJson["RUN3"]["IO_WAIT_LIST"]

    stats[model][gpu][batch]["PREP_STALL_TIME"] = dagJson["RUN3"]["TRAIN"] - dagJson["RUN1"]["TRAIN"]
    stats[model][gpu][batch]["FETCH_STALL_TIME"] = dagJson["RUN2"]["TRAIN"] - stats[model][gpu][batch]["PREP_STALL_TIME"]
    if "RUN0" in dagJson:
        stats[model][gpu][batch]["INTERCONNECT_STALL_TIME"] = dagJson["RUN1"]["TRAIN"] - dagJson["RUN0"]["TRAIN"]

    stats[model][gpu][batch]["PREP_STALL_PCT"] = stats[model][gpu][batch]["PREP_STALL_TIME"] / stats[model][gpu][batch]["TRAIN_TIME_CACHED"] * 100
    stats[model][gpu][batch]["FETCH_STALL_PCT"] = stats[model][gpu][batch]["FETCH_STALL_TIME"] / stats[model][gpu][batch]["TRAIN_TIME_DISK"] * 100
    if "INTERCONNECT_STALL_TIME" in stats[model][gpu][batch]:
        stats[model][gpu][batch]["INTERCONNECT_STALL_PCT"] = stats[model][gpu][batch]["INTERCONNECT_STALL_TIME"] / stats[model][gpu][batch]["TRAIN_TIME_INGESTION"] * 100


def process_csv(model, instance, batch, csv_path):

    gpu = gpu_map[instance]
    stats[model][gpu][batch]["DATA_TIME_LIST"] = []
    stats[model][gpu][batch]["COMPUTE_TIME_LIST"] = []
    stats[model][gpu][batch]["COMPUTE_TIME_FWD_LIST"] = []
    stats[model][gpu][batch]["COMPUTE_TIME_BWD_LIST"] = []

    files = glob.glob(csv_path + 'time*.csv')
    csv_readers = []

    for csv_file in files:
        csv_reader = csv.reader(open(csv_file))
        next(csv_reader)
        csv_readers.append(csv_reader)

    num_lines = min(len(open(files[i]).readlines()) for i in range(len(files)))

    for i in range(num_lines-1):
        
        data_time = []
        compute_time = []
        compute_fwd_time = []
        compute_bwd_time = []

        for csv_reader in csv_readers:

            row = next(csv_reader)

            data_time.append(float(row[1]))
            compute_time.append(float(row[2]))
            compute_fwd_time.append(float(row[2]) - float(row[3]))
            compute_bwd_time.append(float(row[3]))

        stats[model][gpu][batch]["DATA_TIME_LIST"].append(statistics.mean(data_time))
        stats[model][gpu][batch]["COMPUTE_TIME_LIST"].append(statistics.mean(compute_time))
        stats[model][gpu][batch]["COMPUTE_TIME_FWD_LIST"].append(statistics.mean(compute_fwd_time))
        stats[model][gpu][batch]["COMPUTE_TIME_BWD_LIST"].append(statistics.mean(compute_bwd_time))

def add_text(X, Y, axs):
    for idx, value in enumerate(X):
        axs.text(value, Y[idx]+2, str(int(Y[idx])), fontsize=20)

def compare_instances(result_dir):

    """
    for instance in instances:

        gpu = gpu_map[instance]

        for model in models:

            if gpu not in stats[model]:
                del stats[model]
                continue

    X = [model for model in stats.keys()]
    """


    X_small = ['alexnet', 'resnet18', 'shufflenet_v2_x0_5', 'mobilenet_v2', 'squeezenet1_0']
    X_large = ['resnet50', 'vgg11']
    desc = ["-Small_models", "-Large_models"]
    desc_i = 0

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    for X in [X_small, X_large]:


        X_axis = np.arange(len(X))

        for batch in BATCH_SIZES:
            diff = 0

            fig1, axs1 = plt.subplots(3, 1, figsize=(30, 20))
            fig2, axs2 = plt.subplots(3, 1, figsize=(30, 20))
            fig3, axs3 = plt.subplots(3, 1, figsize=(30, 20))
            fig4, axs4 = plt.subplots(3, 1, figsize=(30, 20))
            fig5, axs5 = plt.subplots(3, 1, figsize=(30, 20))
            fig6, axs6 = plt.subplots(3, 1, figsize=(30, 20))
            fig7, axs7 = plt.subplots(figsize=(30, 20))
            fig8, axs8 = plt.subplots(2, 1, figsize=(30, 20))

            for instance in instances:

                gpu = gpu_map[instance]
                
                Y_PREP_STALL_PCT = [stats[model][gpu][batch]["PREP_STALL_PCT"]
                                    if "PREP_STALL_PCT" in stats[model][gpu][batch] else 0 for model in X]
                Y_FETCH_STALL_PCT = [stats[model][gpu][batch]["FETCH_STALL_PCT"]
                                     if "FETCH_STALL_PCT" in stats[model][gpu][batch] else 0 for model in X]
                if not (instance == "p2.xlarge" or instance == "p3.2xlarge"):
                    Y_INTERCONNECT_STALL_PCT = [stats[model][gpu][batch]["INTERCONNECT_STALL_PCT"]
                                                if "INTERCONNECT_STALL_PCT" in stats[model][gpu][batch] else 0 for model in X]

                Y_TRAIN_TIME_DISK = [stats[model][gpu][batch]["TRAIN_TIME_DISK"]
                                     if "TRAIN_TIME_DISK" in stats[model][gpu][batch] else 0 for model in X]
                Y_TRAIN_TIME_CACHED = [stats[model][gpu][batch]["TRAIN_TIME_CACHED"]
                                       if "TRAIN_TIME_CACHED" in stats[model][gpu][batch] else 0 for model in X]

                Y_COST_DISK = [stats[model][gpu][batch]["TRAIN_TIME_DISK"] * cost_map[instance]  / 3600
                               if "TRAIN_TIME_DISK" in stats[model][gpu][batch] else 0 for model in X]
                Y_COST_CACHED = [stats[model][gpu][batch]["TRAIN_TIME_CACHED"] * cost_map[instance] / 3600
                                 if "TRAIN_TIME_CACHED" in stats[model][gpu][batch] else 0 for model in X]

                Y_DISK_THR = [stats[model][gpu][batch]["DISK_THR"]
                              if "DISK_THR" in stats[model][gpu][batch] else 0 for model in X]

                Y_TRAIN_SPEED_INGESTION = [stats[model][gpu][batch]["TRAIN_SPEED_INGESTION"]
                                           if "TRAIN_SPEED_INGESTION" in stats[model][gpu][batch] else 0 for model in X]
                Y_TRAIN_SPEED_DISK = [stats[model][gpu][batch]["TRAIN_SPEED_DISK"]
                                      if "TRAIN_SPEED_DISK" in stats[model][gpu][batch] else 0 for model in X]
                Y_TRAIN_SPEED_CACHED = [stats[model][gpu][batch]["TRAIN_SPEED_CACHED"]
                                        if "TRAIN_SPEED_CACHED" in stats[model][gpu][batch] else 0 for model in X]

                Y_CPU_UTIL_DISK_PCT = [stats[model][gpu][batch]["CPU_UTIL_DISK_PCT"]
                                       if "CPU_UTIL_DISK_PCT" in stats[model][gpu][batch] else 0 for model in X]
                Y_CPU_UTIL_CACHED_PCT = [stats[model][gpu][batch]["CPU_UTIL_CACHED_PCT"]
                                         if "CPU_UTIL_CACHED_PCT" in stats[model][gpu][batch] else 0 for model in X]

                Y_GPU_UTIL_DISK_PCT = [stats[model][gpu][batch]["GPU_UTIL_DISK_PCT"]
                                       if "GPU_UTIL_DISK_PCT" in stats[model][gpu][batch] else 0 for model in X]
                Y_GPU_UTIL_CACHED_PCT = [stats[model][gpu][batch]["GPU_UTIL_CACHED_PCT"]
                                         if "GPU_UTIL_CACHED_PCT" in stats[model][gpu][batch] else 0 for model in X]
                Y_GPU_MEM_UTIL_DISK_PCT = [stats[model][gpu][batch]["GPU_MEM_UTIL_DISK_PCT"]
                                           if "GPU_MEM_UTIL_DISK_PCT" in stats[model][gpu][batch] else 0 for model in X]
                Y_GPU_MEM_UTIL_CACHED_PCT = [stats[model][gpu][batch]["GPU_MEM_UTIL_CACHED_PCT"]
                                             if "GPU_MEM_UTIL_CACHED_PCT" in stats[model][gpu][batch] else 0 for model in X]

                Y_MEMCPY_TIME = [stats[model][gpu][batch]["MEMCPY_TIME"]
                                 if "MEMCPY_TIME" in stats[model][gpu][batch] else 0 for model in X]
                Y_COMPUTE_TIME = [stats[model][gpu][batch]["COMPUTE_TIME"]
                                  if "COMPUTE_TIME" in stats[model][gpu][batch] else 0 for model in X]
                Y_COMPUTE_FWD_TIME = [stats[model][gpu][batch]["COMPUTE_FWD_TIME"]
                                      if "COMPUTE_FWD_TIME" in stats[model][gpu][batch] else 0 for model in X]
                Y_COMPUTE_BWD_TIME = [stats[model][gpu][batch]["COMPUTE_BWD_TIME"]
                                      if "COMPUTE_BWD_TIME" in stats[model][gpu][batch] else 0 for model in X]

                axs1[0].bar(X_axis-0.2 + diff, Y_PREP_STALL_PCT, 0.2, label=instance)
                axs1[1].bar(X_axis-0.2 + diff, Y_FETCH_STALL_PCT, 0.2, label=instance)
                add_text(X_axis-0.25 + diff, Y_PREP_STALL_PCT, axs1[0])
                add_text(X_axis-0.25 + diff, Y_FETCH_STALL_PCT, axs1[1])

                if not (instance == "p2.xlarge" or instance == "p3.2xlarge"):
                    axs1[2].bar(X_axis-0.2 + diff, Y_INTERCONNECT_STALL_PCT, 0.2, label=instance)
                    add_text(X_axis-0.25 + diff, Y_INTERCONNECT_STALL_PCT, axs1[2])

                axs2[0].bar(X_axis-0.2 + diff, Y_TRAIN_TIME_DISK, 0.2, label=instance)
                axs2[1].bar(X_axis-0.2 + diff, Y_TRAIN_TIME_CACHED, 0.2, label=instance)
                axs2[2].bar(X_axis-0.2 + diff, Y_DISK_THR, 0.2, label=instance)
                add_text(X_axis-0.25 + diff, Y_TRAIN_TIME_DISK, axs2[0])
                add_text(X_axis-0.25 + diff, Y_TRAIN_TIME_CACHED, axs2[1])
                add_text(X_axis-0.25 + diff, Y_DISK_THR, axs2[2])

                axs8[0].bar(X_axis-0.2 + diff, Y_COST_DISK, 0.2, label=instance)
                axs8[1].bar(X_axis-0.2 + diff, Y_COST_CACHED, 0.2, label=instance)
                add_text(X_axis-0.25 + diff, Y_COST_DISK, axs2[0])
                add_text(X_axis-0.25 + diff, Y_COST_CACHED, axs2[1])

                axs3[0].bar(X_axis-0.2 + diff, Y_TRAIN_SPEED_INGESTION, 0.2, label = instance)
                axs3[1].bar(X_axis-0.2 + diff , Y_TRAIN_SPEED_DISK, 0.2, label = instance)
                axs3[2].bar(X_axis-0.2 + diff, Y_TRAIN_SPEED_CACHED, 0.2, label = instance)
                add_text(X_axis-0.25 + diff, Y_TRAIN_SPEED_INGESTION, axs3[0])
                add_text(X_axis-0.25 + diff, Y_TRAIN_SPEED_DISK, axs3[1])
                add_text(X_axis-0.25 + diff, Y_TRAIN_SPEED_CACHED, axs3[2])

                axs4[0].bar(X_axis-0.2 + diff, Y_CPU_UTIL_DISK_PCT, 0.2, label = instance)
                axs4[1].bar(X_axis-0.2 + diff, Y_GPU_UTIL_DISK_PCT, 0.2, label = instance)
                axs4[2].bar(X_axis-0.2 + diff, Y_GPU_MEM_UTIL_DISK_PCT, 0.2, label = instance)
                add_text(X_axis-0.25 + diff, Y_CPU_UTIL_DISK_PCT, axs4[0])
                add_text(X_axis-0.25 + diff, Y_GPU_UTIL_DISK_PCT, axs4[1])
                add_text(X_axis-0.25 + diff, Y_GPU_MEM_UTIL_DISK_PCT, axs4[2])

                axs5[0].bar(X_axis-0.2 + diff, Y_CPU_UTIL_CACHED_PCT, 0.2, label = instance)
                axs5[1].bar(X_axis-0.2 + diff, Y_GPU_UTIL_CACHED_PCT, 0.2, label = instance)
                axs5[2].bar(X_axis-0.2 + diff, Y_GPU_MEM_UTIL_CACHED_PCT, 0.2, label = instance)
                add_text(X_axis-0.25 + diff, Y_CPU_UTIL_CACHED_PCT, axs5[0])
                add_text(X_axis-0.25 + diff, Y_GPU_UTIL_CACHED_PCT, axs5[1])
                add_text(X_axis-0.25 + diff, Y_GPU_MEM_UTIL_CACHED_PCT, axs5[2])

                axs6[0].bar(X_axis-0.2 + diff, Y_MEMCPY_TIME, 0.2, label = instance)
                axs6[1].bar(X_axis-0.2 + diff, Y_COMPUTE_FWD_TIME, 0.2, label = instance)
                axs6[2].bar(X_axis-0.2 + diff, Y_COMPUTE_BWD_TIME, 0.2, label = instance)
                add_text(X_axis-0.25 + diff, Y_MEMCPY_TIME, axs6[0])
                add_text(X_axis-0.25 + diff, Y_COMPUTE_FWD_TIME, axs6[1])
                add_text(X_axis-0.25 + diff, Y_COMPUTE_BWD_TIME, axs6[2])

                axs7.bar(X_axis-0.2 + diff, Y_MEMCPY_TIME, 0.2, color = 'g', edgecolor='black')
                axs7.bar(X_axis-0.2 + diff, Y_COMPUTE_FWD_TIME, 0.2, bottom = Y_MEMCPY_TIME, color = 'b', edgecolor='black')
                axs7.bar(X_axis-0.2 + diff, Y_COMPUTE_BWD_TIME, 0.2, bottom = Y_COMPUTE_FWD_TIME, color = 'c', edgecolor='black')

                diff += 0.2

            axs1[0].set_xticks(X_axis)
            axs1[0].set_xticklabels(X, fontsize=20)
            axs1[0].set_xlabel("Models", fontsize=20)
            axs1[0].set_ylabel("Percentage", fontsize=20)
            axs1[0].set_title("Prep stall comparison", fontsize=20)
            axs1[0].legend(fontsize=20)

            axs1[1].set_xticks(X_axis)
            axs1[1].set_xticklabels(X, fontsize=20)
            axs1[1].set_xlabel("Models", fontsize=20)
            axs1[1].set_ylabel("Percentage", fontsize=20)
            axs1[1].set_title("Fetch stall comparison", fontsize=20)
            axs1[1].legend(fontsize=20)

            axs1[2].set_xticks(X_axis)
            axs1[2].set_xticklabels(X, fontsize=20)
            axs1[2].set_xlabel("Models", fontsize=20)
            axs1[2].set_ylabel("Percentage", fontsize=20)
            axs1[2].set_title("Interconnect stall comparison", fontsize=20)
            axs1[2].legend(fontsize=20)

            fig1.suptitle("Stall comparison - batch " + batch, fontsize=20, fontweight ="bold")
            fig1.savefig(result_dir + "/figures/stall_comparison_batch-" + batch + desc[desc_i])
    
            axs2[0].set_xticks(X_axis)
            axs2[0].set_xticklabels(X, fontsize=20)
            axs2[0].set_ylabel("Time", fontsize=20)
            axs2[0].set_title("Training time disk comparison", fontsize=20)
            axs2[0].legend(fontsize=20)

            axs2[1].set_xticks(X_axis)
            axs2[1].set_xticklabels(X, fontsize=20)
            axs2[1].set_ylabel("Time", fontsize=20)
            axs2[1].set_title("Training time cached comparison", fontsize=20)
            axs2[1].legend(fontsize=20)

            axs2[2].set_xticks(X_axis)
            axs2[2].set_xticklabels(X, fontsize=20)
            axs2[2].set_ylabel("Throughput", fontsize=20)
            axs2[2].set_title("Disk throughput comparison", fontsize=20)
            axs2[2].legend(fontsize=20)

            fig2.suptitle("Training time comparison - batch " + batch, fontsize=20, fontweight ="bold")
            fig2.savefig(result_dir + "/figures/training_time_batch-" + batch + desc[desc_i])

            axs8[0].set_xticks(X_axis)
            axs8[0].set_xticklabels(X, fontsize=20)
            axs8[0].set_ylabel("Dollar cost", fontsize=20)
            axs8[0].set_title("Training cost disk comparison", fontsize=20)
            axs8[0].legend(fontsize=20)

            axs8[1].set_xticks(X_axis)
            axs8[1].set_xticklabels(X, fontsize=20)
            axs8[1].set_ylabel("Dollar cost", fontsize=20)
            axs8[1].set_title("Training cost cached comparison", fontsize=20)
            axs8[1].legend(fontsize=20)

            fig8.suptitle("Training cost comparison - batch " + batch, fontsize=20, fontweight ="bold")
            fig8.savefig(result_dir + "/figures/training_cost_batch-" + batch + desc[desc_i])

            axs3[0].set_xticks(X_axis)
            axs3[0].set_xticklabels(X, fontsize=20)
            axs3[0].set_ylabel("Samples/sec", fontsize=20)
            axs3[0].set_title("Training speed ingestion comparison", fontsize=20)
            axs3[0].legend(fontsize=20)

            axs3[1].set_xticks(X_axis)
            axs3[1].set_xticklabels(X, fontsize=20)
            axs3[1].set_ylabel("Samples/sec", fontsize=20)
            axs3[1].set_title("Training speed disk comparison", fontsize=20)
            axs3[1].legend(fontsize=20)

            axs3[2].set_xticks(X_axis)
            axs3[2].set_xticklabels(X, fontsize=20)
            axs3[2].set_ylabel("Samples/sec", fontsize=20)
            axs3[2].set_title("Training speed cached comparison", fontsize=20)
            axs3[2].legend(fontsize=20)

            fig3.suptitle("Training speed comparison - batch " + batch, fontsize=20, fontweight ="bold")
            fig3.savefig(result_dir + "/figures/training_speed_batch-" + batch + desc[desc_i])

            axs4[0].set_xticks(X_axis)
            axs4[0].set_xticklabels(X, fontsize=20)
            axs4[0].set_ylabel("Average CPU utilization", fontsize=20)
            axs4[0].set_title("CPU utilization comparison", fontsize=20)
            axs4[0].legend(fontsize=20)

            axs4[1].set_xticks(X_axis)
            axs4[1].set_xticklabels(X, fontsize=20)
            axs4[1].set_ylabel("Average GPU utilization", fontsize=20)
            axs4[1].set_title("GPU utilization comparison", fontsize=20)
            axs4[1].legend(fontsize=20)

            axs4[2].set_xticks(X_axis)
            axs4[2].set_xticklabels(X, fontsize=20)
            axs4[2].set_ylabel("Average GPU memory utilization", fontsize=20)
            axs4[2].set_title("GPU memory utilization comparison", fontsize=20)
            axs4[2].legend(fontsize=20)

            fig4.suptitle("CPU and GPU utilization DISK comparison - batch " + batch, fontsize=20, fontweight ="bold")
            fig4.savefig(result_dir + "/figures/cpu_gpu_util_disk_batch-" + batch + desc[desc_i])

            axs5[0].set_xticks(X_axis)
            axs5[0].set_xticklabels(X, fontsize=20)
            axs5[0].set_ylabel("Average CPU utilization", fontsize=20)
            axs5[0].set_title("CPU utilization comparison", fontsize=20)
            axs5[0].legend(fontsize=20)

            axs5[1].set_xticks(X_axis)
            axs5[1].set_xticklabels(X, fontsize=20)
            axs5[1].set_ylabel("Average GPU utilization", fontsize=20)
            axs5[1].set_title("GPU utilization comparison", fontsize=20)
            axs5[1].legend(fontsize=20)

            axs5[2].set_xticks(X_axis)
            axs5[2].set_xticklabels(X, fontsize=20)
            axs5[2].set_ylabel("Average GPU memory utilization", fontsize=20)
            axs5[2].set_title("GPU memory utilization comparison", fontsize=20)
            axs5[2].legend(fontsize=20)

            fig5.suptitle("CPU and GPU utilization CACHED comparison - batch " + batch, fontsize=20, fontweight ="bold")
            fig5.savefig(result_dir + "/figures/cpu_gpu_util_cached_batch-" + batch + desc[desc_i])

            axs6[0].set_xticks(X_axis)
            axs6[0].set_xticklabels(X, fontsize=20)
            axs6[0].set_ylabel("Avg Total Time (Seconds)", fontsize=20)
            axs6[0].set_title("Memcpy time", fontsize=20)
            axs6[0].legend(fontsize=20)

            axs6[1].set_xticks(X_axis)
            axs6[1].set_xticklabels(X, fontsize=20)
            axs6[1].set_ylabel("Avg Total Time (Seconds)", fontsize=20)
            axs6[1].set_title("Fwd propogation compute time", fontsize=20)
            axs6[1].legend(fontsize=20)

            axs6[2].set_xticks(X_axis)
            axs6[2].set_xticklabels(X, fontsize=20)
            axs6[2].set_ylabel("Avg Total Time (Seconds)", fontsize=20)
            axs6[2].set_title("Bwd propogation compute time", fontsize=20)
            axs6[2].legend(fontsize=20)

            fig6.suptitle("Time comparison - batch " + batch, fontsize=20, fontweight ="bold")
            fig6.savefig(result_dir + "/figures/memcpy_compute_time_comparison_batch-" + batch + desc[desc_i])

            axs7.set_xticks(X_axis)
            axs7.set_xticklabels(X, fontsize=20)
            axs7.set_ylabel("Avg Total Time (Seconds)", fontsize=20)
            axs7.set_title("Stacked time comparison", fontsize=20)
            leg = ["Memcpy Time", "Fwd Propogation Time", "Bwd Propogation Time"]
            axs7.legend(leg, fontsize=20)

            fig7.suptitle("Time comparison - batch " + batch, fontsize=20, fontweight ="bold")
            fig7.savefig(result_dir + "/figures/stacked_time_comparison_batch-" + batch + desc[desc_i])

    #        plt.show()
            plt.close('all')

        desc_i += 1



def compare_models(result_dir):

    models = list(stats.keys())

    X = ["Disk Throughput", "Train speed", "Memory", "Page cache"]
    X_IO = ["Read Write", "IOWait"]
    X_BAT = ['Batch-'+ batch for batch in BATCH_SIZES]
    styles = ['r--', 'b--', 'g--']
    colors = [['green', 'red', 'blue'], ['orange', 'cyan', 'purple'], ['green', 'red', 'blue']]

    X_BAT_axis = np.arange(len(BATCH_SIZES))

    for model in models:

        fig3, axs3 = plt.subplots(1, 2, figsize=(30, 20))
        Y_GPU_UTIL_CACHED_PCT_LIST = [[None for i in range(len(BATCH_SIZES))] for j in range(len(instances))]
        Y_GPU_MEM_UTIL_CACHED_PCT_LIST = [[None for i in range(len(BATCH_SIZES))] for j in range(len(instances))]
        batch_i = 0

        for batch in BATCH_SIZES:
            max_dstat_len = 0
            max_nvidia_len = 0
            max_itrs = 0

            for instance in instances:
                gpu = gpu_map[instance]
                if gpu not in stats[model]:
                    del stats[model]
                    continue

                if "CPU_UTIL_DISK_LIST" not in stats[model][gpu][batch]:
                    stats[model][gpu][batch]["CPU_UTIL_DISK_LIST"] = []
                if "CPU_UTIL_CACHED_LIST" not in stats[model][gpu][batch]:
                    stats[model][gpu][batch]["CPU_UTIL_CACHED_LIST"] = []
                if "GPU_UTIL_DISK_LIST" not in stats[model][gpu][batch]:
                    stats[model][gpu][batch]["GPU_UTIL_DISK_LIST"] = []
                if "GPU_UTIL_CACHED_LIST" not in stats[model][gpu][batch]:
                    stats[model][gpu][batch]["GPU_UTIL_CACHED_LIST"] = []
                if "DATA_TIME_LIST" not in stats[model][gpu][batch]:
                    stats[model][gpu][batch]["DATA_TIME_LIST"] = []

                max_dstat_len = max(max_dstat_len, len(stats[model][gpu][batch]["CPU_UTIL_DISK_LIST"]))
                max_dstat_len = max(max_dstat_len, len(stats[model][gpu][batch]["CPU_UTIL_CACHED_LIST"]))
                max_nvidia_len = max(max_nvidia_len, len(stats[model][gpu][batch]["GPU_UTIL_DISK_LIST"]))
                max_nvidia_len = max(max_nvidia_len, len(stats[model][gpu][batch]["GPU_UTIL_CACHED_LIST"]))
                max_itrs = max(max_itrs, len(stats[model][gpu][batch]["DATA_TIME_LIST"]))

            fig1, axs1 = plt.subplots(3, 2, figsize=(30,20))
            fig2, axs2 = plt.subplots(3, 2, figsize=(30,20))

            X_dstat_axis = np.arange(max_dstat_len)
            X_nvidia_axis = np.arange(max_nvidia_len)
            X_metrics_axis = np.arange(len(X))
            X_metrics_io_axis = np.arange(len(X_IO))
            diff = 0
            idx = 0

            for instance in instances:

                gpu = gpu_map[instance]
                if gpu not in stats[model]:
                    stats[model][gpu][batch] = {}


                style = styles[idx]
                color = colors[idx]

                """
                if instance == "p2.xlarge":
                    style = 'r--'
                    color = ['green', 'red', 'blue']
                elif instance == "p3.2xlarge":
                    style = 'b--'
                    color = ['orange', 'cyan', 'purple']
                """

                overlapping = 0.50
        
                Y_METRICS_DISK = []
                Y_METRICS_CACHED = []
                Y_METRICS_IO_DISK = []
                Y_METRICS_IO_CACHED = []

                print(model, gpu, batch)

                Y_METRICS_DISK.append(stats[model][gpu][batch]["DISK_THR"] if "DISK_THR" in stats[model][gpu][batch] else 0)
                Y_METRICS_DISK.append(stats[model][gpu][batch]["TRAIN_SPEED_DISK"] if "DISK_THR" in stats[model][gpu][batch] else 0)
                Y_METRICS_DISK.append(stats[model][gpu][batch]["MEM_DISK"] if "DISK_THR" in stats[model][gpu][batch] else 0)
                Y_METRICS_DISK.append(stats[model][gpu][batch]["PCACHE_DISK"] if "DISK_THR" in stats[model][gpu][batch] else 0)
                Y_METRICS_IO_DISK.append(stats[model][gpu][batch]["READ_WRITE_DISK"] if "DISK_THR" in stats[model][gpu][batch] else 0)
                Y_METRICS_IO_DISK.append(stats[model][gpu][batch]["IO_WAIT_DISK"] if "DISK_THR" in stats[model][gpu][batch] else 0)

                Y_METRICS_CACHED.append(stats[model][gpu][batch]["DISK_THR"] if "DISK_THR" in stats[model][gpu][batch] else 0)
                Y_METRICS_CACHED.append(stats[model][gpu][batch]["TRAIN_SPEED_CACHED"] if "DISK_THR" in stats[model][gpu][batch] else 0)
                Y_METRICS_CACHED.append(stats[model][gpu][batch]["MEM_CACHED"] if "DISK_THR" in stats[model][gpu][batch] else 0)
                Y_METRICS_CACHED.append(stats[model][gpu][batch]["PCACHE_CACHED"] if "DISK_THR" in stats[model][gpu][batch] else 0)
                Y_METRICS_IO_CACHED.append(stats[model][gpu][batch]["READ_WRITE_CACHED"] if "DISK_THR" in stats[model][gpu][batch] else 0)
                Y_METRICS_IO_CACHED.append(stats[model][gpu][batch]["IO_WAIT_CACHED"] if "DISK_THR" in stats[model][gpu][batch] else 0)

                Y_CPU_UTIL_DISK = stats[model][gpu][batch]["CPU_UTIL_DISK_LIST"] if "DISK_THR" in stats[model][gpu][batch] else []
                Y_CPU_UTIL_CACHED = stats[model][gpu][batch]["CPU_UTIL_CACHED_LIST"] if "DISK_THR" in stats[model][gpu][batch] else []

                Y_GPU_UTIL_DISK = stats[model][gpu][batch]["GPU_UTIL_DISK_LIST"] if "DISK_THR" in stats[model][gpu][batch] else []
                Y_GPU_UTIL_CACHED = stats[model][gpu][batch]["GPU_UTIL_CACHED_LIST"] if "DISK_THR" in stats[model][gpu][batch] else []

                Y_GPU_MEM_UTIL_DISK = stats[model][gpu][batch]["GPU_MEM_UTIL_DISK_LIST"] if "DISK_THR" in stats[model][gpu][batch] else []
                Y_GPU_MEM_UTIL_CACHED = stats[model][gpu][batch]["GPU_MEM_UTIL_CACHED_LIST"] if "DISK_THR" in stats[model][gpu][batch] else []

                Y_IO_WAIT_LIST_DISK = stats[model][gpu][batch]["IO_WAIT_LIST_DISK"] if "DISK_THR" in stats[model][gpu][batch] else []
                Y_IO_WAIT_LIST_CACHED = stats[model][gpu][batch]["IO_WAIT_LIST_CACHED"] if "DISK_THR" in stats[model][gpu][batch] else []

                Y_DATA_TIME_LIST = stats[model][gpu][batch]["DATA_TIME_LIST"] if "DISK_THR" in stats[model][gpu][batch] else []
                Y_COMPUTE_TIME_FWD_LIST = stats[model][gpu][batch]["COMPUTE_TIME_FWD_LIST"] if "DISK_THR" in stats[model][gpu][batch] else []
                Y_COMPUTE_TIME_BWD_LIST = stats[model][gpu][batch]["COMPUTE_TIME_BWD_LIST"] if "DISK_THR" in stats[model][gpu][batch] else []

                Y_GPU_UTIL_CACHED_PCT_LIST[idx][batch_i] = stats[model][gpu][batch]["GPU_UTIL_CACHED_PCT"] if "GPU_UTIL_CACHED_PCT" in stats[model][gpu][batch] else 0
                Y_GPU_MEM_UTIL_CACHED_PCT_LIST[idx][batch_i] = stats[model][gpu][batch]["GPU_MEM_UTIL_CACHED_PCT"] if "GPU_MEM_UTIL_CACHED_PCT" in stats[model][gpu][batch] else 0
                
                if len(Y_CPU_UTIL_DISK) < max_dstat_len:
                    Y_CPU_UTIL_DISK.extend([0] * (max_dstat_len - len(Y_CPU_UTIL_DISK)))
                if len(Y_CPU_UTIL_CACHED) < max_dstat_len:
                    Y_CPU_UTIL_CACHED.extend([0] * (max_dstat_len - len(Y_CPU_UTIL_CACHED)))
                if len(Y_GPU_UTIL_DISK) < max_nvidia_len:
                    Y_GPU_UTIL_DISK.extend([0] * (max_nvidia_len - len(Y_GPU_UTIL_DISK)))
                if len(Y_GPU_UTIL_CACHED) < max_nvidia_len:
                    Y_GPU_UTIL_CACHED.extend([0] * (max_nvidia_len - len(Y_GPU_UTIL_CACHED)))
                if len(Y_GPU_MEM_UTIL_DISK) < max_nvidia_len:
                    Y_GPU_MEM_UTIL_DISK.extend([0] * (max_nvidia_len - len(Y_GPU_MEM_UTIL_DISK)))
                if len(Y_GPU_MEM_UTIL_CACHED) < max_nvidia_len:
                    Y_GPU_MEM_UTIL_CACHED.extend([0] * (max_nvidia_len - len(Y_GPU_MEM_UTIL_CACHED)))
                if len(Y_IO_WAIT_LIST_DISK) < max_dstat_len:
                    Y_IO_WAIT_LIST_DISK.extend([0] * (max_dstat_len - len(Y_IO_WAIT_LIST_DISK)))
                if len(Y_IO_WAIT_LIST_CACHED) < max_dstat_len:
                    Y_IO_WAIT_LIST_CACHED.extend([0] * (max_dstat_len - len(Y_IO_WAIT_LIST_CACHED)))
                if len(Y_DATA_TIME_LIST) < max_itrs:
                    Y_DATA_TIME_LIST.extend([0] * (max_itrs - len(Y_DATA_TIME_LIST)))
                if len(Y_COMPUTE_TIME_FWD_LIST) < max_itrs:
                    Y_COMPUTE_TIME_FWD_LIST.extend([0] * (max_itrs - len(Y_COMPUTE_TIME_FWD_LIST)))
                if len(Y_COMPUTE_TIME_BWD_LIST) < max_itrs:
                    Y_COMPUTE_TIME_BWD_LIST.extend([0] * (max_itrs - len(Y_COMPUTE_TIME_BWD_LIST)))

                axs1[0,0].bar(X_metrics_axis - 0.2 + diff, Y_METRICS_CACHED, 0.2, label = instance)
                axs1[0,1].plot(X_dstat_axis, Y_CPU_UTIL_CACHED, style, alpha=overlapping, label = instance)
                axs1[1,0].plot(X_nvidia_axis, Y_GPU_UTIL_CACHED, style, alpha=overlapping, label = instance)
                axs1[1,1].plot(X_nvidia_axis, Y_GPU_MEM_UTIL_CACHED, style, alpha=overlapping, label = instance)
                axs1[2,0].bar(X_metrics_io_axis -0.2 + diff, Y_METRICS_IO_CACHED, 0.2, label = instance)
                axs1[2,1].plot(X_dstat_axis, Y_IO_WAIT_LIST_CACHED, style, alpha=overlapping, label = instance)

                axs2[0,0].bar(X_metrics_axis - 0.2 + diff, Y_METRICS_DISK, 0.2, label = instance)
                axs2[0,1].plot(X_dstat_axis, Y_CPU_UTIL_DISK, style, alpha=overlapping, label = instance)
                axs2[1,0].plot(X_nvidia_axis, Y_GPU_UTIL_DISK, style, alpha=overlapping, label = instance)
                axs2[1,1].plot(X_nvidia_axis, Y_GPU_MEM_UTIL_DISK, style, alpha=overlapping, label = instance)
                axs2[2,0].bar(X_metrics_io_axis - 0.2 + diff, Y_METRICS_IO_DISK, 0.2, label = instance)
                axs2[2,1].plot(X_dstat_axis, Y_IO_WAIT_LIST_DISK, style, alpha=overlapping, label = instance)

                #axs3[0].plot(X_itrs_axis, Y_DATA_TIME_LIST, style, alpha=overlapping, label = instance)
                #axs3[1].plot(X_itrs_axis, Y_COMPUTE_TIME_FWD_LIST, style, alpha=overlapping, label = instance)
                #axs3[2].plot(X_itrs_axis, Y_COMPUTE_TIME_BWD_LIST, style, alpha=overlapping, label = instance)


                #axs3.bar(X_itrs_axis - 0.2 + diff, Y_DATA_TIME_LIST, 0.2, color = color[0])
                #axs3.bar(X_itrs_axis - 0.2 + diff, Y_COMPUTE_TIME_FWD_LIST, 0.2, bottom = Y_DATA_TIME_LIST, color = color[1])
                #axs3.bar(X_itrs_axis - 0.2 + diff, Y_COMPUTE_TIME_BWD_LIST, 0.2, bottom = Y_COMPUTE_TIME_FWD_LIST, color = color[2])

                diff += 0.2
                idx += 1

            axs1[0,0].set_xticks(X_metrics_axis)
            axs1[0,0].set_xticklabels(X, fontsize=20)
            axs1[0,0].set_xlabel("Metrics", fontsize=20)
            axs1[0,0].set_ylabel("Values", fontsize=20)
            axs1[0,0].set_title("Metric comparison cached", fontsize=20)
            axs1[0,0].legend(fontsize=20)

            axs1[0,1].set_xlabel("Time", fontsize=20)
            axs1[0,1].set_ylabel("Percentage", fontsize=20)
            axs1[0,1].set_title("CPU utilization comparison cached", fontsize=20)
            axs1[0,1].legend()

            axs1[1,0].set_xlabel("Time", fontsize=20)
            axs1[1,0].set_ylabel("Percentage", fontsize=20)
            axs1[1,0].set_title("GPU utilization comparison cached", fontsize=20)
            axs1[1,0].legend()

            axs1[1,1].set_xlabel("Time", fontsize=20)
            axs1[1,1].set_ylabel("Percentage", fontsize=20)
            axs1[1,1].set_title("GPU memory utilization comparison cached", fontsize=20)
            axs1[1,1].legend()

            axs1[2,0].set_xticks(X_metrics_io_axis)
            axs1[2,0].set_xticklabels(X_IO, fontsize=20)
            axs1[2,0].set_xlabel("Metrics", fontsize=20)
            axs1[2,0].set_ylabel("Values", fontsize=20)
            axs1[2,0].set_title("IO Metric comparison cached", fontsize=20)
            axs1[2,0].legend()

            axs1[2,1].set_xlabel("Time", fontsize=20)
            axs1[2,1].set_ylabel("Percentage", fontsize=20)
            axs1[2,1].set_title("IO wait percentage cached", fontsize=20)
            axs1[2,1].legend()

            fig1.suptitle("Cached comparison- " + model, fontsize=20, fontweight ="bold")
            fig1.savefig(result_dir + "/figures/cached_comparison-" + model + "_batch-" + batch)

            axs2[0,0].set_xticks(X_metrics_axis)
            axs2[0,0].set_xticklabels(X, fontsize=20)
            axs2[0,0].set_xlabel("Metrics", fontsize=20)
            axs2[0,0].set_ylabel("Values", fontsize=20)
            axs2[0,0].set_title("Metric comparison cached", fontsize=20)
            axs2[0,0].legend()

            axs2[0,1].set_xlabel("Time", fontsize=20)
            axs2[0,1].set_ylabel("Percentage", fontsize=20)
            axs2[0,1].set_title("CPU utilization comparison cached", fontsize=20)
            axs2[0,1].legend()

            axs2[1,0].set_xlabel("Time", fontsize=20)
            axs2[1,0].set_ylabel("Percentage", fontsize=20)
            axs2[1,0].set_title("GPU utilization comparison cached", fontsize=20)
            axs2[1,0].legend()

            axs2[1,1].set_xlabel("Time", fontsize=20)
            axs2[1,1].set_ylabel("Percentage", fontsize=20)
            axs2[1,1].set_title("GPU memory utilization comparison cached", fontsize=20)
            axs2[1,1].legend()

            axs2[2,0].set_xticks(X_metrics_io_axis)
            axs2[2,0].set_xticklabels(X_IO, fontsize=20)
            axs2[2,0].set_xlabel("Metrics", fontsize=20)
            axs2[2,0].set_ylabel("Values", fontsize=20)
            axs2[2,0].set_title("IO Metric comparison disk", fontsize=20)
            axs2[2,0].legend()

            axs2[2,1].set_xlabel("Time", fontsize=20)
            axs2[2,1].set_ylabel("Percentage", fontsize=20)
            axs2[2,1].set_title("io wait percentage disk", fontsize=20)
            axs2[2,1].legend()

            fig2.suptitle("Disk comparison - " + model, fontsize=20, fontweight ="bold")
            fig2.savefig(result_dir + "/figures/disk_comparison-" + model + "_batch-" + batch)

            batch_i += 1

        diff = 0
        for i in range(len(instances)):
            axs3[0].bar(X_BAT_axis - 0.2 + diff, Y_GPU_UTIL_CACHED_PCT_LIST[i], 0.2, label=instances[i])
            add_text(X_BAT_axis - 0.25 + diff, Y_GPU_UTIL_CACHED_PCT_LIST[i], axs3[0])
            axs3[1].bar(X_BAT_axis - 0.2 + diff, Y_GPU_MEM_UTIL_CACHED_PCT_LIST[i], 0.2, label=instances[i])
            add_text(X_BAT_axis - 0.25 + diff, Y_GPU_MEM_UTIL_CACHED_PCT_LIST[i], axs3[1])
            diff += 0.2

        axs3[0].set_xticks(X_BAT_axis)
        axs3[0].set_xticklabels(X_BAT, fontsize=20)
        axs3[0].set_xlabel("Batch size", fontsize=20)
        axs3[0].set_ylabel("Percentage", fontsize=20)
        axs3[0].set_title("GPU utilization", fontsize=20)
        axs3[0].legend()

        axs3[1].set_xticks(X_BAT_axis)
        axs3[1].set_xticklabels(X_BAT, fontsize=20)
        axs3[1].set_xlabel("Batch size", fontsize=20)
        axs3[1].set_ylabel("Percentage", fontsize=20)
        axs3[1].set_title("GPU memory utilization", fontsize=20)
        axs3[1].legend()

        fig3.suptitle("GPU utilization-" + model, fontsize=20, fontweight="bold")
        fig3.savefig(result_dir + "/figures/gpu_util_batch_compare-" + model)

        #plt.show()


def main():

    if len(sys.argv) <= 1:
        return

    result_dir = sys.argv[1]

    itr = 0
    for instance in sys.argv[2:]:
        instances.append(instance)
        result_path1 = result_dir + "/" + instance + "/" + "dali-gpu"
        result_path2 = result_dir + "/" + instance + "/" + "dali-cpu"

        for result_path in [result_path1, result_path2]:
            try:
                model_paths = [os.path.join(result_path, o) for o in os.listdir(result_path) if os.path.isdir(os.path.join(result_path,o))]
            except:
                continue

            for model_path in model_paths:
                model = model_path.split('/')[-1]
                batch_paths = [os.path.join(model_path, o) for o in os.listdir(model_path) if os.path.isdir(os.path.join(model_path,o))]
                for batch_path in batch_paths:
                    batch = batch_path.split('-')[-1]
                    gpu_paths = [os.path.join(batch_path, o) for o in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path,o))]
                    for gpu_path in gpu_paths:
                        #gpu = gpu_path.split('/')[-1] + str(itr)
                        gpu = gpu_path.split('/')[-1] 
                        cpu_paths = [os.path.join(gpu_path, o) for o in os.listdir(gpu_path) if os.path.isdir(os.path.join(gpu_path,o))]
                        for cpu_path in cpu_paths:
                            json_path = cpu_path + "/MODEL.json"
                            json_path2 = cpu_path + "/MODEL2.json"
                            if not os.path.isfile(json_path2):
                                continue

                            #process_json(model, gpu, json_path)
                            process_json2(model, instance, batch, json_path2)

                            csv_path = cpu_path + "/rank-0/run3-preprocess/"
                            process_csv(model, instance, batch, csv_path)
        itr += 1

    compare_instances(result_dir)
    compare_models(result_dir)


if __name__ == "__main__":
    main()


