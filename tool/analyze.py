import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import csv
import statistics
import glob
import xlwt
import xlrd

stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
BATCH_SIZES = ['32', '64', '80', '128', '256']
FONTSIZE = 20
BAR_MARGIN = 0
TEXT_MARGIN = 0.01

gpu_map = {
        "p2.xlarge" : "K80-1",
        "chameleon.xlarge" : "gpus-11",
        "p2.8xlarge" : "K80-8",
        "p2.8xlarge_2" : "K80-8_2",
        "p2.16xlarge" : "K80-16",
        "p3.2xlarge" : "V100-1",
        "p3.8xlarge" : "V100-4",
        "p3.8xlarge_2" : "V100-4_2",
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
batch_map = {}

def process_json(model, instance, batch, json_path):

    with open(json_path) as fd:
        dagJson = json.load(fd)

    stats[model][instance][batch]["TRAIN_SPEED_INGESTION"] = dagJson["SPEED_INGESTION"]
    stats[model][instance][batch]["TRAIN_SPEED_DISK"]  = dagJson["SPEED_DISK"]
    stats[model][instance][batch]["TRAIN_SPEED_CACHED"] = dagJson["SPEED_CACHED"]
    stats[model][instance][batch]["DISK_THR"] = dagJson["DISK_THR"]
#    stats[model][instance][batch]["MEM_THR"] = dagJson["MEM_THR"]
    stats[model][instance][batch]["TRAIN_TIME_INGESTION"] = dagJson["RUN1"]["TRAIN"]
    stats[model][instance][batch]["TRAIN_TIME_DISK"] = dagJson["RUN2"]["TRAIN"]
    stats[model][instance][batch]["TRAIN_TIME_CACHED"] = dagJson["RUN3"]["TRAIN"]
    stats[model][instance][batch]["CPU_UTIL_DISK_PCT"] = dagJson["RUN2"]["CPU"]
    stats[model][instance][batch]["CPU_UTIL_CACHED_PCT"] = dagJson["RUN3"]["CPU"]
    stats[model][instance][batch]["GPU_UTIL_DISK_PCT"] = dagJson["RUN2"]["GPU_UTIL"]
    stats[model][instance][batch]["GPU_UTIL_CACHED_PCT"] = dagJson["RUN3"]["GPU_UTIL"]
    stats[model][instance][batch]["GPU_MEM_UTIL_DISK_PCT"] = dagJson["RUN2"]["GPU_MEM_UTIL"]
    stats[model][instance][batch]["GPU_MEM_UTIL_CACHED_PCT"] = dagJson["RUN3"]["GPU_MEM_UTIL"]
    stats[model][instance][batch]["MEMCPY_TIME"] = dagJson["RUN1"]["MEMCPY"]
    stats[model][instance][batch]["COMPUTE_TIME"] = dagJson["RUN3"]["COMPUTE"]
    stats[model][instance][batch]["COMPUTE_BWD_TIME"] = dagJson["RUN3"]["COMPUTE_BWD"]
    stats[model][instance][batch]["COMPUTE_FWD_TIME"] = stats[model][instance][batch]["COMPUTE_TIME"] - stats[model][instance][batch]["COMPUTE_BWD_TIME"]

    stats[model][instance][batch]["PREP_STALL_TIME"] = dagJson["RUN3"]["TRAIN"] - dagJson["RUN1"]["TRAIN"]
    stats[model][instance][batch]["FETCH_STALL_TIME"] = dagJson["RUN2"]["TRAIN"] - stats[model][instance][batch]["PREP_STALL_TIME"]
    if "RUN0" in dagJson:
        stats[model][instance][batch]["INTERCONNECT_STALL_TIME"] = dagJson["RUN1"]["TRAIN"] - dagJson["RUN0"]["TRAIN"]

    stats[model][instance][batch]["PREP_STALL_PCT"] = stats[model][instance][batch]["PREP_STALL_TIME"] / stats[model][instance][batch]["TRAIN_TIME_CACHED"] * 100
    stats[model][instance][batch]["FETCH_STALL_PCT"] = stats[model][instance][batch]["FETCH_STALL_TIME"] / stats[model][instance][batch]["TRAIN_TIME_DISK"] * 100
    stats[model][instance][batch]["INTERCONNECT_STALL_PCT"] = stats[model][instance][batch]["INTERCONNECT_STALL_TIME"] / stats[model][instance][batch]["TRAIN_TIME_INGESTION"] * 100



def process_json2(model, instance, batch, json_path):

    gpu = gpu_map[instance]
    with open(json_path) as fd:
        dagJson = json.load(fd)

    if batch not in batch_map:
        batch_map[batch] = []
        batch_map[batch].append(instance)
    elif instance not in batch_map[batch]:
        batch_map[batch].append(instance)

    stats[model][instance][batch]["TRAIN_SPEED_INGESTION"] = dagJson["SPEED_INGESTION"]
    stats[model][instance][batch]["TRAIN_SPEED_DISK"] = dagJson["SPEED_DISK"]
    stats[model][instance][batch]["TRAIN_SPEED_CACHED"] = dagJson["SPEED_CACHED"]
    stats[model][instance][batch]["DISK_THR"] = dagJson["DISK_THR"]
    stats[model][instance][batch]["TRAIN_TIME_INGESTION"] = dagJson["RUN1"]["TRAIN"]
    stats[model][instance][batch]["TRAIN_TIME_DISK"] = dagJson["RUN2"]["TRAIN"]
    stats[model][instance][batch]["TRAIN_TIME_CACHED"] = dagJson["RUN3"]["TRAIN"]
    stats[model][instance][batch]["MEM_DISK"] = dagJson["RUN2"]["MEM"]
    stats[model][instance][batch]["PCACHE_DISK"] = dagJson["RUN2"]["PCACHE"]
    stats[model][instance][batch]["MEM_CACHED"] = dagJson["RUN3"]["MEM"]
    stats[model][instance][batch]["PCACHE_CACHED"] = dagJson["RUN3"]["PCACHE"]
    stats[model][instance][batch]["READ_WRITE_DISK"] = dagJson["RUN2"]["READ"] + dagJson["RUN2"]["WRITE"]
    stats[model][instance][batch]["IO_WAIT_DISK"] = dagJson["RUN2"]["IO_WAIT"]
    stats[model][instance][batch]["READ_WRITE_CACHED"] = dagJson["RUN3"]["READ"] + dagJson["RUN3"]["WRITE"]
    stats[model][instance][batch]["IO_WAIT_CACHED"] = dagJson["RUN3"]["IO_WAIT"]
    stats[model][instance][batch]["COST_DISK"] = stats[model][instance][batch]["TRAIN_TIME_DISK"] * cost_map[instance] / 3600
    stats[model][instance][batch]["COST_CACHED"] = stats[model][instance][batch]["TRAIN_TIME_CACHED"] * cost_map[instance] / 3600

    stats[model][instance][batch]["CPU_UTIL_DISK_PCT"] = dagJson["RUN2"]["CPU"]
    stats[model][instance][batch]["CPU_UTIL_CACHED_PCT"] = dagJson["RUN3"]["CPU"]
    stats[model][instance][batch]["GPU_UTIL_DISK_PCT"] = dagJson["RUN2"]["GPU_UTIL"]
    stats[model][instance][batch]["GPU_UTIL_CACHED_PCT"] = dagJson["RUN3"]["GPU_UTIL"]
    stats[model][instance][batch]["GPU_MEM_UTIL_DISK_PCT"] = dagJson["RUN2"]["GPU_MEM_UTIL"]
    stats[model][instance][batch]["GPU_MEM_UTIL_CACHED_PCT"] = dagJson["RUN3"]["GPU_MEM_UTIL"]

    stats[model][instance][batch]["MEMCPY_TIME"] = dagJson["RUN1"]["MEMCPY"]
    stats[model][instance][batch]["COMPUTE_TIME"] = dagJson["RUN3"]["COMPUTE"]
    stats[model][instance][batch]["COMPUTE_BWD_TIME"] = dagJson["RUN3"]["COMPUTE_BWD"]
    stats[model][instance][batch]["COMPUTE_FWD_TIME"] = stats[model][instance][batch]["COMPUTE_TIME"] - stats[model][instance][batch]["COMPUTE_BWD_TIME"]

    stats[model][instance][batch]["CPU_UTIL_DISK_LIST"] = dagJson["RUN2"]["CPU_LIST"]
    stats[model][instance][batch]["CPU_UTIL_CACHED_LIST"] = dagJson["RUN3"]["CPU_LIST"]
    stats[model][instance][batch]["GPU_UTIL_DISK_LIST"] = dagJson["RUN2"]["GPU_UTIL_LIST"]
    stats[model][instance][batch]["GPU_UTIL_CACHED_LIST"] = dagJson["RUN3"]["GPU_UTIL_LIST"]
    stats[model][instance][batch]["GPU_MEM_UTIL_DISK_LIST"] = dagJson["RUN2"]["GPU_MEM_UTIL_LIST"]
    stats[model][instance][batch]["GPU_MEM_UTIL_CACHED_LIST"] = dagJson["RUN3"]["GPU_MEM_UTIL_LIST"]
    stats[model][instance][batch]["READ_WRITE_LIST_DISK"] = dagJson["RUN2"]["READ_LIST"] + dagJson["RUN2"]["WRITE_LIST"]
    stats[model][instance][batch]["READ_WRITE_LIST_CACHED"] = dagJson["RUN3"]["READ_LIST"] + dagJson["RUN3"]["WRITE_LIST"]
    stats[model][instance][batch]["IO_WAIT_LIST_DISK"] = dagJson["RUN2"]["IO_WAIT_LIST"]
    stats[model][instance][batch]["IO_WAIT_LIST_CACHED"] = dagJson["RUN3"]["IO_WAIT_LIST"]

    stats[model][instance][batch]["PREP_STALL_TIME"] = dagJson["RUN3"]["TRAIN"] - dagJson["RUN1"]["TRAIN"]
    stats[model][instance][batch]["FETCH_STALL_TIME"] = dagJson["RUN2"]["TRAIN"] - stats[model][instance][batch]["PREP_STALL_TIME"]
    if "RUN0" in dagJson:
        stats[model][instance][batch]["INTERCONNECT_STALL_TIME"] = dagJson["RUN1"]["TRAIN"] - dagJson["RUN0"]["TRAIN"]
    else:
        stats[model][instance][batch]["INTERCONNECT_STALL_TIME"] = 0

    stats[model][instance][batch]["PREP_STALL_PCT"] = stats[model][instance][batch]["PREP_STALL_TIME"] / stats[model][instance][batch]["TRAIN_TIME_CACHED"] * 100
    stats[model][instance][batch]["FETCH_STALL_PCT"] = stats[model][instance][batch]["FETCH_STALL_TIME"] / stats[model][instance][batch]["TRAIN_TIME_DISK"] * 100
    stats[model][instance][batch]["INTERCONNECT_STALL_PCT"] = stats[model][instance][batch]["INTERCONNECT_STALL_TIME"] / stats[model][instance][batch]["TRAIN_TIME_INGESTION"] * 100
        
    if instance == "p3.16xlarge" and "V100-4_2" in stats[model]:
        if batch in stats[model]["V100-4_2"]:
            stats[model]["V100-4_2"][batch]["NETWORK_STALL_TIME"] = stats[model]["V100-4_2"][batch]["TRAIN_TIME_INGESTION"] - stats[model][instance][batch]["TRAIN_TIME_INGESTION"]
            stats[model]["V100-4_2"][batch]["NETWORK_STALL_PCT"] = stats[model]["V100-4_2"][batch]["INTERCONNECT_STALL_TIME"] / stats[model][instance][batch]["TRAIN_TIME_INGESTION"] * 100
        else:
            stats[model]["V100-4_2"][batch]["NETWORK_STALL_TIME"] = 0
            stats[model]["V100-4_2"][batch]["NETWORK_STALL_PCT"] = 0




def process_csv(model, instance, batch, csv_path):

    gpu = gpu_map[instance]
    stats[model][instance][batch]["DATA_TIME_LIST"] = []
    stats[model][instance][batch]["COMPUTE_TIME_LIST"] = []
    stats[model][instance][batch]["COMPUTE_TIME_FWD_LIST"] = []
    stats[model][instance][batch]["COMPUTE_TIME_BWD_LIST"] = []

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

        stats[model][instance][batch]["DATA_TIME_LIST"].append(statistics.mean(data_time))
        stats[model][instance][batch]["COMPUTE_TIME_LIST"].append(statistics.mean(compute_time))
        stats[model][instance][batch]["COMPUTE_TIME_FWD_LIST"].append(statistics.mean(compute_fwd_time))
        stats[model][instance][batch]["COMPUTE_TIME_BWD_LIST"].append(statistics.mean(compute_bwd_time))

def add_text(X, Y, axs):
    for idx, value in enumerate(X):
        axs.text(value - (0.1 * value), Y[idx] + (0.01 * Y[idx]), "{:.2f}".format(Y[idx]), fontsize=FONTSIZE)

def compare_instances(result_dir):

    X_small = ['alexnet', 'resnet18', 'shufflenet_v2_x0_5', 'mobilenet_v2', 'squeezenet1_0']
    X_large = ['resnet50', 'vgg11']
    desc = ["-Large_models", "-Small_models"]
    desc_i = 0

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    #matplotlib.rc('font', **font)
    plt.rcParams.update({'font.size': 22})

    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    for X in [X_large, X_small]:


        X_axis = np.arange(len(X))

        for batch in BATCH_SIZES:
            diff = 0

            fig1, axs1 = plt.subplots(2, 1, figsize=(30, 20))
            fig2, axs2 = plt.subplots(3, 1, figsize=(30, 20))
            fig3, axs3 = plt.subplots(3, 1, figsize=(30, 20))
            fig4, axs4 = plt.subplots(3, 1, figsize=(30, 20))
            fig5, axs5 = plt.subplots(3, 1, figsize=(30, 20))
            fig6, axs6 = plt.subplots(3, 1, figsize=(30, 20))
            fig7, axs7 = plt.subplots(figsize=(30, 20))
            fig8, axs8 = plt.subplots(2, 1, figsize=(30, 20))
            fig9, axs9 = plt.subplots(2, 1, figsize=(30, 20))

            for instance in instances:

                if instance not in batch_map[batch]:
                    continue

                gpu = gpu_map[instance]
                
                Y_PREP_STALL_PCT = [stats[model][instance][batch]["PREP_STALL_PCT"]
                                    if "PREP_STALL_PCT" in stats[model][instance][batch] else 0 for model in X]
                Y_FETCH_STALL_PCT = [stats[model][instance][batch]["FETCH_STALL_PCT"]
                                     if "FETCH_STALL_PCT" in stats[model][instance][batch] else 0 for model in X]
#                if not (instance == "p2.xlarge" or instance == "p3.2xlarge"):
                Y_INTERCONNECT_STALL_PCT = [stats[model][instance][batch]["INTERCONNECT_STALL_PCT"]
                                            if "INTERCONNECT_STALL_PCT" in stats[model][instance][batch] else 0 for model in X]

                Y_NETWORK_STALL_PCT = [stats[model][instance][batch]["NETWORK_STALL_PCT"]
                                            if "NETWORK_STALL_PCT" in stats[model][instance][batch] else 0 for model in X]

                Y_TRAIN_TIME_DISK = [stats[model][instance][batch]["TRAIN_TIME_DISK"]
                                     if "TRAIN_TIME_DISK" in stats[model][instance][batch] else 0 for model in X]
                Y_TRAIN_TIME_CACHED = [stats[model][instance][batch]["TRAIN_TIME_CACHED"]
                                       if "TRAIN_TIME_CACHED" in stats[model][instance][batch] else 0 for model in X]

                Y_COST_DISK = [stats[model][instance][batch]["COST_DISK"]
                               if "COST_DISK" in stats[model][instance][batch] else 0 for model in X]
                Y_COST_CACHED = [stats[model][instance][batch]["COST_CACHED"]
                                 if "COST_CACHED" in stats[model][instance][batch] else 0 for model in X]

                Y_DISK_THR = [stats[model][instance][batch]["DISK_THR"]
                              if "DISK_THR" in stats[model][instance][batch] else 0 for model in X]

                Y_TRAIN_SPEED_INGESTION = [stats[model][instance][batch]["TRAIN_SPEED_INGESTION"]
                                           if "TRAIN_SPEED_INGESTION" in stats[model][instance][batch] else 0 for model in X]
                Y_TRAIN_SPEED_DISK = [stats[model][instance][batch]["TRAIN_SPEED_DISK"]
                                      if "TRAIN_SPEED_DISK" in stats[model][instance][batch] else 0 for model in X]
                Y_TRAIN_SPEED_CACHED = [stats[model][instance][batch]["TRAIN_SPEED_CACHED"]
                                        if "TRAIN_SPEED_CACHED" in stats[model][instance][batch] else 0 for model in X]

                Y_CPU_UTIL_DISK_PCT = [stats[model][instance][batch]["CPU_UTIL_DISK_PCT"]
                                       if "CPU_UTIL_DISK_PCT" in stats[model][instance][batch] else 0 for model in X]
                Y_CPU_UTIL_CACHED_PCT = [stats[model][instance][batch]["CPU_UTIL_CACHED_PCT"]
                                         if "CPU_UTIL_CACHED_PCT" in stats[model][instance][batch] else 0 for model in X]

                Y_GPU_UTIL_DISK_PCT = [stats[model][instance][batch]["GPU_UTIL_DISK_PCT"]
                                       if "GPU_UTIL_DISK_PCT" in stats[model][instance][batch] else 0 for model in X]
                Y_GPU_UTIL_CACHED_PCT = [stats[model][instance][batch]["GPU_UTIL_CACHED_PCT"]
                                         if "GPU_UTIL_CACHED_PCT" in stats[model][instance][batch] else 0 for model in X]
                Y_GPU_MEM_UTIL_DISK_PCT = [stats[model][instance][batch]["GPU_MEM_UTIL_DISK_PCT"]
                                           if "GPU_MEM_UTIL_DISK_PCT" in stats[model][instance][batch] else 0 for model in X]
                Y_GPU_MEM_UTIL_CACHED_PCT = [stats[model][instance][batch]["GPU_MEM_UTIL_CACHED_PCT"]
                                             if "GPU_MEM_UTIL_CACHED_PCT" in stats[model][instance][batch] else 0 for model in X]

                Y_MEMCPY_TIME = [stats[model][instance][batch]["MEMCPY_TIME"]
                                 if "MEMCPY_TIME" in stats[model][instance][batch] else 0 for model in X]
                Y_COMPUTE_TIME = [stats[model][instance][batch]["COMPUTE_TIME"]
                                  if "COMPUTE_TIME" in stats[model][instance][batch] else 0 for model in X]
                Y_COMPUTE_FWD_TIME = [stats[model][instance][batch]["COMPUTE_FWD_TIME"]
                                      if "COMPUTE_FWD_TIME" in stats[model][instance][batch] else 0 for model in X]
                Y_COMPUTE_BWD_TIME = [stats[model][instance][batch]["COMPUTE_BWD_TIME"]
                                      if "COMPUTE_BWD_TIME" in stats[model][instance][batch] else 0 for model in X]

                axs1[0].bar(X_axis-BAR_MARGIN + diff, Y_PREP_STALL_PCT, 0.2, label=instance)
                axs1[1].bar(X_axis-BAR_MARGIN + diff, Y_FETCH_STALL_PCT, 0.2, label=instance)
                add_text(X_axis-TEXT_MARGIN + diff, Y_PREP_STALL_PCT, axs1[0])
                add_text(X_axis-TEXT_MARGIN + diff, Y_FETCH_STALL_PCT, axs1[1])

#                if not (instance == "p2.xlarge" or instance == "p3.2xlarge"):
                axs9[0].bar(X_axis-BAR_MARGIN + diff, Y_INTERCONNECT_STALL_PCT, 0.2, label=instance)
                axs9[1].bar(X_axis-BAR_MARGIN + diff, Y_NETWORK_STALL_PCT, 0.2, label=instance)
                add_text(X_axis-TEXT_MARGIN + diff, Y_INTERCONNECT_STALL_PCT, axs9[0])
                add_text(X_axis-TEXT_MARGIN + diff, Y_NETWORK_STALL_PCT, axs9[1])

                axs2[0].bar(X_axis-BAR_MARGIN + diff, Y_TRAIN_TIME_DISK, 0.2, label=instance)
                axs2[1].bar(X_axis-BAR_MARGIN + diff, Y_TRAIN_TIME_CACHED, 0.2, label=instance)
                axs2[2].bar(X_axis-BAR_MARGIN + diff, Y_DISK_THR, 0.2, label=instance)
                add_text(X_axis-TEXT_MARGIN + diff, Y_TRAIN_TIME_DISK, axs2[0])
                add_text(X_axis-TEXT_MARGIN + diff, Y_TRAIN_TIME_CACHED, axs2[1])
                add_text(X_axis-TEXT_MARGIN + diff, Y_DISK_THR, axs2[2])

                axs8[0].bar(X_axis-BAR_MARGIN + diff, Y_COST_DISK, 0.2, label=instance)
                axs8[1].bar(X_axis-BAR_MARGIN + diff, Y_COST_CACHED, 0.2, label=instance)
                add_text(X_axis-TEXT_MARGIN + diff, Y_COST_DISK, axs2[0])
                add_text(X_axis-TEXT_MARGIN + diff, Y_COST_CACHED, axs2[1])

                axs3[0].bar(X_axis-BAR_MARGIN + diff, Y_TRAIN_SPEED_INGESTION, 0.2, label = instance)
                axs3[1].bar(X_axis-BAR_MARGIN + diff , Y_TRAIN_SPEED_DISK, 0.2, label = instance)
                axs3[2].bar(X_axis-BAR_MARGIN + diff, Y_TRAIN_SPEED_CACHED, 0.2, label = instance)
                add_text(X_axis-TEXT_MARGIN + diff, Y_TRAIN_SPEED_INGESTION, axs3[0])
                add_text(X_axis-TEXT_MARGIN + diff, Y_TRAIN_SPEED_DISK, axs3[1])
                add_text(X_axis-TEXT_MARGIN + diff, Y_TRAIN_SPEED_CACHED, axs3[2])

                axs4[0].bar(X_axis-BAR_MARGIN + diff, Y_CPU_UTIL_DISK_PCT, 0.2, label = instance)
                axs4[1].bar(X_axis-BAR_MARGIN + diff, Y_GPU_UTIL_DISK_PCT, 0.2, label = instance)
                axs4[2].bar(X_axis-BAR_MARGIN + diff, Y_GPU_MEM_UTIL_DISK_PCT, 0.2, label = instance)
                add_text(X_axis-TEXT_MARGIN + diff, Y_CPU_UTIL_DISK_PCT, axs4[0])
                add_text(X_axis-TEXT_MARGIN + diff, Y_GPU_UTIL_DISK_PCT, axs4[1])
                add_text(X_axis-TEXT_MARGIN + diff, Y_GPU_MEM_UTIL_DISK_PCT, axs4[2])

                axs5[0].bar(X_axis-BAR_MARGIN + diff, Y_CPU_UTIL_CACHED_PCT, 0.2, label = instance)
                axs5[1].bar(X_axis-BAR_MARGIN + diff, Y_GPU_UTIL_CACHED_PCT, 0.2, label = instance)
                axs5[2].bar(X_axis-BAR_MARGIN + diff, Y_GPU_MEM_UTIL_CACHED_PCT, 0.2, label = instance)
                add_text(X_axis-TEXT_MARGIN + diff, Y_CPU_UTIL_CACHED_PCT, axs5[0])
                add_text(X_axis-TEXT_MARGIN + diff, Y_GPU_UTIL_CACHED_PCT, axs5[1])
                add_text(X_axis-TEXT_MARGIN + diff, Y_GPU_MEM_UTIL_CACHED_PCT, axs5[2])

                axs6[0].bar(X_axis-BAR_MARGIN + diff, Y_MEMCPY_TIME, 0.2, label = instance)
                axs6[1].bar(X_axis-BAR_MARGIN + diff, Y_COMPUTE_FWD_TIME, 0.2, label = instance)
                axs6[2].bar(X_axis-BAR_MARGIN + diff, Y_COMPUTE_BWD_TIME, 0.2, label = instance)
                add_text(X_axis-TEXT_MARGIN + diff, Y_MEMCPY_TIME, axs6[0])
                add_text(X_axis-TEXT_MARGIN + diff, Y_COMPUTE_FWD_TIME, axs6[1])
                add_text(X_axis-TEXT_MARGIN + diff, Y_COMPUTE_BWD_TIME, axs6[2])

                axs7.bar(X_axis-BAR_MARGIN + diff, Y_MEMCPY_TIME, 0.2, color = 'g', edgecolor='black')
                axs7.bar(X_axis-BAR_MARGIN + diff, Y_COMPUTE_FWD_TIME, 0.2, bottom = Y_MEMCPY_TIME, color = 'b', edgecolor='black')
                axs7.bar(X_axis-BAR_MARGIN + diff, Y_COMPUTE_BWD_TIME, 0.2, bottom = Y_COMPUTE_FWD_TIME, color = 'c', edgecolor='black')

                diff += 0.2

            axs1[0].set_xticks(X_axis)
            axs1[0].set_xticklabels(X, fontsize=FONTSIZE)
            axs1[0].set_xlabel("Models", fontsize=FONTSIZE)
            axs1[0].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs1[0].set_title("Prep stall comparison", fontsize=FONTSIZE)
            axs1[0].legend(fontsize=FONTSIZE)

            axs1[1].set_xticks(X_axis)
            axs1[1].set_xticklabels(X, fontsize=FONTSIZE)
            axs1[1].set_xlabel("Models", fontsize=FONTSIZE)
            axs1[1].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs1[1].set_title("Fetch stall comparison", fontsize=FONTSIZE)
            axs1[1].legend(fontsize=FONTSIZE)

            fig1.suptitle("Stall comparison - batch " + batch, fontsize=FONTSIZE, fontweight ="bold")
            fig1.savefig(result_dir + "/figures/stall_comparison_batch-" + batch + desc[desc_i])
    
            axs2[0].set_xticks(X_axis)
            axs2[0].set_xticklabels(X, fontsize=FONTSIZE)
            axs2[0].set_ylabel("Time", fontsize=FONTSIZE)
            axs2[0].set_title("Training time disk comparison", fontsize=FONTSIZE)
            axs2[0].legend(fontsize=FONTSIZE)

            axs2[1].set_xticks(X_axis)
            axs2[1].set_xticklabels(X, fontsize=FONTSIZE)
            axs2[1].set_ylabel("Time", fontsize=FONTSIZE)
            axs2[1].set_title("Training time cached comparison", fontsize=FONTSIZE)
            axs2[1].legend(fontsize=FONTSIZE)

            axs2[2].set_xticks(X_axis)
            axs2[2].set_xticklabels(X, fontsize=FONTSIZE)
            axs2[2].set_ylabel("Throughput", fontsize=FONTSIZE)
            axs2[2].set_title("Disk throughput comparison", fontsize=FONTSIZE)
            axs2[2].legend(fontsize=FONTSIZE)

            fig2.suptitle("Training time comparison - batch " + batch, fontsize=FONTSIZE, fontweight ="bold")
            fig2.savefig(result_dir + "/figures/training_time_batch-" + batch + desc[desc_i])

            axs8[0].set_xticks(X_axis)
            axs8[0].set_xticklabels(X, fontsize=FONTSIZE)
            axs8[0].set_ylabel("Dollar cost", fontsize=FONTSIZE)
            axs8[0].set_title("Training cost disk comparison", fontsize=FONTSIZE)
            axs8[0].legend(fontsize=FONTSIZE)

            axs8[1].set_xticks(X_axis)
            axs8[1].set_xticklabels(X, fontsize=FONTSIZE)
            axs8[1].set_ylabel("Dollar cost", fontsize=FONTSIZE)
            axs8[1].set_title("Training cost cached comparison", fontsize=FONTSIZE)
            axs8[1].legend(fontsize=FONTSIZE)

            fig8.suptitle("Training cost comparison - batch " + batch, fontsize=FONTSIZE, fontweight ="bold")
            fig8.savefig(result_dir + "/figures/training_cost_batch-" + batch + desc[desc_i])

            axs3[0].set_xticks(X_axis)
            axs3[0].set_xticklabels(X, fontsize=FONTSIZE)
            axs3[0].set_ylabel("Samples/sec", fontsize=FONTSIZE)
            axs3[0].set_title("Training speed ingestion comparison", fontsize=FONTSIZE)
            axs3[0].legend(fontsize=FONTSIZE)

            axs3[1].set_xticks(X_axis)
            axs3[1].set_xticklabels(X, fontsize=FONTSIZE)
            axs3[1].set_ylabel("Samples/sec", fontsize=FONTSIZE)
            axs3[1].set_title("Training speed disk comparison", fontsize=FONTSIZE)
            axs3[1].legend(fontsize=FONTSIZE)

            axs3[2].set_xticks(X_axis)
            axs3[2].set_xticklabels(X, fontsize=FONTSIZE)
            axs3[2].set_ylabel("Samples/sec", fontsize=FONTSIZE)
            axs3[2].set_title("Training speed cached comparison", fontsize=FONTSIZE)
            axs3[2].legend(fontsize=FONTSIZE)

            fig3.suptitle("Training speed comparison - batch " + batch, fontsize=FONTSIZE, fontweight ="bold")
            fig3.savefig(result_dir + "/figures/training_speed_batch-" + batch + desc[desc_i])

            axs4[0].set_xticks(X_axis)
            axs4[0].set_xticklabels(X, fontsize=FONTSIZE)
            axs4[0].set_ylabel("Average CPU utilization", fontsize=FONTSIZE)
            axs4[0].set_title("CPU utilization comparison", fontsize=FONTSIZE)
            axs4[0].legend(fontsize=FONTSIZE)

            axs4[1].set_xticks(X_axis)
            axs4[1].set_xticklabels(X, fontsize=FONTSIZE)
            axs4[1].set_ylabel("Average GPU utilization", fontsize=FONTSIZE)
            axs4[1].set_title("GPU utilization comparison", fontsize=FONTSIZE)
            axs4[1].legend(fontsize=FONTSIZE)

            axs4[2].set_xticks(X_axis)
            axs4[2].set_xticklabels(X, fontsize=FONTSIZE)
            axs4[2].set_ylabel("Average GPU memory utilization", fontsize=FONTSIZE)
            axs4[2].set_title("GPU memory utilization comparison", fontsize=FONTSIZE)
            axs4[2].legend(fontsize=FONTSIZE)

            fig4.suptitle("CPU and GPU utilization DISK comparison - batch " + batch, fontsize=FONTSIZE, fontweight ="bold")
            fig4.savefig(result_dir + "/figures/cpu_gpu_util_disk_batch-" + batch + desc[desc_i])

            axs5[0].set_xticks(X_axis)
            axs5[0].set_xticklabels(X, fontsize=FONTSIZE)
            axs5[0].set_ylabel("Average CPU utilization", fontsize=FONTSIZE)
            axs5[0].set_title("CPU utilization comparison", fontsize=FONTSIZE)
            axs5[0].legend(fontsize=FONTSIZE)

            axs5[1].set_xticks(X_axis)
            axs5[1].set_xticklabels(X, fontsize=FONTSIZE)
            axs5[1].set_ylabel("Average GPU utilization", fontsize=FONTSIZE)
            axs5[1].set_title("GPU utilization comparison", fontsize=FONTSIZE)
            axs5[1].legend(fontsize=FONTSIZE)

            axs5[2].set_xticks(X_axis)
            axs5[2].set_xticklabels(X, fontsize=FONTSIZE)
            axs5[2].set_ylabel("Average GPU memory utilization", fontsize=FONTSIZE)
            axs5[2].set_title("GPU memory utilization comparison", fontsize=FONTSIZE)
            axs5[2].legend(fontsize=FONTSIZE)

            fig5.suptitle("CPU and GPU utilization CACHED comparison - batch " + batch, fontsize=FONTSIZE, fontweight ="bold")
            fig5.savefig(result_dir + "/figures/cpu_gpu_util_cached_batch-" + batch + desc[desc_i])

            axs6[0].set_xticks(X_axis)
            axs6[0].set_xticklabels(X, fontsize=FONTSIZE)
            axs6[0].set_ylabel("Avg Total Time (Seconds)", fontsize=FONTSIZE)
            axs6[0].set_title("Memcpy time", fontsize=FONTSIZE)
            axs6[0].legend(fontsize=FONTSIZE)

            axs6[1].set_xticks(X_axis)
            axs6[1].set_xticklabels(X, fontsize=FONTSIZE)
            axs6[1].set_ylabel("Avg Total Time (Seconds)", fontsize=FONTSIZE)
            axs6[1].set_title("Fwd propogation compute time", fontsize=FONTSIZE)
            axs6[1].legend(fontsize=FONTSIZE)

            axs6[2].set_xticks(X_axis)
            axs6[2].set_xticklabels(X, fontsize=FONTSIZE)
            axs6[2].set_ylabel("Avg Total Time (Seconds)", fontsize=FONTSIZE)
            axs6[2].set_title("Bwd propogation compute time", fontsize=FONTSIZE)
            axs6[2].legend(fontsize=FONTSIZE)

            fig6.suptitle("Time comparison - batch " + batch, fontsize=FONTSIZE, fontweight ="bold")
            fig6.savefig(result_dir + "/figures/memcpy_compute_time_comparison_batch-" + batch + desc[desc_i])

            axs7.set_xticks(X_axis)
            axs7.set_xticklabels(X, fontsize=FONTSIZE)
            axs7.set_ylabel("Avg Total Time (Seconds)", fontsize=FONTSIZE)
            axs7.set_title("Stacked time comparison", fontsize=FONTSIZE)
            leg = ["Memcpy Time", "Fwd Propogation Time", "Bwd Propogation Time"]
            axs7.legend(leg, fontsize=FONTSIZE)

            fig7.suptitle("Time comparison - batch " + batch, fontsize=FONTSIZE, fontweight ="bold")
            fig7.savefig(result_dir + "/figures/stacked_time_comparison_batch-" + batch + desc[desc_i])

            axs9[0].set_xticks(X_axis)
            axs9[0].set_xticklabels(X, fontsize=FONTSIZE)
            axs9[0].set_xlabel("Models", fontsize=FONTSIZE)
            axs9[0].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs9[0].set_title("Interconnect stall comparison", fontsize=FONTSIZE)
            axs9[0].legend(fontsize=FONTSIZE)

            axs9[1].set_xticks(X_axis)
            axs9[1].set_xticklabels(X, fontsize=FONTSIZE)
            axs9[1].set_xlabel("Models", fontsize=FONTSIZE)
            axs9[1].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs9[1].set_title("Network stall comparison", fontsize=FONTSIZE)
            axs9[1].legend(fontsize=FONTSIZE)

            fig9.suptitle("Stall comparison - batch " + batch, fontsize=FONTSIZE, fontweight ="bold")
            fig9.savefig(result_dir + "/figures/stall_comparison2_batch-" + batch + desc[desc_i])



#            plt.show()
            plt.close('all')

        desc_i += 1



def compare_models(result_dir):

    models = list(stats.keys())

    X = ["Disk Throughput", "Train speed", "Memory", "Page cache"]
    X_IO = ["Read/Write", "IOWait"]
    X_BAT = ['Batch-'+ batch for batch in BATCH_SIZES]
    styles = ['r--', 'b--', 'g--', 'c--']
    colors = [['green', 'red', 'blue'], ['orange', 'cyan', 'purple'], ['green', 'red', 'blue']]

    X_BAT_axis = np.arange(len(BATCH_SIZES))

    for model in models:

        fig3, axs3 = plt.subplots(1, 2, figsize=(30, 20))
        fig4, axs4 = plt.subplots(1, 2, figsize=(30, 20))

        Y_GPU_UTIL_CACHED_PCT_LIST = [[None for i in range(len(BATCH_SIZES))] for j in range(len(instances))]
        Y_GPU_MEM_UTIL_CACHED_PCT_LIST = [[None for i in range(len(BATCH_SIZES))] for j in range(len(instances))]
        Y_COST_DISK_LIST = [[None for i in range(len(BATCH_SIZES))] for j in range(len(instances))]
        Y_COST_CACHED_LIST = [[None for i in range(len(BATCH_SIZES))] for j in range(len(instances))]

        """
        Y_GPU_UTIL_CACHED_PCT_LIST = []
        Y_GPU_MEM_UTIL_CACHED_PCT_LIST = []
        Y_COST_DISK_LIST = []
        Y_COST_CACHED_LIST = []

        for i in range(len(instances)):
            batches = []
            for j in range(len(BATCH_SIZES)):
                if instances[i] not in batch_map[BATCH_SIZES[j]]:
                    continue
                batches.append(None)
            
            Y_GPU_UTIL_CACHED_PCT_LIST.append(batches.copy())
            Y_GPU_MEM_UTIL_CACHED_PCT_LIST.append(batches.copy())
            Y_COST_DISK_LIST.append(batches.copy())
            Y_COST_CACHED_LIST.append(batches.copy())
        """

        batch_i = 0

        for batch in BATCH_SIZES:
            max_dstat_len = 0
            max_nvidia_len = 0
            max_itrs = 0

            for instance in instances:

                gpu = gpu_map[instance]
                if instance not in stats[model]:
                    del stats[model]
                    continue

                if "CPU_UTIL_DISK_LIST" not in stats[model][instance][batch]:
                    stats[model][instance][batch]["CPU_UTIL_DISK_LIST"] = []
                if "CPU_UTIL_CACHED_LIST" not in stats[model][instance][batch]:
                    stats[model][instance][batch]["CPU_UTIL_CACHED_LIST"] = []
                if "GPU_UTIL_DISK_LIST" not in stats[model][instance][batch]:
                    stats[model][instance][batch]["GPU_UTIL_DISK_LIST"] = []
                if "GPU_UTIL_CACHED_LIST" not in stats[model][instance][batch]:
                    stats[model][instance][batch]["GPU_UTIL_CACHED_LIST"] = []
                if "DATA_TIME_LIST" not in stats[model][instance][batch]:
                    stats[model][instance][batch]["DATA_TIME_LIST"] = []

                max_dstat_len = max(max_dstat_len, len(stats[model][instance][batch]["CPU_UTIL_DISK_LIST"]))
                max_dstat_len = max(max_dstat_len, len(stats[model][instance][batch]["CPU_UTIL_CACHED_LIST"]))
                max_nvidia_len = max(max_nvidia_len, len(stats[model][instance][batch]["GPU_UTIL_DISK_LIST"]))
                max_nvidia_len = max(max_nvidia_len, len(stats[model][instance][batch]["GPU_UTIL_CACHED_LIST"]))
                max_itrs = max(max_itrs, len(stats[model][instance][batch]["DATA_TIME_LIST"]))

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
                if instance not in stats[model]:
                    stats[model][instance][batch] = {}


                style = styles[idx]
                #color = colors[idx]

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

                print(model, instance, batch)

                Y_METRICS_DISK.append(stats[model][instance][batch]["DISK_THR"] if "DISK_THR" in stats[model][instance][batch] else 0)
                Y_METRICS_DISK.append(stats[model][instance][batch]["TRAIN_SPEED_DISK"] if "DISK_THR" in stats[model][instance][batch] else 0)
                Y_METRICS_DISK.append(stats[model][instance][batch]["MEM_DISK"] if "DISK_THR" in stats[model][instance][batch] else 0)
                Y_METRICS_DISK.append(stats[model][instance][batch]["PCACHE_DISK"] if "DISK_THR" in stats[model][instance][batch] else 0)
                Y_METRICS_IO_DISK.append(stats[model][instance][batch]["READ_WRITE_DISK"] if "READ_WRITE_DISK" in stats[model][instance][batch] else 0)
                Y_METRICS_IO_DISK.append(stats[model][instance][batch]["IO_WAIT_DISK"] if "IO_WAIT_DISK" in stats[model][instance][batch] else 0)

                Y_METRICS_CACHED.append(stats[model][instance][batch]["DISK_THR"] if "DISK_THR" in stats[model][instance][batch] else 0)
                Y_METRICS_CACHED.append(stats[model][instance][batch]["TRAIN_SPEED_CACHED"] if "DISK_THR" in stats[model][instance][batch] else 0)
                Y_METRICS_CACHED.append(stats[model][instance][batch]["MEM_CACHED"] if "DISK_THR" in stats[model][instance][batch] else 0)
                Y_METRICS_CACHED.append(stats[model][instance][batch]["PCACHE_CACHED"] if "DISK_THR" in stats[model][instance][batch] else 0)
                Y_METRICS_IO_CACHED.append(stats[model][instance][batch]["READ_WRITE_CACHED"] if "READ_WRITE_CACHED" in stats[model][instance][batch] else 0)
                Y_METRICS_IO_CACHED.append(stats[model][instance][batch]["IO_WAIT_CACHED"] if "IO_WAIT_CACHED" in stats[model][instance][batch] else 0)

                Y_CPU_UTIL_DISK = stats[model][instance][batch]["CPU_UTIL_DISK_LIST"] if "DISK_THR" in stats[model][instance][batch] else []
                Y_CPU_UTIL_CACHED = stats[model][instance][batch]["CPU_UTIL_CACHED_LIST"] if "DISK_THR" in stats[model][instance][batch] else []

                Y_GPU_UTIL_DISK = stats[model][instance][batch]["GPU_UTIL_DISK_LIST"] if "DISK_THR" in stats[model][instance][batch] else []
                Y_GPU_UTIL_CACHED = stats[model][instance][batch]["GPU_UTIL_CACHED_LIST"] if "DISK_THR" in stats[model][instance][batch] else []

                Y_GPU_MEM_UTIL_DISK = stats[model][instance][batch]["GPU_MEM_UTIL_DISK_LIST"] if "DISK_THR" in stats[model][instance][batch] else []
                Y_GPU_MEM_UTIL_CACHED = stats[model][instance][batch]["GPU_MEM_UTIL_CACHED_LIST"] if "DISK_THR" in stats[model][instance][batch] else []

                Y_IO_WAIT_LIST_DISK = stats[model][instance][batch]["IO_WAIT_LIST_DISK"] if "DISK_THR" in stats[model][instance][batch] else []
                Y_IO_WAIT_LIST_CACHED = stats[model][instance][batch]["IO_WAIT_LIST_CACHED"] if "DISK_THR" in stats[model][instance][batch] else []

                Y_DATA_TIME_LIST = stats[model][instance][batch]["DATA_TIME_LIST"] if "DISK_THR" in stats[model][instance][batch] else []
                Y_COMPUTE_TIME_FWD_LIST = stats[model][instance][batch]["COMPUTE_TIME_FWD_LIST"] if "DISK_THR" in stats[model][instance][batch] else []
                Y_COMPUTE_TIME_BWD_LIST = stats[model][instance][batch]["COMPUTE_TIME_BWD_LIST"] if "DISK_THR" in stats[model][instance][batch] else []

                Y_GPU_UTIL_CACHED_PCT_LIST[idx][batch_i] = stats[model][instance][batch]["GPU_UTIL_CACHED_PCT"] if "GPU_UTIL_CACHED_PCT" in stats[model][instance][batch] else 0
                Y_GPU_MEM_UTIL_CACHED_PCT_LIST[idx][batch_i] = stats[model][instance][batch]["GPU_MEM_UTIL_CACHED_PCT"] if "GPU_MEM_UTIL_CACHED_PCT" in stats[model][instance][batch] else 0

                Y_COST_DISK_LIST[idx][batch_i] = stats[model][instance][batch]["COST_DISK"] if "COST_DISK" in stats[model][instance][batch] else 0
                Y_COST_CACHED_LIST[idx][batch_i] = stats[model][instance][batch]["COST_CACHED"] if "COST_CACHED" in stats[model][instance][batch] else 0

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

                axs1[0,0].bar(X_metrics_axis - BAR_MARGIN + diff, Y_METRICS_CACHED, 0.2, label = instance)
                axs1[0,1].plot(X_dstat_axis, Y_CPU_UTIL_CACHED, style, alpha=overlapping, label = instance)
                axs1[1,0].plot(X_nvidia_axis, Y_GPU_UTIL_CACHED, style, alpha=overlapping, label = instance)
                axs1[1,1].plot(X_nvidia_axis, Y_GPU_MEM_UTIL_CACHED, style, alpha=overlapping, label = instance)
                axs1[2,0].bar(X_metrics_io_axis - BAR_MARGIN + diff, Y_METRICS_IO_CACHED, 0.2, label = instance)
                axs1[2,1].plot(X_dstat_axis, Y_IO_WAIT_LIST_CACHED, style, alpha=overlapping, label = instance)

                axs2[0,0].bar(X_metrics_axis - BAR_MARGIN + diff, Y_METRICS_DISK, 0.2, label = instance)
                axs2[0,1].plot(X_dstat_axis, Y_CPU_UTIL_DISK, style, alpha=overlapping, label = instance)
                axs2[1,0].plot(X_nvidia_axis, Y_GPU_UTIL_DISK, style, alpha=overlapping, label = instance)
                axs2[1,1].plot(X_nvidia_axis, Y_GPU_MEM_UTIL_DISK, style, alpha=overlapping, label = instance)
                axs2[2,0].bar(X_metrics_io_axis - BAR_MARGIN + diff, Y_METRICS_IO_DISK, 0.2, label = instance)
                axs2[2,1].plot(X_dstat_axis, Y_IO_WAIT_LIST_DISK, style, alpha=overlapping, label = instance)

                #axs3[0].plot(X_itrs_axis, Y_DATA_TIME_LIST, style, alpha=overlapping, label = instance)
                #axs3[1].plot(X_itrs_axis, Y_COMPUTE_TIME_FWD_LIST, style, alpha=overlapping, label = instance)
                #axs3[2].plot(X_itrs_axis, Y_COMPUTE_TIME_BWD_LIST, style, alpha=overlapping, label = instance)


                #axs3.bar(X_itrs_axis - BAR_MARGIN + diff, Y_DATA_TIME_LIST, 0.2, color = color[0])
                #axs3.bar(X_itrs_axis - BAR_MARGIN + diff, Y_COMPUTE_TIME_FWD_LIST, 0.2, bottom = Y_DATA_TIME_LIST, color = color[1])
                #axs3.bar(X_itrs_axis - BAR_MARGIN + diff, Y_COMPUTE_TIME_BWD_LIST, 0.2, bottom = Y_COMPUTE_TIME_FWD_LIST, color = color[2])

                diff += 0.2
                idx += 1

            axs1[0,0].set_xticks(X_metrics_axis)
            axs1[0,0].set_xticklabels(X, fontsize=FONTSIZE)
            axs1[0,0].set_xlabel("Metrics", fontsize=FONTSIZE)
            axs1[0,0].set_ylabel("Values", fontsize=FONTSIZE)
            axs1[0,0].set_title("Metric comparison cached", fontsize=FONTSIZE)
            axs1[0,0].legend(fontsize=FONTSIZE)

            axs1[0,1].set_xlabel("Time", fontsize=FONTSIZE)
            axs1[0,1].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs1[0,1].set_title("CPU utilization comparison cached", fontsize=FONTSIZE)
            axs1[0,1].legend()

            axs1[1,0].set_xlabel("Time", fontsize=FONTSIZE)
            axs1[1,0].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs1[1,0].set_title("GPU utilization comparison cached", fontsize=FONTSIZE)
            axs1[1,0].legend()

            axs1[1,1].set_xlabel("Time", fontsize=FONTSIZE)
            axs1[1,1].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs1[1,1].set_title("GPU memory utilization comparison cached", fontsize=FONTSIZE)
            axs1[1,1].legend()

            axs1[2,0].set_xticks(X_metrics_io_axis)
            axs1[2,0].set_xticklabels(X_IO, fontsize=FONTSIZE)
            axs1[2,0].set_xlabel("Metrics", fontsize=FONTSIZE)
            axs1[2,0].set_ylabel("Values", fontsize=FONTSIZE)
            axs1[2,0].set_title("IO Metric comparison cached", fontsize=FONTSIZE)
            axs1[2,0].legend()

            axs1[2,1].set_xlabel("Time", fontsize=FONTSIZE)
            axs1[2,1].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs1[2,1].set_title("IO wait percentage cached", fontsize=FONTSIZE)
            axs1[2,1].legend()

            fig1.suptitle("Cached comparison- " + model, fontsize=FONTSIZE, fontweight ="bold")
            fig1.savefig(result_dir + "/figures/cached_comparison-" + model + "_batch-" + batch)

            axs2[0,0].set_xticks(X_metrics_axis)
            axs2[0,0].set_xticklabels(X, fontsize=FONTSIZE)
            axs2[0,0].set_xlabel("Metrics", fontsize=FONTSIZE)
            axs2[0,0].set_ylabel("Values", fontsize=FONTSIZE)
            axs2[0,0].set_title("Metric comparison cached", fontsize=FONTSIZE)
            axs2[0,0].legend()

            axs2[0,1].set_xlabel("Time", fontsize=FONTSIZE)
            axs2[0,1].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs2[0,1].set_title("CPU utilization comparison cached", fontsize=FONTSIZE)
            axs2[0,1].legend()

            axs2[1,0].set_xlabel("Time", fontsize=FONTSIZE)
            axs2[1,0].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs2[1,0].set_title("GPU utilization comparison cached", fontsize=FONTSIZE)
            axs2[1,0].legend()

            axs2[1,1].set_xlabel("Time", fontsize=FONTSIZE)
            axs2[1,1].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs2[1,1].set_title("GPU memory utilization comparison cached", fontsize=FONTSIZE)
            axs2[1,1].legend()

            axs2[2,0].set_xticks(X_metrics_io_axis)
            axs2[2,0].set_xticklabels(X_IO, fontsize=FONTSIZE)
            axs2[2,0].set_xlabel("Metrics", fontsize=FONTSIZE)
            axs2[2,0].set_ylabel("Values", fontsize=FONTSIZE)
            axs2[2,0].set_title("IO Metric comparison disk", fontsize=FONTSIZE)
            axs2[2,0].legend()

            axs2[2,1].set_xlabel("Time", fontsize=FONTSIZE)
            axs2[2,1].set_ylabel("Percentage", fontsize=FONTSIZE)
            axs2[2,1].set_title("io wait percentage disk", fontsize=FONTSIZE)
            axs2[2,1].legend()

            fig2.suptitle("Disk comparison - " + model, fontsize=FONTSIZE, fontweight ="bold")
            fig2.savefig(result_dir + "/figures/disk_comparison-" + model + "_batch-" + batch)

            batch_i += 1

        diff = 0
        for i in range(len(instances)):
            if instances[i] not in batch_map[batch]:
                continue
            print(instances[i])
            print(Y_GPU_UTIL_CACHED_PCT_LIST[i])
            axs3[0].bar(X_BAT_axis -BAR_MARGIN + diff, Y_GPU_UTIL_CACHED_PCT_LIST[i], 0.2, label=instances[i])
            add_text(X_BAT_axis -TEXT_MARGIN + diff, Y_GPU_UTIL_CACHED_PCT_LIST[i], axs3[0])
            axs3[1].bar(X_BAT_axis -BAR_MARGIN + diff, Y_GPU_MEM_UTIL_CACHED_PCT_LIST[i], 0.2, label=instances[i])
            add_text(X_BAT_axis -TEXT_MARGIN + diff, Y_GPU_MEM_UTIL_CACHED_PCT_LIST[i], axs3[1])

            axs4[0].bar(X_BAT_axis -BAR_MARGIN + diff, Y_COST_DISK_LIST[i], 0.2, label=instances[i])
            add_text(X_BAT_axis -TEXT_MARGIN + diff, Y_COST_DISK_LIST[i], axs4[0])
            axs4[1].bar(X_BAT_axis -BAR_MARGIN + diff, Y_COST_CACHED_LIST[i], 0.2, label=instances[i])
            add_text(X_BAT_axis -TEXT_MARGIN + diff, Y_COST_CACHED_LIST[i], axs4[1])

            diff += 0.2

        axs3[0].set_xticks(X_BAT_axis)
        axs3[0].set_xticklabels(X_BAT, fontsize=FONTSIZE)
        axs3[0].set_xlabel("Batch size", fontsize=FONTSIZE)
        axs3[0].set_ylabel("Percentage", fontsize=FONTSIZE)
        axs3[0].set_title("GPU utilization", fontsize=FONTSIZE)
        axs3[0].legend(fontsize=FONTSIZE)

        axs3[1].set_xticks(X_BAT_axis)
        axs3[1].set_xticklabels(X_BAT, fontsize=FONTSIZE)
        axs3[1].set_xlabel("Batch size", fontsize=FONTSIZE)
        axs3[1].set_ylabel("Percentage", fontsize=FONTSIZE)
        axs3[1].set_title("GPU memory utilization", fontsize=FONTSIZE)
        axs3[1].legend(fontsize=FONTSIZE)

        fig3.suptitle("GPU utilization-" + model, fontsize=FONTSIZE, fontweight="bold")
        fig3.savefig(result_dir + "/figures/gpu_util_batch_compare-" + model)

        axs4[0].set_xticks(X_BAT_axis)
        axs4[0].set_xticklabels(X_BAT, fontsize=FONTSIZE)
        axs4[0].set_xlabel("Batch size", fontsize=FONTSIZE)
        axs4[0].set_ylabel("Dollar cost", fontsize=FONTSIZE)
        axs4[0].set_title("Training Cost Disk", fontsize=FONTSIZE)
        axs4[0].legend(fontsize=FONTSIZE)

        axs4[1].set_xticks(X_BAT_axis)
        axs4[1].set_xticklabels(X_BAT, fontsize=FONTSIZE)
        axs4[1].set_xlabel("Batch size", fontsize=FONTSIZE)
        axs4[1].set_ylabel("Dollar cost", fontsize=FONTSIZE)
        axs4[1].set_title("Training cost cached", fontsize=FONTSIZE)
        axs4[1].legend(fontsize=FONTSIZE)

        fig4.suptitle("Training Cost -" + model, fontsize=FONTSIZE, fontweight="bold")
        fig4.savefig(result_dir + "/figures/training_cost_batch_compare-" + model)

        plt.close('all')
        #plt.show()
        
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def dump_to_excel(result_dir):

    header_metrics = []
    header_list_metrics = []
    for model in stats:
        for instance in stats[model]:
            for batch in stats[model][instance]:
                for metric in stats[model][instance][batch].keys():
                    if "LIST" in metric:
                        header_list_metrics.append(metric)
                    else:
                        header_metrics.append(metric)
                break
            break
        break

    style = xlwt.XFStyle()
    style.num_format_str = '#,###0.00'

    for model in stats:
        workbook = xlwt.Workbook()
        for instance in stats[model]:
            row_list = []
            row_list.append(["Metric"] + header_metrics)
            row_list2 = []
            row_list2.append(["Metric"] + header_list_metrics)

            for batch in sorted(stats[model][instance], key=int):
                row = []
                row2 = []
                row.append('batch-' + batch)

                """
                for key in header_list_metrics:
                    if key not in stats[model][instance][batch]:
                        row2.append(0)
                    else:
                        row2.append(stats[model][instance][batch][key])
                """

                for key in header_metrics:
                    if key not in stats[model][instance][batch]:
                        row.append(0)
                    else:
                        row.append(stats[model][instance][batch][key])

                row_list.append(row.copy())

            worksheet = workbook.add_sheet(instance)
            i = 0
            for column in row_list:
                for item in range(len(column)):
                    value = column[item]
                    # print(value)
                    if value == None:
                        value = 0
                    if is_number(value):
                        # print(value)
                        worksheet.write(item, i, value, style=style)
                    else:
                        worksheet.write(item, i, value)
                i += 1
        workbook.save(result_dir + '/data_dump/' + model + '.xls')

    models = ['alexnet', 'resnet18', 'shufflenet_v2_x0_5', 'mobilenet_v2', 'squeezenet1_0', 'resnet50', 'vgg11']

    for instance in instances:
        workbook = xlwt.Workbook()
        for model in models:
            if instance not in stats[model]:
                continue
            row_list = []
            row_list.append(["Metric"] + header_metrics)
            for batch in sorted(stats[model][instance], key=int):
                row = []
                row.append('batch-' + batch)
                for key in header_metrics:
                    if key not in stats[model][instance][batch]:
                        row.append(0)
                    else:
                        row.append(stats[model][instance][batch][key])

                row_list.append(row.copy())

            worksheet = workbook.add_sheet(model)
            i = 0
            for column in row_list:
                for item in range(len(column)):
                    value = column[item]
                    # print(value)
                    if value == None:
                        value = 0
                    if is_number(value):
                        # print(value)
                        worksheet.write(item, i, value, style=style)
                    else:
                        worksheet.write(item, i, value)
                i += 1

        workbook.save(result_dir + '/data_dump/' + instance + '.xls')




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

    dump_to_excel(result_dir)
    compare_instances(result_dir)
    compare_models(result_dir)


if __name__ == "__main__":
    main()


