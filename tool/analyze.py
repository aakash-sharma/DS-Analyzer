import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


stats = defaultdict(lambda: defaultdict(dict))
stats2 = defaultdict(lambda: defaultdict(dict))

def process_json(model, gpu, json_path):

    with open(json_path) as fd:
        dagJson = json.load(fd)

#    stats[model][gpu] = {}
#    stats2[gpu][model] = {}

    stats[model][gpu]["SPEED_INGESTION"] = dagJson["SPEED_INGESTION"]
    stats[model][gpu]["SPEED_DISK"]  = dagJson["SPEED_DISK"]
    stats[model][gpu]["SPEED_CACHED"] = dagJson["SPEED_CACHED"]
    stats[model][gpu]["DISK_THR"] = dagJson["DISK_THR"]
    stats[model][gpu]["MEM_THR"] = dagJson["MEM_THR"]
    stats[model][gpu]["TRAIN_TIME"] = dagJson["RUN2"]["TRAIN"]
    stats[model][gpu]["PREP_STALL_TIME"] = dagJson["RUN3"]["TRAIN"] - dagJson["RUN1"]["TRAIN"]
    stats[model][gpu]["FETCH_STALL_TIME"] = dagJson["RUN2"]["TRAIN"] - stats[model][gpu]["PREP_STALL_TIME"]
    
    stats2[gpu][model]["SPEED_INGESTION"] = dagJson["SPEED_INGESTION"]
    stats2[gpu][model]["SPEED_DISK"]  = dagJson["SPEED_DISK"]
    stats2[gpu][model]["SPEED_CACHED"] = dagJson["SPEED_CACHED"]
    stats2[gpu][model]["DISK_THR"] = dagJson["DISK_THR"]
    stats2[gpu][model]["MEM_THR"] = dagJson["MEM_THR"]
    stats2[gpu][model]["TRAIN_TIME"] = dagJson["RUN2"]["TRAIN"]
    stats2[gpu][model]["PREP_STALL_TIME"] = dagJson["RUN3"]["TRAIN"] - dagJson["RUN1"]["TRAIN"]
    stats2[gpu][model]["FETCH_STALL_TIME"] = dagJson["RUN2"]["TRAIN"] - stats[model][gpu]["PREP_STALL_TIME"]


def plotModels():

    gpu = "gpus-1"
    X = [model for model in stats.keys()]
    print(X)
    #X_axis = [i for i in range(len(X))]
    X_axis = np.arange(len(X))

    Y_PREP_STALL_TIME = [stats[model][gpu]["PREP_STALL_TIME"] for model in X]
    Y_FETCH_STALL_TIME = [stats[model][gpu]["FETCH_STALL_TIME"] for model in X]
    Y_TRAIN_TIME = [stats[model][gpu]["TRAIN_TIME"] for model in X]

    plt.bar(X_axis-0.2, Y_TRAIN_TIME, 0.2, label = 'train time')
    plt.bar(X_axis, Y_PREP_STALL_TIME, 0.2, label = 'prep stall time')
    plt.bar(X_axis+0.2, Y_FETCH_STALL_TIME, 0.2, label = 'fetch stall time')

    plt.xticks(X_axis, X)
    plt.xlabel("Models")
    plt.ylabel("Time")
    plt.legend()
    plt.show()



def main():

    if len(sys.argv) <= 1:
        return

    result_path = sys.argv[1]

    model_paths = [os.path.join(result_path, o) for o in os.listdir(result_path) if os.path.isdir(os.path.join(result_path,o))]

    for model_path in model_paths:
        model = model_path.split('/')[-1]
        model_path_ = model_path + "/jobs-1"
        gpu_paths = [os.path.join(model_path_, o) for o in os.listdir(model_path_) if os.path.isdir(os.path.join(model_path_,o))]
        for gpu_path in gpu_paths:
            gpu = gpu_path.split('/')[-1]
            cpu_paths = [os.path.join(gpu_path, o) for o in os.listdir(gpu_path) if os.path.isdir(os.path.join(gpu_path,o))]
            for cpu_path in cpu_paths:
                json_path = cpu_path + "/MODEL.json"

                process_json(model, gpu, json_path)

    plotModels()


if __name__ == "__main__":
    main()


