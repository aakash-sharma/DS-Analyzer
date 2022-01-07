import sys
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

stats = defaultdict(lambda: defaultdict(int))

def process_json(model, gpu, json_path):

    with open(json_path) as fd:
        dagJson = json.load(fd)

    stats[model][gpu] = {}

    stats[model][gpu]["SPEED_INGESTION"] = dagJson["SPEED_INGESTION"]
    stats[model][gpu]["SPEED_DISK"]  = dagJson["SPEED_DISK"]
    stats[model][gpu]["SPEED_CACHED"] = dagJson["SPEED_CACHED"]
    stats[model][gpu]["DISK_THR"] = dagJson["DISK_THR"]
    stats[model][gpu]["MEM_THR"] = dagJson["MEM_THR"]
    stats[model][gpu]["PREP_STALL"] = dagJson["RUN3"]["TRAIN"] - dagJson["RUN1"]["TRAIN"]
    stats[model][gpu]["FETCH_STALL"] = dagJson["RUN2"]["TRAIN"] - stats[model][gpu]["PREP_STALL"]
    stats[model][gpu]["MEM_THR"] = dagJson["MEM_THR"]
    

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


if __name__ == "__main__":
    main()


