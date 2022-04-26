import os
import sys

batch_sizes = {"alexnet":128,
                "resnet18":128,
                "shufflenet_v2_x0_5":128,
                "mobilenet_v2":64,
                "squeezenet1_0":64,
                "resnet50":64,
                "vgg11":64}
        
def main():

    if len(sys.argv) <= 1:
        return

    result_dir = sys.argv[1]

    for instance in sys.argv[2:]:
        result_path1 = result_dir + "/" + instance + "/" + "dali-gpu"
        result_path2 = result_dir + "/" + instance + "/" + "dali-cpu"

        for result_path in [result_path1, result_path2]:
            try:
                model_paths = [os.path.join(result_path, o) for o in os.listdir(result_path) if os.path.isdir(os.path.join(result_path,o))]
            except:
                continue

            for model_path in model_paths:
                model = model_path.split('/')[-1]
                os.chdir(model_path)
                os.system("mv jobs-1 batch_size-" + str(batch_sizes[model]))
                continue
                model_path_ = model_path + "/" + "batch_size-" + str(batch_sizes[model])
                gpu_paths = [os.path.join(model_path_, o) for o in os.listdir(model_path_) ]
                for gpu_path in gpu_paths:
                    #gpu = gpu_path.split('/')[-1] + str(itr)
                    gpu = gpu_path.split('/')[-1]
                    cpu_paths = [os.path.join(gpu_path, o) for o in os.listdir(gpu_path) if os.path.isdir(os.path.join(gpu_path,o))]
                    for cpu_path in cpu_paths:
                        rank_path = cpu_path + "/rank-0"
                        if not os.path.exists(rank_path):
                            os.chdir(cpu_path)
                            os.system("ln -s . rank-0")


if __name__ == "__main__":
    main()
