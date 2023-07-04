"""
This is a test harness for profiling DNN training scripts 
to answer what-if questions on data stalls during training.

**How does this framework work?:**

This framework profiles the workload and collects statistics such as
    1. Avg per-iteration time ( MAX INGESTION RATE of the model)
    2. Avg per-iteration pre-processing stalls 
    3. Avg per-iteration data fetch stalls
    4. Optimal number of CPU per GPU to mask pre-processing stalls
    5. Avg disk throughput
    6. Available network bandwidth
    7. Optimal cache size

Our framework does this in a series of steps:

1. Train the model for a few iterations with synthetic data
   - Synchronize after cuda memcpy
   - Synchronize at iteration boundaries
   - Profiles the memcpy time and actual GPU time per iteration

2. Train the model for a few iterations with actual data
   with a cold cache
   - Synchronize after data get
   - SYnchronize at iteration boundaries
   - Profiles the actual GPU time and pre-processing + data fetch time
   
3. Train the model for a few iterations with actual data
   that is fully cached
   - Synchronize after data get
   - Synchronize at iteration boundaries
   - Profiles the actual GPU time and pre-processing time

MAX INGESTION RATE = Num_minibatches * Size per minibatch / (1)
PRE PROCESING RATE = Num_minibatches * Size per minibatch / (3)
DATA FETCH RATE    = Num_minibatches * Size per minibatch / ((2) - (3))
  
"""


import sys
import subprocess
import os
import utils
from argparse import ArgumentParser, REMAINDER
from synthetic_data import get_shared_image_classification_tensors
from utils import aggregate_run1_maps, print_as_table, print_header
import multiprocessing
import json
import statistics
import signal

def signal_handler(sig, frame):
    utils.stop_resource_profiling()

signal.signal(signal.SIGINT, signal_handler)


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch data stall profiler")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
    parser.add_argument("--use_env", default=False, action="store_true",
                        help="Use environment variable to pass "
                             "'local rank'. For legacy reasons, the default value is False. "
                             "If set to True, the script will not pass "
                             "--local_rank as argument, and will instead set LOCAL_RANK.")

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")


    # profiling
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                         help='number of data loading workers')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                         help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                         metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
    parser.add_argument('--synthetic', action='store_true',
                         help='Use synthetic dataset')
    parser.add_argument('--synthetic_div_factor', default=2, type=int, metavar='N',
                        help='division factor for synthetic data')
    parser.add_argument('--data-profile', action='store_true',
                         help='Set profiler on')  
    parser.add_argument('--precreate', action='store_true')
    parser.add_argument('--use_precreate', action='store_true')
    parser.add_argument("--classes", default=1000, type=int)
    parser.add_argument("--tensor_path", default="./train", type=str)
    parser.add_argument("--num_minibatches", default=50, type=int)
    parser.add_argument("--full_epoch", default=False, action='store_true')
    parser.add_argument("--resume_json", default=None, type=str)
    parser.add_argument("--resume_dir", default=None, type=str)
    parser.add_argument("--distributed", default=False, type=bool)
    parser.add_argument("--prefix", default="", type=str)
    parser.add_argument("--steps", default=["RUN0", "RUN1", "RUN2", "RUN3"], nargs='+')


    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

args = parse_args()

def run_synthetic_singleGPU(root_log_path):
    current_env = os.environ.copy()
    # world size in terms of number of processes
    dist_world_size = 1
    current_env["WORLD_SIZE"] = str(dist_world_size)

    if args.precreate:
        print("Precreating tensors in {}".format(args.tensor_path))
        if not os.path.exists(args.tensor_path):
            os.makedirs(args.tensor_path)
        procs = []
        for i in range(5):
            procs.append(multiprocessing.Process(target=get_shared_image_classification_tensors, args=(
                args.batch_size, int(args.num_minibatches/ (dist_world_size * 5)), i * int(args.num_minibatches / (dist_world_size * 5)), args.classes,
                args.tensor_path)))
            procs[i].start()

        for i in range(5):
            procs[i].join()

        args.use_precreate = True

    utils.start_resource_profiling()

    # spawn the processes
    if args.use_env:
        cmd = [sys.executable, "-u",
               args.training_script] + args.training_script_args
    elif args.use_precreate:
        cmd = [sys.executable,
               "-u",
               args.training_script,
               "--batch-size={}".format(args.batch_size),
               "--workers={}".format(args.workers),
               "--classes={}".format(args.classes),
               "--num_minibatches={}".format(args.num_minibatches // args.synthetic_div_factor),
               "--precreate",
               "--tensor_path={}".format(args.tensor_path),
               "--arch={}".format(args.arch),
               "--synthetic",
               "--epochs={}".format(args.epochs)] + args.training_script_args

    else:
        cmd = [sys.executable,
               "-u",
               args.training_script,
               "--batch-size={}".format(args.batch_size),
               "--workers={}".format(args.workers),
               "--classes={}".format(args.classes),
               "--num_minibatches={}".format(args.num_minibatches // args.synthetic_div_factor),
               "--arch={}".format(args.arch),
               "--synthetic",
               "--epochs={}".format(args.epochs)] + args.training_script_args

    process = subprocess.Popen(cmd, env=current_env)

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode,
                                            cmd=process.args)
#    log_path = os.getcwd() + "/" + args.prefix + "/" + args.arch + "/jobs-1" + "/gpus-1" + "/cpus-" + str(args.workers) + "/run0-synthetic_singleGPU/"
    log_path = root_log_path + "/rank-" + str(args.node_rank) + "/run0-synthetic_singleGPU/"

    res_dstat, res_free, res_nvidia = utils.stop_resource_profiling()
    utils.move_logs(log_path)
    print("FINISHED STEP 0 : SYNTHETIC WORKLOAD ON SINGLE GPU")
    return log_path, res_dstat, res_free, res_nvidia

def run_synthetic(root_log_path):
    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)


    if args.precreate:
        print("Precreating tensors in {}".format(args.tensor_path))
        if not os.path.exists(args.tensor_path):
            os.makedirs(args.tensor_path)
        procs = []
        for i in range(5):
            procs.append(multiprocessing.Process(target=get_shared_image_classification_tensors, args=(args.batch_size, int(args.num_minibatches/5),  i*int(args.num_minibatches/5), args.classes, args.tensor_path)))
            procs[i].start()

        for i in range(5):
            procs[i].join()
        
        args.use_precreate = True


    processes = []

    utils.start_resource_profiling()

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        if args.use_env:
            cmd = [sys.executable, "-u",
                   args.training_script] + args.training_script_args
        elif args.use_precreate:
            cmd = [sys.executable,
                   "-u",
                   args.training_script,
                   "--local_rank={}".format(local_rank), 
                   "--nnodes={}".format(args.nnodes),
                   "--node_rank={}".format(args.node_rank),
                   "--batch-size={}".format(args.batch_size),
                   "--workers={}".format(args.workers),
                   "--classes={}".format(args.classes),
                   "--num_minibatches={}".format(args.num_minibatches // args.synthetic_div_factor),
                   "--precreate",
                   "--tensor_path={}".format(args.tensor_path),
                   "--arch={}".format(args.arch),
                   "--synthetic",
                   "--epochs={}".format(args.epochs)] + args.training_script_args

        else:
            cmd = [sys.executable,
                   "-u",
                   args.training_script,
                   "--local_rank={}".format(local_rank), 
                   "--nnodes={}".format(args.nnodes),
                   "--node_rank={}".format(args.node_rank),
                   "--batch-size={}".format(args.batch_size),
                   "--workers={}".format(args.workers),
                   "--classes={}".format(args.classes),
                   "--num_minibatches={}".format(args.num_minibatches // args.synthetic_div_factor),
                   "--arch={}".format(args.arch),
                   "--synthetic",
                   "--epochs={}".format(args.epochs)] + args.training_script_args

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=process.args)
    #log_path = os.getcwd() + "/" + args.prefix + "/" +  args.arch + "/batch_size-" + str(args.batch_size) + "/gpus-" + str(dist_world_size) + "/cpus-" + str(args.workers) + "/rank-" + str(args.node_rank) + "/run1-synthetic/"
    log_path = root_log_path + "/rank-" + str(args.node_rank) + "/run1-synthetic/"


    res_dstat, res_free, res_nvidia = utils.stop_resource_profiling()
    utils.move_logs(log_path)
    print("FINISHED STEP 1 : SYNTHETIC WORKLOAD")
    return log_path, res_dstat, res_free, res_nvidia


def run_with_data(root_log_path, cached=False):
    dist_world_size = args.nproc_per_node * args.nnodes
    if not cached: 
        log_path = root_log_path + "/rank-" + str(args.node_rank) + "/run2-fetch-preprocess/"
    else:
        log_path = root_log_path + "/rank-" + str(args.node_rank) + "/run3-preprocess/"
      
    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    processes = []

    utils.start_resource_profiling()

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        if args.use_env:
            cmd = [sys.executable, "-u",
                   args.training_script] + args.training_script_args
        elif not args.full_epoch:
            cmd = [sys.executable,
                   "-u",
                   args.training_script,
                   "--local_rank={}".format(local_rank), 
                   "--nnodes={}".format(args.nnodes),
                   "--node_rank={}".format(args.node_rank),
                   "--batch-size={}".format(args.batch_size),
                   "--workers={}".format(args.workers),
                   "--classes={}".format(args.classes),
                   "--num_minibatches={}".format(args.num_minibatches),
                   "--arch={}".format(args.arch),
                   "--epochs={}".format(args.epochs)] + args.training_script_args
        else:
            cmd = [sys.executable,
                   "-u",
                   args.training_script,
                   "--local_rank={}".format(local_rank), 
                   "--nnodes={}".format(args.nnodes),
                   "--node_rank={}".format(args.node_rank),
                   "--batch-size={}".format(args.batch_size),
                   "--workers={}".format(args.workers),
                   "--classes={}".format(args.classes),
                   "--num_minibatches={}".format(args.num_minibatches),
                   "--full_epoch ",
                   "--arch={}".format(args.arch),
                   "--epochs={}".format(args.epochs)] + args.training_script_args

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=process.args)

    res_dstat, res_free, res_nvidia = utils.stop_resource_profiling()
    utils.move_logs(log_path)
    if not cached:
        print("FINISHED STEP 2 : PREPROCESS + FETCH ")
    else:
        print("FINISHED STEP 3 : PREPROCESS ONLY")
    return log_path, res_dstat, res_free, res_nvidia


def run_mem_test(cmd):
   process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
   (output,err)=process.communicate()
   exit_code = process.wait()
   return exit_code

def get_dataset_stats(dir_path):
   train_path = dir_path + "/train/"
   cmd = "du -sh " + train_path
   process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
   (output,err)=process.communicate()
   exit_code = process.wait()
   size = output.decode('utf-8').split()[0][:-1]
   metric = output.decode('utf-8').split()[0][-1]
   if str(metric) == "T":
       size = int(float(size)*1024)
 
   cmd = "find " + train_path + " -type f | wc -l"
   process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
   (output,err)=process.communicate()
   exit_code = process.wait()
   samples = output.decode('utf-8').split()[0]

   return size, samples

def run_stats_only(resume_path, local_gpus, num_nodes, steps):
    args.stats["LOCAL_GPUS"] = local_gpus
    args.stats["NUM_NODES"] = num_nodes
    num_gpu = local_gpus * num_nodes

    if "RUN0" in steps:
        run0_stats = []

        for rank in range(num_nodes):
            run0_path = resume_path + 'rank-' + str(rank) + '/run0-synthetic_singleGPU/'
            json_file = run0_path + 'profile-0.json'
            run0_stats.append(json.load(open(json_file)))

        if len(run0_stats) != num_nodes:
            print("Something went wrong in run0")
            sys.exit(1)

        args.stats["RUN0"], stddev_map, min_map, max_map = aggregate_run1_maps(run0_stats)
        args.stats["RUN0"]["SPEED"] = args.stats["RUN0"]["SAMPLES"] / args.stats["RUN0"]["COMPUTE"]
        args.stats["SPEED_INGESTION"] = args.stats["RUN0"]["SPEED"]
        args.stats["TRAIN_MIN"] = max_map["TRAIN"]
        args.stats["TRAIN_MAX"] = max_map["TRAIN"]
        
        if os.path.isfile(run0_path + 'all-utils.csv'):
            populate_run_stats("RUN0", run0_path)

        for value in list(stddev_map.values()):
            if value > 1:
                print("High STDDEV in values. Run for more minibatches for stable results")
                # sys.exit(1)

        #print_as_table(args.stats["RUN0"])

    if "RUN1" in steps:
        run1_stats = []

        for rank in range(num_nodes):
            run1_path = resume_path + 'rank-' + str(rank) + '/run1-synthetic/'
            for i in range(local_gpus):
                json_file = run1_path + 'profile-' + str(i) + '.json'
                run1_stats.append(json.load(open(json_file)))

        if len(run1_stats) != num_gpu:
            print("Something went wrong in run1")
            sys.exit(1)

        args.stats["RUN1"], stddev_map, min_map, max_map = aggregate_run1_maps(run1_stats)
        args.stats["RUN1"]["SPEED"] = args.stats["RUN1"]["SAMPLES"] / args.stats["RUN1"]["COMPUTE"]
        args.stats["SPEED_INGESTION"] = args.stats["RUN1"]["SPEED"]
        args.stats["RUN1"]["TRAIN_MIN"] = min_map["TRAIN"]
        args.stats["RUN1"]["TRAIN_MAX"] = max_map["TRAIN"]
        if os.path.isfile(run1_path + 'all-utils.csv'):
            populate_run_stats("RUN1", run1_path)

        for value in list(stddev_map.values()):
            if value > 1:
                print("High STDDEV in values. Run for more minibatches for stable results")
                # sys.exit(1)

        #print_as_table(args.stats["RUN1"])

    if "RUN2" in steps:
        run2_stats = []
        for rank in range(num_nodes):
            run2_path = resume_path + 'rank-' + str(rank) + '/run2-fetch-preprocess/'
            for i in range(local_gpus):
                json_file = run2_path + 'profile-' + str(i) + '.json'
                run2_stats.append(json.load(open(json_file)))

        if len(run2_stats) != num_gpu:
            print("Something went wrong in run1")
            sys.exit(1)

        args.stats["RUN2"], stddev_map, min_map, max_map = aggregate_run1_maps(run2_stats)
        populate_run_stats("RUN2", run2_path)
        args.stats["DISK_THR"] = args.stats["RUN2"]["READ"]
        args.stats["SPEED_DISK"] = args.stats["RUN2"]["SPEED"]
        args.stats["RUN2"]["GPU_UTIL_LIST"] = avg_list(args.stats["RUN2"]["GPU_UTIL_LIST"], num_gpu)
        args.stats["RUN2"]["GPU_MEM_UTIL_LIST"] = avg_list(args.stats["RUN2"]["GPU_MEM_UTIL_LIST"], num_gpu)

    if "RUN3" in steps:
        run3_stats = []
        for rank in range(num_nodes):
            run3_path = resume_path + 'rank-' + str(rank) + '/run3-preprocess/'
            for i in range(local_gpus):
                json_file = run3_path + 'profile-' + str(i) + '.json'
                run3_stats.append(json.load(open(json_file)))

        if len(run3_stats) != num_gpu:
            print("Something went wrong in run1")
            sys.exit(1)

        args.stats["RUN3"], stddev_map, min_map, max_map = aggregate_run1_maps(run3_stats)
        populate_run_stats("RUN3", run3_path)
        args.stats["SPEED_CACHED"] = args.stats["RUN3"]["SPEED"]
        args.stats["RUN3"]["GPU_UTIL_LIST"] = avg_list(args.stats["RUN3"]["GPU_UTIL_LIST"], num_gpu)
        args.stats["RUN3"]["GPU_MEM_UTIL_LIST"] = avg_list(args.stats["RUN3"]["GPU_MEM_UTIL_LIST"], num_gpu)

    json_outfile = resume_path + 'MODEL2.json'
    with open(json_outfile, 'w') as jf:
        json.dump(args.stats, jf)

def populate_run_stats(run, run_path):

    res_dstat = utils.parseDstat(run_path + 'all-utils.csv', True)
    res_nvidia = utils.parseNvidia(run_path + 'nvidia.csv', True)
    res_free = utils.parseFree(run_path + 'free.csv')
    idle, wait, read, write, recv, send, idle_list, wai_list, read_list, write_list, recv_list, send_list = res_dstat
    pmem, shm, page_cache, total = res_free
    gpu_util, gpu_mem_util, gpu_util_pct_list, gpu_mem_util_pct_list = res_nvidia

    args.stats[run]["SPEED"] = args.stats[run]["SAMPLES"] / args.stats[run]["TRAIN"]
    args.stats[run]["RECV"] = recv
    args.stats[run]["SEND"] = send
    args.stats[run]["READ"] = read
    args.stats[run]["WRITE"] = write
    args.stats[run]["IO_WAIT"] = wait
    args.stats[run]["CPU"] = 100 - idle
    args.stats[run]["MEM"] = pmem + shm
    args.stats[run]["PCACHE"] = page_cache
    args.stats[run]["GPU_UTIL"] = gpu_util
    args.stats[run]["GPU_MEM_UTIL"] = gpu_mem_util
    args.stats[run]["CPU_LIST"] = [100 - idle for idle in idle_list]
    args.stats[run]["GPU_UTIL_LIST"] = gpu_util_pct_list
    args.stats[run]["GPU_MEM_UTIL_LIST"] = gpu_mem_util_pct_list
    args.stats[run]["IO_WAIT_LIST"] = wai_list
    args.stats[run]["READ_LIST"] = read_list
    args.stats[run]["WRITE_LIST"] = write_list

def avg_list(gpu_list, num_gpus):
    new_list = []
    i = 0
    for j in range(num_gpus, len(gpu_list), num_gpus):
        new_list.append(statistics.mean(gpu_list[i:j]))
        i += num_gpus

    return new_list


def main():
    print_header(args)
    num_gpu = args.nproc_per_node * args.nnodes
    args.stats = {}
    resume = False
    steps = args.steps
    print(steps)
    if not (args.resume_json is None):
        resume = True
        print("Resuming from existing profile stats at {}".format(args.resume_json))
        if not os.path.exists(args.resume_json):
            print("Incorrect resume stat path")
            sys.exit(1)
        with open(args.resume_json, 'r') as jf:
            args.stats = json.load(jf)

    if not (args.resume_dir is None):
        resume = True
        print("Only calculating statistics from existing profile directory at {}".format(args.resume_dir))
        if not os.path.exists(args.resume_dir):
            print("Incorrect resume stat path")
            sys.exit(1)
        else:
            resume_path = args.resume_dir + "/" + args.arch + "/" + "/batch_size-" + \
                          str(args.batch_size) + "/gpus-" + str(num_gpu) + "/cpus-" + str(args.workers) + "/"
            run_stats_only(resume_path, args.nproc_per_node, args.nnodes, args.steps)
            sys.exit(0)
            

    root_log_path = os.getcwd() + "/" + args.prefix + "/" + args.arch + "/batch_size-" + str(args.batch_size) + "/gpus-" + str(num_gpu) +  "/cpus-" + str(args.workers)

    args.stats["LOCAL_GPUS"] = args.nproc_per_node
    args.stats["NUM_NODES"] = args.nnodes

    """
     JSON is of the following format : All times are total for BATCHES num batches
     1. MEMCPY - Time to memcpy the synthetic tensors to GPU DRAM
     2. DATA - Total time for a batch to be ready at GPU - Includes the memcpy time 
     3. COMPUTE - Total GPU computation time the batch   
     4. TRAIN - Total training time, sum of data and compute
     5. BATCHES - The number of batches whose collective times are shown above
     6. SAMPLES - Total samples used in this profiling phase
     All these numbers exclude any warmup batches specified 
    """

    # Stage 0 : Run with synthetic dataset on single GPU
    if not "RUN0" in args.steps:
        print("STEP0 is omitted")
    elif resume and 'RUN0' in args.stats:
        print_as_table(args.stats["RUN0"])
        print("STEP 0 already done. Continuing to step 1")

    else:
        try:
            log_path, res_dstat, res_free, res_nvidia = run_synthetic_singleGPU(root_log_path)
        except:
            utils.stop_resource_profiling()
            print("Exception in step 0")
            sys.exit(1)
        idle, wait, read, write, recv, send= res_dstat
        pmem, shm,page_cache, total = res_free
        gpu_util, gpu_mem_util = res_nvidia

        print("Parsing Step 0 results ...")
        run0_stats = []
        json_file = log_path + 'profile-0.json'
        run0_stats.append(json.load(open(json_file)))

        args.stats["RUN0"], stddev_map, min_map, max_map = aggregate_run1_maps(run0_stats)
        args.stats["RUN0"]["SPEED"] = args.stats["RUN0"]["SAMPLES"] / args.stats["RUN0"]["COMPUTE"]
        args.stats["SPEED_INGESTION"] = args.stats["RUN0"]["SPEED"]
        args.stats["RUN0"]["CPU"] = 100 - idle
        args.stats["RUN0"]["MEM"] = pmem + shm
        args.stats["RUN0"]["PCACHE"] = page_cache
        args.stats["RUN0"]["GPU_UTIL"] = gpu_util

        print_as_table(args.stats["RUN0"])

    # Stage 1 : Run with synthetic dataset
    if not "RUN1" in args.steps:
        print("STEP1 is omitted")
    elif resume and 'RUN1' in args.stats:
        print_as_table(args.stats["RUN1"])
        print("STEP 1 already done. Continuing to step 2")

    else:
        try:
            log_path, res_dstat, res_free, res_nvidia = run_synthetic(root_log_path)
        except:
            utils.stop_resource_profiling()
            print("Exception in step 1")
            sys.exit(1)
        idle, wait, read, write, recv, send= res_dstat
        pmem, shm,page_cache, total = res_free
        gpu_util, gpu_mem_util = res_nvidia

        print("Parsing Step 1 results ...")
        run1_stats = []
        local_gpus = args.nproc_per_node
        print('LOCAL GPUs ',local_gpus)

        for i in range(0,local_gpus):
            json_file = log_path + 'profile-' + str(i) + '.json'
            run1_stats.append(json.load(open(json_file)))
            
        if len(run1_stats) != local_gpus:
            print("Something went wrong in run1")
            sys.exit(1)
    
        args.stats["RUN1"], stddev_map, min_map, max_map = aggregate_run1_maps(run1_stats)
        args.stats["RUN1"]["SPEED"] = args.stats["RUN1"]["SAMPLES"]/args.stats["RUN1"]["COMPUTE"]
        args.stats["SPEED_INGESTION"] = args.stats["RUN1"]["SPEED"]
        args.stats["RUN1"]["RECV"] = recv
        args.stats["RUN1"]["SEND"] = send
        args.stats["RUN1"]["READ"] = read
        args.stats["RUN1"]["CPU"] = 100 - idle
        args.stats["RUN1"]["MEM"] = pmem + shm
        args.stats["RUN1"]["PCACHE"] = page_cache
        args.stats["RUN1"]["GPU_UTIL"] = gpu_util
        args.stats["RUN1"]["GPU_MEM_UTIL"] = gpu_mem_util

        for value in list(stddev_map.values()):
            if value > 1:
                print("High STDDEV in values. Run for more minibatches for stable results")
                #sys.exit(1)
        
        print_as_table(args.stats["RUN1"])

    # Stage 2 : Run with both fetch and pre-processing on 
    if not "RUN2" in args.steps:
        print("STEP2 is omitted")
    elif resume and 'RUN2' in args.stats:
        print_as_table(args.stats["RUN2"])
        print("STEP 2 already done. Continuing to step 3\n")
    else:
        #Drop cache here
        utils.clear_cache()

        try:
            log_path, res_dstat, res_free, res_nvidia = run_with_data(root_log_path)
        except:
            utils.stop_resource_profiling()
            print("Exception in step 2")
            sys.exit(1)

        idle, wait, read, write, recv, send= res_dstat
        pmem, shm,page_cache, total = res_free
        gpu_util, gpu_mem_util = res_nvidia

        print("\nParsing Step 2 results ...")
        run2_stats = []
        local_gpus = args.nproc_per_node
        print('LOCAL GPUs ',local_gpus)

        for i in range(0,local_gpus):
            json_file =  log_path + 'profile-' + str(i) + '.json'
            run2_stats.append(json.load(open(json_file)))
            
        if len(run2_stats) != local_gpus:
            print("Something went wrong in run1")
            sys.exit(1)
    
        args.stats["RUN2"], stddev_map, min_map, max_map = aggregate_run1_maps(run2_stats)
        args.stats["RUN2"]["SPEED"] = args.stats["RUN2"]["SAMPLES"]/args.stats["RUN2"]["TRAIN"]
        args.stats["RUN2"]["RECV"] = recv
        args.stats["RUN2"]["SEND"] = send
        args.stats["RUN2"]["READ"] = read
        args.stats["RUN2"]["CPU"] = 100 - idle
        args.stats["RUN2"]["MEM"] = pmem + shm
        args.stats["RUN2"]["PCACHE"] = page_cache
        args.stats["RUN2"]["GPU_UTIL"] = gpu_util
        args.stats["RUN2"]["GPU_MEM_UTIL"] = gpu_mem_util

        args.stats["DISK_THR"] = args.stats["RUN2"]["READ"]
        args.stats["SPEED_DISK"] = args.stats["RUN2"]["SPEED"]

        print_as_table(args.stats["RUN2"])

    # Stage 3 : Run with only pre-processing
    if not "RUN3" in args.steps:
        print("STEP3 is omitted")
    elif resume and 'RUN3' in args.stats:
        print_as_table(args.stats["RUN3"])
        print("STEP 3 already done. Continuing to step 4\n")
    else:
        try:
            log_path, res_dstat, res_free, res_nvidia = run_with_data(root_log_path, cached = True)
        except:
            utils.stop_resource_profiling()
            print("Exception in step 3")
            sys.exit(1)

        idle, wait, read, write, recv, send = res_dstat
        pmem, shm,page_cache, total = res_free
        gpu_util, gpu_mem_util = res_nvidia
        print("\nParsing Step 3 results ...")
        run3_stats = []
        local_gpus = args.nproc_per_node
        print('LOCAL GPUs ',local_gpus)

        for i in range(0,local_gpus):
            json_file =  log_path + 'profile-' + str(i) + '.json'
            run3_stats.append(json.load(open(json_file)))
            
        if len(run3_stats) != local_gpus:
            print("Something went wrong in run1")
            sys.exit(1)
    
        args.stats["RUN3"], stddev_map, min_map, max_map = aggregate_run1_maps(run3_stats)
        args.stats["RUN3"]["SPEED"] = args.stats["RUN3"]["SAMPLES"]/args.stats["RUN3"]["TRAIN"]
        args.stats["RUN3"]["RECV"] = recv
        args.stats["RUN3"]["SEND"] = send
        args.stats["RUN3"]["READ"] = read
        args.stats["RUN3"]["CPU"] = 100 - idle
        args.stats["RUN3"]["MEM"] = pmem + shm
        args.stats["RUN3"]["PCACHE"] = page_cache
        args.stats["RUN3"]["GPU_UTIL"] = gpu_util
        args.stats["RUN3"]["GPU_MEM_UTIL"] = gpu_mem_util

        args.stats["SPEED_CACHED"] = args.stats["RUN3"]["SPEED"]

        print_as_table(args.stats["RUN3"])

    # Stage 4 : Run with synthetic dataset and overlap
    if not "RUN4" in args.steps:
        print("STEP4 is omitted")
    elif not args.distributed or (resume and 'RUN4' in args.stats):
        #print_as_table(args.stats["RUN4"])
        print("STEP 4 already done")

    else:
        log_path = run_synthetic(False)

        print("Parsing Step 4 results ...")
        local_gpus = args.nproc_per_node
        print('LOCAL GPUs ', local_gpus)

        for i in range(0, local_gpus):
            json_file = log_path + 'profile-' + str(i) + '.json'
            run1_stats.append(json.load(open(json_file)))

        if len(run1_stats) != local_gpus:
            print("Something went wrong in run1")
            sys.exit(1)

        args.stats["RUN4"], stddev_map, min_map, max_map = aggregate_run1_maps(run1_stats)
        args.stats["RUN4"]["SPEED"] = args.stats["RUN4"]["SAMPLES"] / args.stats["RUN4"]["COMPUTE"]
        args.stats["SPEED_INGESTION"] = args.stats["RUN4"]["SPEED"]

        for value in list(stddev_map.values()):
            if value > 1:
                print("High STDDEV in values. Run for more minibatches for stable results")
                # sys.exit(1)

        print_as_table(args.stats["RUN4"])

    if resume and 'AVG_SAMPLE_SIZE' in args.stats:
        print("Datasets statistics already collected. Continuing to step 6\n")
    elif args.arch != "BERT":
        size, total_samples =  get_dataset_stats(args.training_script_args[-1])
        args.stats["AVG_SAMPLE_SIZE"] = int(size)
        args.stats["TOTAL_SAMPLES"] = int(total_samples)
        

    # Finally dump all stats to a json which can be queried later
    json_outfile = root_log_path + '/' + 'MODEL.json'
    with open(json_outfile, 'w') as jf:
        json.dump(args.stats, jf)

    utils.print_all(args.stats, expand=False)
    print("Done writing final JSON : {}".format(json_outfile))

if __name__ == "__main__":
    main()
