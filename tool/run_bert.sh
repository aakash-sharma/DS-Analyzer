python harness.py --nproc_per_node=8 -j 8 -a BERT -b 4 --synthetic_div_factor 1 --num_minibatches 678 --steps  RUN0 RUN1 --prefix results/BERT/p3.24xlarge BERT/bert_squad_dist_stash.py
