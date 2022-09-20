python harness.py --nproc_per_node=4 -j 8 -a BERT -b 4 --num_minibatches 50 --steps RUN0 RUN1 RUN2 --prefix results/test/ BERT/bert_squad_dist_stash.py
