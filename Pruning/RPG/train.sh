# sparsity 80, 76.66%
python execimgnet.py --alpha 0.3 --batch_size 128 --delta 100 --distributed --distribution magnitude-exponential --epochs 100 --eval_batch_size 25 --lr 0.4 --sparsity 0.8 --lamb 0.5 --partial_k 0.2 --iterative_T_end_percent 0.9 --T_end_percent 0.901 --data ${DATA_DIR}
# sparsity 90, 75.80%
python execimgnet.py --alpha 0.3 --batch_size 128 --delta 100 --distributed --distribution magnitude-exponential --epochs 100 --eval_batch_size 25 --lr 0.4 --sparsity 0.9 --lamb 0.5 --partial_k 0.1 --iterative_T_end_percent 0.9 --T_end_percent 0.901 --data ${DATA_DIR}
# sparsity 95, 74.05%
python execimgnet.py --alpha 0.3 --batch_size 128 --delta 100 --distributed --distribution magnitude-exponential --epochs 100 --eval_batch_size 25 --lr 0.4 --sparsity 0.95 --lamb 3 --partial_k 0.2 --iterative_T_end_percent 0.9 --T_end_percent 0.901 --data ${DATA_DIR}
# sparsity 98, 69.57%
python execimgnet.py --alpha 0.3 --batch_size 128 --delta 100 --distributed --distribution magnitude-exponential --epochs 100 --eval_batch_size 25 --lr 0.4 --sparsity 0.98 --lamb 4 --partial_k 0.2 --iterative_T_end_percent 0.9 --T_end_percent 0.901 --data ${DATA_DIR}
