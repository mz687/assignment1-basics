ml pytorch/2.8.0

# token budget is fixed: 327_680_000.
# context length also fixed, but can change batch size and number of steps

python3 pretrain.py \
    --train-data-path /pscratch/sd/m/mzheng/cs336_data/tinystories_train.npy \
    --valid-data-path /pscratch/sd/m/mzheng/cs336_data/tinystories_valid.npy \
    --eval-interval 100 \
    --eval-batch-size 256 \
    --batch-size 128 \
    --train-iters 10_000 \
    --save './ckpt.pt' \
    --load './ckpt.pt' \
    --save-interval 5000 \
    --vocab-size 10_000 \
    --context-length 256 \
    --num-layers 4 \
    --num-heads 16 \
    --d-model 512 \
    --d-ff 1344 \
    --rope-theta 10000 \
    --lr 1.5e-4 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --min-lr 1e-5 \
    --warmup-iters 1000 \
    |& tee log.txt