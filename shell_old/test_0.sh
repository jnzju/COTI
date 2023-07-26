#function run
run() {
    number=$1
    shift
    for i in `seq $number`; do
      $@
    done
}

CUDA_VISIBLE_DEVICES=5 run 1 python main.py \
    --stable-diffusion-url http://127.0.0.1:7856 \
    --categories axolotl frilled_lizard \
    --cls-n-epoch 50 \
    --scoring-n-epoch 50 \
    --scoring-batch-size 64 \
    --embedding-steps-per-lr 2000 \
    --save-embedding-every 100 \
    --embedding-learn-rate 0.0005 0.00025 0.000075 0.00005 0.000025 \
    --hypernetwork-steps-per-lr 1000 \
    --save-hypernetwork-every 50 \
    --hypernetwork-learn-rate 0.000005 0.0000025 0.00000075 0.0000005 0.00000025
