#function run
run() {
    number=$1
    shift
    for i in `seq $number`; do
      $@
    done
}

CUDA_VISIBLE_DEVICES=5 run 1 python main.py \
    --stable-diffusion-url http://127.0.0.1:7862 \
    --sd-train-method  testembed\
    --cls-n-epoch 5 \
    --scoring-n-epoch 5 \
    --validation-generated-images-per-class 50 \
    --embedding-steps-per-lr 2000 \
    --hypernetwork-steps-per-lr 1000 \