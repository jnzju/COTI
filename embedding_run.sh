#function run
run() {
    number=$1
    shift
    for i in `seq $number`; do
      $@
    done
}

CUDA_VISIBLE_DEVICES=7 run 1 python main.py \
    --task-num 6 \
    --stable-diffusion-url http://127.0.0.1:7860 \
    --sd-train-method embed \
    --cls-n-epoch 50 \
    --scoring-n-epoch 50 \
    --validation-generated-images-per-class 100 \