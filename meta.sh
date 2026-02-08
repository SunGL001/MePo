python ./metaViT4_rep_noW.py \
    -gpu_ids '5,6' \
    -inner_lr 0.1 \
    -num_tasks 50 \
    -model 'ibot' \
    -epochs 50\
    -num_inner_steps 1 \
    -samples 200 &

python ./metaViT4_rep_noW.py \
    -gpu_ids '6,7' \
    -inner_lr 0.1 \
    -num_tasks 50 \
    -model 'sup1k' \
    -epochs 5\
    -num_inner_steps 1 \
    -samples 200 &




python ./metaViT4_rep_noW.py \
    -gpu_ids '0,1' \
    -inner_lr 0.1 \
    -num_tasks 50 \
    -model 'sup' \
    -epochs 50\
    -num_inner_steps 1 \
    -samples 200 &




python ./metaViT4_rep_noW.py \
    -gpu_ids '4,5' \
    -inner_lr 0.1 \
    -num_tasks 10 \
    -model 'sup' \
    -epochs 50\
    -num_inner_steps 1 \
    -samples 100 &




python ./metaViT4_rep_noW.py \
    -gpu_ids '4,5' \
    -inner_lr 0.1 \
    -num_tasks 20 \
    -model 'sup' \
    -epochs 50\
    -num_inner_steps 1 \
    -samples 100 &


python ./metaViT4_rep_noW.py \
    -gpu_ids '0,1' \
    -inner_lr 0.1 \
    -num_tasks 10 \
    -model 'sup' \
    -epochs 50\
    -num_inner_steps 1 \
    -samples 200 &




python ./metaViT4_rep.py \
    -gpu_ids '6,7' \
    -inner_lr 0.1 \
    -num_tasks 50 \
    -model 'sup' \
    -epochs 50\
    -num_inner_steps 1 \
    -samples 200 &




python ./metaViT4_rep_noW.py \
    -gpu_ids '6,7' \
    -inner_lr 0.1 \
    -num_tasks 10 \
    -model 'sup' \
    -epochs 50\
    -num_inner_steps 1 \
    -samples 100 &





