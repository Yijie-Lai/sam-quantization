 CUDA_VISIBLE_DEVICES=7 python -m model_transfer.real_to_fake \                       
--model /share/MY-DAPO/e2e/qwen3-4b-w2g128-asam-pg/checkpoint-614 \
--save_dir /share/MY-DAPO/e2e/qwen3-4b-w2g128-asam-pg-fake \
--wbits 2 \            
--group_size 128