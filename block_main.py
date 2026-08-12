# This file is modified based on the EfficientQAT projects.

import os
import sys
import random
import numpy as np
import torch
import time
import global_utils as global_utils

from datautil_block import get_loaders, test_ppl
from quantize.block_ap import block_ap

from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from quantize.real_linear import load_quantized_model

from accelerate import infer_auto_device_map, dispatch_model

torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, tokenizer, args, logger):
    block_class_name = model.model.layers[0].__class__.__name__

    device_map = infer_auto_device_map(
        model,
        max_memory={i: args.max_memory for i in range(torch.cuda.device_count())},
        no_split_module_classes=[block_class_name],
    )

    model = dispatch_model(model, device_map=device_map)

    results = {}

    if args.eval_ppl:
        datasets = ["wikitext2"]  # keep original style, no c4
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)

        for dataset in ppl_results:
            logger.info(f"{dataset} perplexity: {ppl_results[dataset]:.2f}")

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table

        task_list = args.eval_tasks.split(",")

        model_eval = HFLM(pretrained=model, batch_size=args.eval_batch_size)

        task_manager = lm_eval.tasks.TaskManager()

        results = lm_eval.simple_evaluate(
            model=model_eval,
            tasks=task_list,
            num_fewshot=5,
            task_manager=task_manager,
        )

        logger.info(make_table(results))

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="Model name or path.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory for datasets.")
    parser.add_argument("--output_dir", type=str, default="./log/", help="Logging directory.")
    parser.add_argument("--save_quant_dir", type=str, default=None, help="Directory to save quantized model.")

    parser.add_argument("--real_quant", action="store_true", help="Use real quantization.")
    parser.add_argument("--resume_quant", type=str, default=None, help="Path to resume quantized model.")

    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        help=(
            "Calibration dataset. "
            "Options: wikitext2, redpajama, or sweep_xx (e.g., sweep_0.9)."
        ),
    )

    parser.add_argument("--train_size", type=int, default=8192, help="Number of training samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation samples.")
    parser.add_argument("--training_seqlen", type=int, default=2048, help="Sequence length for training.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs.")

    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="Sequence length for PPL evaluation.")

    parser.add_argument("--seed", type=int, default=2, help="Random seed.")

    parser.add_argument("--eval_ppl", action="store_true", help="Evaluate perplexity.")
    parser.add_argument("--eval_tasks", type=str, default="", help="LM-Eval tasks.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for eval tasks.")

    parser.add_argument("--wbits", type=int, default=4, help="Weight quantization bits.")
    parser.add_argument("--group_size", type=int, default=128, help="Group size for quantization.")

    parser.add_argument("--quant_lr", type=float, default=1e-4, help="LR for quant params.")
    parser.add_argument("--weight_lr", type=float, default=1e-5, help="LR for weight params.")
    parser.add_argument("--min_lr_factor", type=float, default=20, help="Min LR factor.")

    parser.add_argument("--clip_grad", type=float, default=0.1, help="Gradient clipping.")
    parser.add_argument("--wd", type=float, default=0, help="Weight decay.")

    parser.add_argument("--net", type=str, default=None, help="Model family name.")
    parser.add_argument("--max_memory", type=str, default="70GiB", help="Max GPU memory per device.")

    parser.add_argument("--early_stop", type=int, default=0, help="Early stopping patience.")
    parser.add_argument("--off_load_to_disk", action="store_true", help="Offload dataset to disk.")


    parser.add_argument("--use_progressive", action="store_true",
                        help="Enable progressive quantization.")
    parser.add_argument("--target_bits", type=int, default=2,
                        help="Target bit-width for progressive quantizer.")
    parser.add_argument("--r_shape_mode", type=str, default="scalar",
                        choices=["scalar", "channel", "group"],
                        help="Shape of progressive ratio.")
    parser.add_argument("--rho", type=float, default=0.05,
                        help="Scaling factor for progressive update.")
    parser.add_argument("--ema_momentum", type=float, default=0.05,
                        help="EMA smoothing for progressive ratio.")
    parser.add_argument(
        "--progressive_finalize_epochs", type=int, default=1,
        help="Number of final block epochs trained at the exact target bit-width.",
    )
    
    # ===== SAM =====
    parser.add_argument("--use_sam", action="store_true",
                        help="Enable SAM (Sharpness-Aware Minimization).")

    parser.add_argument("--sam_rho", type=float, default=0.05,
                        help="Perturbation radius for SAM (epsilon scale).")
  
    # ===== Rho Search =====
    parser.add_argument(
        "--rho_search",
        action="store_true",
        help="Search the best rho before block-wise training."
    )

    parser.add_argument(
        "--rho_grid",
        type=str,
        default="0.001,0.002,0.005,0.01,0.02,0.05",
        help="Comma-separated rho candidates."
    )

    parser.add_argument(
        "--rho_search_train_steps",
        type=int,
        default=32,
        help="Number of training batches for each rho candidate."
    )

    parser.add_argument(
        "--rho_search_eval_steps",
        type=int,
        default=32,
        help="Number of validation batches for rho evaluation."
    )

    parser.add_argument("--auto_rho", action="store_true")
    
    # ===== Soft Round =====
    parser.add_argument("--use_soft_round", action="store_true",
                        help="Enable soft rounding instead of STE.")

    parser.add_argument("--soft_round_type", type=str, default="tanh",
                        choices=["tanh"],
                        help="Soft rounding surrogate type.")

    parser.add_argument("--soft_round_t", type=float, default=12,
                        help="Temperature controlling softness of surrogate gradient.")
    
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "nlc"],
    )


    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parser.parse_args()

    if args.use_progressive and args.wbits <= args.target_bits:
        parser.error(
            "Progressive quantization requires --wbits greater than --target_bits "
            "(for example, --wbits 8 --target_bits 2)."
        )
    if args.progressive_finalize_epochs < 0:
        parser.error("--progressive_finalize_epochs must be non-negative")
    if args.progressive_finalize_epochs > args.epochs:
        parser.error("--progressive_finalize_epochs cannot exceed --epochs")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)

    logger = global_utils.create_logger(Path(args.output_dir))
    logger.info(args)

    if args.net is None:
        args.net = args.model.split("/")[-1]
        logger.info(f"net is None, setting as {args.net}")

    if args.resume_quant:
        model, tokenizer = load_quantized_model(
            args.resume_quant, args.wbits, args.group_size
        )
    else:
        config = AutoConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=False, legacy=False
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            device_map="cpu",
            torch_dtype=torch.float16,
        )

        for param in model.parameters():
            param.requires_grad = False

        if args.wbits < 16:

            logger.info("=== start quantization ===")
            tick = time.time()

            cache_train = f'{args.cache_dir}/dataloader_random_window_v1_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
            cache_val = f'{args.cache_dir}/dataloader_random_window_v1_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'

            if os.path.exists(cache_train) and os.path.exists(cache_val):
                trainloader = torch.load(cache_train)
                logger.info(f"load trainloader from {cache_train}")
                valloader = torch.load(cache_val)
                logger.info(f"load valloader from {cache_val}")
            else:
                trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                )
                torch.save(trainloader, cache_train)
                torch.save(valloader, cache_val)

            block_ap(
                model,
                args,
                trainloader,
                valloader,
                logger,
            )

            logger.info(f"quant time: {time.time() - tick:.2f}s")

    torch.cuda.empty_cache()

    if args.save_quant_dir:
        logger.info("start saving model")
        model.save_pretrained(args.save_quant_dir)
        tokenizer.save_pretrained(args.save_quant_dir)
        logger.info("save model success")

    evaluate(model, tokenizer, args, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()