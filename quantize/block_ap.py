# This file is modified based on the EfficientQAT project.

import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.fake_linear as int_linear_fake
from optim.soap import SOAP
from optim.sam import SAM
from quantize.quantizer import TanhRound
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math
from global_utils import NativeScalerWithGradNormCount
import pdb
import gc
from quantize.utils import (
    quant_parameters, weight_parameters, trainable_parameters,
    set_quant_state, quant_inplace, set_quant_parameters,
    set_weight_parameters, trainable_parameters_num, get_named_linears, set_op_by_name
)
import time
from datautil_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os


def get_hidden_states(output):
    if isinstance(output, torch.Tensor):
        return output
    elif isinstance(output, tuple):
        return output[0]
    elif hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    else:
        raise TypeError(f"Unsupported output type: {type(output)}")

def update_dataset(layer, dataset, dev, attention_mask, position_ids, position_embeddings):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)

                if inps.dim() == 2:
                    inps = inps.unsqueeze(0)

                out = layer(
                    inps,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )

                new_data = get_hidden_states(out)

                if new_data.dim() == 2:
                    new_data = new_data.unsqueeze(0)

                dataset.update_data(index, new_data.cpu())


def sam_train_step(
    qlayer,
    input,
    label,
    optimizer,
    loss_func,
    args,
    scaler,
    forward_kwargs,
    trainable_params,
):
    optimizer.zero_grad(set_to_none=True)

    # STEP1
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = qlayer(input, **forward_kwargs)
        quant_out = get_hidden_states(out)
    loss1 = loss_func(label.float(), quant_out.float())
    loss1.backward()

    # progressive
    if args.use_progressive:
        for m in qlayer.modules():
            if isinstance(m, int_linear_fake.QuantLinear):
                m.update_progressive_ratio()

    optimizer.first_step(zero_grad=True)

    # STEP2
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = qlayer(input, **forward_kwargs)
        quant_out = get_hidden_states(out)
    loss2 = loss_func(label.float(), quant_out.float())
    loss2.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_params, args.clip_grad
    )

    optimizer.second_step(zero_grad=False)

    optimizer.step()

    return loss2.detach(), grad_norm.detach()


import torch
import math

def normal_train_step(
    qlayer,
    input,
    label,
    optimizer,
    loss_func,
    args,
    loss_scaler,
    forward_kwargs,
    trainable_params,
):

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = qlayer(input, **forward_kwargs)
        quant_out = get_hidden_states(out)

    if torch.isnan(quant_out).any() or torch.isinf(quant_out).any():
        raise RuntimeError("NaN/Inf detected in quant_out BEFORE loss")

    quant_out = quant_out.float()
    label = label.float()
    loss = loss_func(label, quant_out)

    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Loss is NaN/Inf")

    optimizer.zero_grad(set_to_none=True)

    grad_norm = loss_scaler(
        loss,
        optimizer,
        clip_grad=args.clip_grad,
        parameters=trainable_params,
    )

    if getattr(args, "use_progressive", False):
        for m in qlayer.modules():
            if isinstance(m, int_linear_fake.QuantLinear):
                m.update_progressive_ratio()

    return loss.detach(), grad_norm.detach()



def block_ap(model, args, trainloader, valloader, logger=None):
    logger.info("Starting ...")

    if args.off_load_to_disk:
        logger.info("offload dataset to disk...")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    # ===== move first layer =====
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # ===== dataset =====
    flag = time.time()

    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/quant_val'
        for path in [fp_train_cache_path, fp_val_cache_path, quant_train_cache_path, quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = fp_val_cache_path = None
        quant_train_cache_path = quant_val_cache_path = None

    fp_train_inps = BlockTrainDataset(
        args.train_size, args.training_seqlen,
        model.config.hidden_size, args.batch_size, dtype,
        cache_path=fp_train_cache_path,
        off_load_to_disk=args.off_load_to_disk
    )

    fp_val_inps = BlockTrainDataset(
        args.val_size, args.training_seqlen,
        model.config.hidden_size, args.batch_size, dtype,
        cache_path=fp_val_cache_path,
        off_load_to_disk=args.off_load_to_disk
    )

    # ===== Catcher =====
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None
            self.position_embeddings = None

            # for qwen
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).cpu())
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            if self.position_embeddings is None:
                self.position_embeddings = kwargs.get("position_embeddings", None)
            raise ValueError

    # ===== catch train =====
    layers[0] = Catcher(layers[0], fp_train_inps)
    with torch.no_grad():
        for i in range(len(trainloader) // args.batch_size):
            data = torch.cat([trainloader[j][0] for j in range(i * args.batch_size, (i + 1) * args.batch_size)], dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    # ===== catch val =====
    layers[0] = Catcher(layers[0], fp_val_inps)
    with torch.no_grad():
        for i in range(len(valloader) // args.batch_size):
            data = torch.cat([valloader[j][0] for j in range(i * args.batch_size, (i + 1) * args.batch_size)], dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass

    attention_mask = layers[0].attention_mask
    position_ids = layers[0].position_ids
    position_embeddings = layers[0].position_embeddings
    layers[0] = layers[0].module

    attention_mask_batch = attention_mask.repeat(args.batch_size, 1, 1, 1).float() if attention_mask is not None else None

    # ===== move back =====
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.cpu()

    torch.cuda.empty_cache()

    # ===== init quant dataset =====
    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)

    # ===== training =====
    loss_func = torch.nn.MSELoss()

    for block_index in range(len(layers)):
        logger.info(f"=== Block {block_index} ===")

        layer = layers[block_index].to(dev)
        qlayer = copy.deepcopy(layer)

        # ===== replace linear =====
        for name, module in qlayer.named_modules():
            if isinstance(module, nn.Linear):
                ql = int_linear_fake.QuantLinear(module, args.wbits, args.group_size)

                if hasattr(args, "use_progressive") and args.use_progressive:
                    ql.enable_progressive(
                        target_bits=args.target_bits,
                        r_shape_mode=args.r_shape_mode,
                        rho=args.rho,
                        ema_momentum=args.ema_momentum,
                    )
                if hasattr(args, "use_soft_round") and args.use_soft_round:
                    round_fn = TanhRound(args.soft_round_t)

                    ql.weight_quantizer.set_round_fn(round_fn)

                    if ql.secondary_quantizer is not None:
                        ql.secondary_quantizer.set_round_fn(round_fn)

                set_op_by_name(qlayer, name, ql)
                del module

        qlayer.to(dev)

        # ===== FP outputs =====
        set_quant_state(qlayer, False)
        if args.epochs > 0:
            update_dataset(qlayer, fp_train_inps, dev, attention_mask, position_ids, position_embeddings)
            update_dataset(qlayer, fp_val_inps, dev, attention_mask, position_ids, position_embeddings)

        set_quant_state(qlayer, True)

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()

            param = []
            param_group_index = 0
            total_training_iteration = args.epochs * args.train_size / args.batch_size

            if args.quant_lr > 0:
                set_quant_parameters(qlayer, True)
                param.append({"params": quant_parameters(qlayer), "lr": args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(
                    empty_optimizer_1,
                    T_max=total_training_iteration,
                    eta_min=args.quant_lr / args.min_lr_factor
                )
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer, False)

            if args.weight_lr > 0:
                set_weight_parameters(qlayer, True)
                param.append({"params": weight_parameters(qlayer), "lr": args.weight_lr})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
                weight_scheduler = CosineAnnealingLR(
                    empty_optimizer_2,
                    T_max=total_training_iteration,
                    eta_min=args.weight_lr / args.min_lr_factor
                )
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer, False)

            # ===== optimizer =====
            if args.use_sam:
                optimizer = SAM(param, torch.optim.AdamW, rho=args.sam_rho, weight_decay=args.wd)
                scaler = torch.cuda.amp.GradScaler()
            else:
                optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
                loss_scaler = NativeScalerWithGradNormCount()

            trainable_params = list(trainable_parameters(qlayer))

            for epoch in range(args.epochs):
                loss_list = []
                norm_list = []

                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):

                    input = quant_inps.to(dev)
                    label = fp_inps.to(dev)

                    forward_kwargs = dict(
                        attention_mask=attention_mask_batch,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )

                    if args.use_sam:
                        loss, norm = sam_train_step(
                            qlayer,
                            input,
                            label,
                            optimizer,
                            loss_func,
                            args,
                            scaler,
                            forward_kwargs,
                            trainable_params,
                        )
                    else:
                        loss, norm = normal_train_step(
                            qlayer,
                            input,
                            label,
                            optimizer,
                            loss_func,
                            args,
                            loss_scaler,
                            forward_kwargs,
                            trainable_params,
                        )

                    if loss is None:
                        continue

                    loss_list.append(loss.cpu())
                    norm_list.append(norm.cpu())

                    # ===== LR schedule =====
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_last_lr()[0]

                    if args.weight_lr > 0:
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_last_lr()[0]

                avg_loss = torch.stack(loss_list).mean().item()
                avg_norm = torch.stack(norm_list).mean().item()

                if logger is not None:
                    logger.info(
                        f"[Block {block_index}] "
                        f"Epoch {epoch+1}/{args.epochs} | "
                        f"Train Loss: {avg_loss:.6f} | "
                        f"Grad Norm: {avg_norm:.4f}"
                    )
                else:
                    print(
                        f"[Block {block_index}] "
                        f"Epoch {epoch+1}/{args.epochs} | "
                        f"Train Loss: {avg_loss:.6f} | "
                        f"Grad Norm: {avg_norm:.4f}"
                    )

        # ===== inplace quant =====
        qlayer.half()
        quant_inplace(qlayer)
        set_quant_state(qlayer, False)

        # ===== update dataset =====
        if args.epochs > 0:
            update_dataset(qlayer, quant_train_inps, dev, attention_mask, position_ids, position_embeddings)
            update_dataset(qlayer, quant_val_inps, dev, attention_mask, position_ids, position_embeddings)

        layers[block_index] = qlayer.cpu()

        del layer
        torch.cuda.empty_cache()

    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = use_cache
    return model