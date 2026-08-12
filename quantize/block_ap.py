# This file is modified based on the EfficientQAT project.

import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.fake_linear as int_linear_fake
import quantize.real_linear as int_linear_real
# from optim.asam import ASAM
from optim.autosam import ASAM
# from optim.lightsam import LightSAMAdam
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
    layer = layer.float()

    with torch.no_grad():
        for index, inps in enumerate(dataset):
            inps = inps.to(dev).float()

            if inps.dim() == 2:
                inps = inps.unsqueeze(0)

            out = layer(
                inps,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )

            new_data = get_hidden_states(out).float()

            if not torch.isfinite(new_data).all():
                pdb.set_trace()


            if new_data.dim() == 2:
                new_data = new_data.unsqueeze(0)

            dataset.update_data(index, new_data.cpu())


class MSEPlusNLCLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=2)
        self.eps = eps

    def forward(self, target, pred):
        target = target.float()
        pred = pred.float()

        loss1 = self.mse(target, pred)
        cos = self.cos(pred, target).mean().abs().clamp(min=self.eps)
        loss2 = -torch.log(cos)

        return loss1 + loss2

def get_rho_candidates(args):
    # --rho_grid "0.001,0.002,0.005,0.01"
    rho_grid = getattr(args, "rho_grid", None)

    if rho_grid is None:
        base = args.sam_rho
        return [
            base * 0.25,
            base * 0.5,
            base,
            base * 2.0,
        ]

    if isinstance(rho_grid, str):
        return [float(x) for x in rho_grid.split(",") if x.strip()]

    return [float(x) for x in rho_grid]


@torch.no_grad()
def eval_block_loss(
    qlayer,
    quant_val_inps,
    fp_val_inps,
    loss_func,
    dev,
    forward_kwargs,
    max_batches=5,
):
    qlayer.eval()
    losses = []

    for idx, (quant_inps, fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):
        if idx >= max_batches:
            break

        input = quant_inps.to(dev)
        label = fp_inps.to(dev)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = qlayer(input, **forward_kwargs)
            quant_out = get_hidden_states(out)

        if not torch.isfinite(quant_out).all():
            continue

        loss = loss_func(label.float(), quant_out.float())
        if torch.isfinite(loss):
            losses.append(loss.detach().float().cpu())

    qlayer.train()

    if len(losses) == 0:
        return float("inf")

    return torch.stack(losses).mean().item()


def search_block_rho(
    qlayer,
    param,
    args,
    loss_func,
    quant_train_inps,
    fp_train_inps,
    quant_val_inps,
    fp_val_inps,
    dev,
    forward_kwargs,
    trainable_params,
    logger=None,
):
    candidates = get_rho_candidates(args)

    best_rho = args.sam_rho
    best_loss = float("inf")

    # restore qlayer after every trial
    base_state = copy.deepcopy(qlayer.state_dict())

    for rho in candidates:
        qlayer.load_state_dict(base_state, strict=True)
        qlayer.train()

        optimizer = ASAM(
            param,
            torch.optim.AdamW,
            rho=float(rho),
            adaptive=True,
            weight_decay=args.wd,
        )
        scaler = torch.cuda.amp.GradScaler()

        for idx, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):
            if idx >= args.rho_search_train_steps:
                break

            input = quant_inps.to(dev)
            label = fp_inps.to(dev)

            sam_train_step_v3(
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

        val_loss = eval_block_loss(
            qlayer,
            quant_val_inps,
            fp_val_inps,
            loss_func,
            dev,
            forward_kwargs,
            max_batches=args.rho_search_eval_steps,
        )

        if logger is not None:
            logger.info(f"[Rho Search] rho={float(rho):.6g} | val loss={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_rho = float(rho)

        del optimizer, scaler
        torch.cuda.empty_cache()

    qlayer.load_state_dict(base_state, strict=True)
    qlayer.train()

    return best_rho, best_loss


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

    if torch.isnan(quant_out).any() or torch.isinf(quant_out).any():
        print("NaN/Inf detected in quant_out BEFORE loss")
        pdb.set_trace()
    loss1 = loss_func(label.float(), quant_out.float())
    if not torch.isfinite(loss1):
        optimizer.zero_grad(set_to_none=True)
        print("SAM Loss_1 is NaN/Inf")
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


def sam_train_step_v2(
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

    # ===================== STEP1 =====================
    if getattr(args, "auto_rho", False):
        sdpa_ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
    else:
        sdpa_ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True,
        )

    with sdpa_ctx:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = qlayer(input, **forward_kwargs)
            quant_out = get_hidden_states(out)

            if torch.isnan(quant_out).any() or torch.isinf(quant_out).any():
                raise RuntimeError("NaN/Inf BEFORE loss (STEP1)")

            loss1 = loss_func(label.float(), quant_out.float())

        if not torch.isfinite(loss1):
            raise RuntimeError("Loss1 NaN/Inf")

        # ================== AUTO RHO ==================
        if getattr(args, "auto_rho", False):
            optimizer.zero_grad(set_to_none=True)
            optimizer.compute_auto_rho(loss1.float(), trainable_params)
            optimizer.zero_grad(set_to_none=True)

    # backward (scaled)
    scaler.scale(loss1).backward()

    scaler.unscale_(optimizer)

    if args.use_progressive:
        for m in qlayer.modules():
            if isinstance(m, int_linear_fake.QuantLinear):
                m.update_progressive_ratio()

    # ================== SAM first step ==================
    optimizer.first_step(zero_grad=True)

    # ===================== STEP2 =====================
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = qlayer(input, **forward_kwargs)
        quant_out = get_hidden_states(out)

        if torch.isnan(quant_out).any() or torch.isinf(quant_out).any():
            raise RuntimeError("NaN/Inf BEFORE loss (STEP2)")

        loss2 = loss_func(label.float(), quant_out.float())

    if not torch.isfinite(loss2):
        raise RuntimeError("Loss2 NaN/Inf")

    # backward (scaled)
    scaler.scale(loss2).backward()

    scale = scaler.get_scale()
    inv_scale = 1.0 / scale

    for p in trainable_params:
        if p.grad is not None:
            p.grad.data.mul_(inv_scale)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_params, args.clip_grad
    )

    # ================== SAM second step ==================
    optimizer.second_step(zero_grad=False)

    scaler.step(optimizer)
    scaler.update()

    return loss2.detach(), grad_norm.detach()


def sam_train_step_v3(
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

    # ===================== STEP1 =====================
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = qlayer(input, **forward_kwargs)
        quant_out = get_hidden_states(out)

        if torch.isnan(quant_out).any() or torch.isinf(quant_out).any():
            raise RuntimeError("NaN/Inf BEFORE loss (STEP1)")

        loss1 = loss_func(label.float(), quant_out.float())

    if not torch.isfinite(loss1):
        raise RuntimeError("Loss1 NaN/Inf")

    # backward step1
    scaler.scale(loss1).backward()
    scaler.unscale_(optimizer)

    if args.use_progressive:
        for m in qlayer.modules():
            if isinstance(m, int_linear_fake.QuantLinear):
                m.update_progressive_ratio()

    # ================== first step ==================
    optimizer.first_step(zero_grad=True)

    # ===================== STEP2 =====================
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = qlayer(input, **forward_kwargs)
        quant_out = get_hidden_states(out)

        if torch.isnan(quant_out).any() or torch.isinf(quant_out).any():
            raise RuntimeError("NaN/Inf BEFORE loss (STEP2)")

        loss2 = loss_func(label.float(), quant_out.float())

    if not torch.isfinite(loss2):
        raise RuntimeError("Loss2 NaN/Inf")

    # backward step2
    scaler.scale(loss2).backward()

    scale = scaler.get_scale()
    inv_scale = 1.0 / scale

    for p in trainable_params:
        if p.grad is not None:
            p.grad.data.mul_(inv_scale)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_params,
        args.clip_grad,
    )

    # ================== restore perturbed params ==================
    optimizer.second_step(zero_grad=False)

    # ================== AdamW update ==================
    scaler.step(optimizer)
    scaler.update()

    return loss2.detach(), grad_norm.detach()


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

    if attention_mask is None:
        attention_mask_batch = None
    elif attention_mask.shape[0] == args.batch_size:
        attention_mask_batch = attention_mask.float()
    elif attention_mask.shape[0] == 1:
        attention_mask_batch = attention_mask.expand(
            args.batch_size,
            *attention_mask.shape[1:]
        ).float()
    else:
        raise RuntimeError(
            f"Unexpected attention mask shape: "
            f"{tuple(attention_mask.shape)}, "
            f"expected batch={args.batch_size}"
        )
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
    if getattr(args, "loss_type", "mse") == "nlc":
        loss_func = MSEPlusNLCLoss()
    else:
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
                optimizer = ASAM(param, torch.optim.AdamW, rho=args.sam_rho, adaptive=True, weight_decay=args.wd)
                scaler = torch.cuda.amp.GradScaler()
            else:
                optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
                loss_scaler = NativeScalerWithGradNormCount()

            trainable_params = list(trainable_parameters(qlayer))

            forward_kwargs = dict(
                attention_mask=attention_mask_batch,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

            if args.use_sam and getattr(args, "rho_search", False):
                best_rho, best_rho_loss = search_block_rho(
                    qlayer,
                    param,
                    args,
                    loss_func,
                    quant_train_inps,
                    fp_train_inps,
                    quant_val_inps,
                    fp_val_inps,
                    dev,
                    forward_kwargs,
                    trainable_params,
                    logger=logger,
                )

                optimizer = ASAM(
                    param,
                    torch.optim.AdamW,
                    rho=best_rho,
                    adaptive=True,
                    weight_decay=args.wd,
                )

                if logger is not None:
                    logger.info(
                        f"[Block {block_index}] selected rho={best_rho:.6g} "
                        f"| search val loss={best_rho_loss:.6f}"
                    )

            for epoch in range(args.epochs):
                loss_list = []
                norm_list = []

                finalize_epochs = max(
                    0, int(getattr(args, "progressive_finalize_epochs", 1))
                )
                if (
                    args.use_progressive
                    and finalize_epochs > 0
                    and epoch == max(0, args.epochs - finalize_epochs)
                ):
                    for module in qlayer.modules():
                        if isinstance(module, int_linear_fake.QuantLinear):
                            module.finalize_progressive()
                    if logger is not None:
                        logger.info(
                            f"[Block {block_index}] progressive ratio finalized "
                            f"to target W{args.target_bits}"
                        )

                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):

                    input = quant_inps.to(dev)
                    label = fp_inps.to(dev)

                    forward_kwargs = dict(
                        attention_mask=attention_mask_batch,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )

                    if args.use_sam:
                        loss, norm = sam_train_step_v3(
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
        qlayer.float()
        quant_inplace(qlayer)
        set_quant_state(qlayer, False)

        # ===== update dataset =====
        if args.epochs > 0:
            update_dataset(qlayer, quant_train_inps, dev, attention_mask, position_ids, position_embeddings)
            update_dataset(qlayer, quant_val_inps, dev, attention_mask, position_ids, position_embeddings)

        layers[block_index] = qlayer.cpu()

        # step 7: pack quantized weights into low-bits format
        if args.real_quant:
            named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)

            for name, module in named_linears.items():
                if hasattr(module, "use_progressive") and module.use_progressive:
                    quantizer = module.secondary_quantizer
                    bits = args.target_bits
                else:
                    quantizer = module.weight_quantizer
                    bits = args.wbits

                scales = quantizer.scale.clamp(1e-4, 1e4).detach()
                zeros = quantizer.zero_point.detach().round()

                group_size = quantizer.group_size
                dim0 = module.weight.shape[0]

                scales = scales.view(dim0, -1).transpose(0, 1).contiguous()
                zeros = zeros.view(dim0, -1).transpose(0, 1).contiguous()

                q_linear = int_linear_real.QuantLinear(
                    bits,
                    group_size,
                    module.in_features,
                    module.out_features,
                    not module.bias is None,
                )

                q_linear.pack(
                    module,
                    scales.float().cpu(),
                    zeros.float().cpu()
                )

                set_op_by_name(qlayer, name, q_linear)

                logger.info(f"pack quantized {name} finished")
                del module

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