import torch
from quantize.real_linear import load_quantized_model
from quantize.fake_linear import QuantLinear

@torch.no_grad()
def check_reconstruction(model):

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):

            w = module.weight.data

            q_before = w.reshape(-1, module.group_size)

            scale = module.weight_quantizer.scale
            zp = module.weight_quantizer.zero_point

            code = q_before / scale + zp.round()

            frac = (code - code.round()).abs().max()

            print(name, frac.item())


@torch.no_grad()
def check_quant_param_match(model):

    for name, module in model.named_modules():

        if not hasattr(module, "weight_quantizer"):
            continue

        q = module.weight_quantizer

        w = module.weight.data

        scale = q.scale.data
        zp = q.zero_point.data

        x = w.reshape(-1, q.group_size)

        scale = scale.reshape_as(x[:, :1])
        zp = zp.reshape_as(x[:, :1])

        q_int = torch.round(x / scale) + zp
        q_int = q_int.clamp(q.qmin, q.qmax)

        w_recon = (q_int - zp) * scale

        err = (x - w_recon).abs()

        print(
            f"{name}: "
            f"max={err.max().item():.6f} "
            f"mean={err.mean().item():.6f}"
        )


@torch.no_grad()
def check_negative(model):
    for name, module in model.named_modules():

        if isinstance(module, QuantLinear):

            scale = module.weight_quantizer.scale

            neg_ratio = (scale < 0).float().mean()

            if neg_ratio > 0:
                print(
                    name,
                    "negative scale ratio:",
                    neg_ratio.item(),
                    "min:",
                    scale.min().item(),
                    "max:",
                    scale.max().item()
                )

@torch.no_grad()
def materialize_quant_weight(model):
    for name, module in model.named_modules():

        if not isinstance(module, QuantLinear):
            continue

        qweight = module.weight_quantizer(module.weight)
        module.weight.data.copy_(qweight)

        module.use_weight_quant = False

@torch.no_grad()
def evaluate(model):
    results = {}
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    task_list = "gsm8k"

    model_eval = HFLM(pretrained=model, batch_size=64)

    task_manager = lm_eval.tasks.TaskManager()

    results = lm_eval.simple_evaluate(
        model=model_eval,
        tasks=task_list,
        num_fewshot=5,
        task_manager=task_manager,
    )

    print(make_table(results))

    return results


model_path = "/share/MY-DAPO/block_qat/qwen3-4b-w2g128-asam-pg"

model, _ = load_quantized_model(model_path, 2, 128)

# check_reconstruction(model)
# check_quant_param_match(model)
# check_negative(model)

materialize_quant_weight(model)
evaluate(model)