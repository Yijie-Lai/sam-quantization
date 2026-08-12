import torch
from quantize.real_linear import load_quantized_model

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
save_quant_dir = "/share/MY-DAPO/dapo/qwen3-4b-w2g128-asam-pg-block"

model, tokenizer = load_quantized_model(model_path, 2, 128)

model.save_pretrained(save_quant_dir)
tokenizer.save_pretrained(save_quant_dir)

for module in model.modules():
    if hasattr(model, "use_weight_quant"):
        module.use_weight_quant = True
evaluate(model)