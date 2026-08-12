from quantize.real_linear import load_quantized_model
import torch
from pathlib import Path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,  help="model path of resumed quantized model")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--wbits", type=int, default=4, help="quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="quantization group size")
    parser.add_argument("--target_type",type=str,default="fp16",choices=["fp16", "bf16"])


    args = parser.parse_args()
    model, tokenizer = load_quantized_model(args.model,args.wbits, args.group_size)
    model.cuda()


    if args.target_type =='fp16':
        dtype = torch.float16
    elif args.target_type =='bf16':
        dtype = torch.bfloat16
    else:
        raise NotImplementedError
        
    model.to(dtype)
    print(f"transfer model to {args.target_type} format")

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        print("start saving model")
        model.save_pretrained(args.save_dir)  
        tokenizer.save_pretrained(args.save_dir) 
        print(f"save model to {args.save_dir} success")

if __name__ =='__main__':
    main()