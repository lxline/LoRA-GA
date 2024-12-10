import argparse

from peft import get_peft_model
from torch.utils.data import DataLoader
from accelerate import Accelerator

from .lora_ga_utils import (estimate_gradient, LoraGAConfig, LoraGAContext, find_all_linear_modules)

def lora_ga_init(model,
         dataset,
         batch_size: int=2,
         num_iters: int=64,
         direction: str="ArB2r",
         max_length: int=1024,
         dtype: str="fp32",
         scale: str="stable",
         stable_gamma: int=16):
    peft_config = LoraGAConfig(
        bsz=batch_size,
        iters=num_iters,
        direction=direction,
        max_length=max_length,
        dtype=dtype,
        scale=scale,
        stable_gamma=stable_gamma,
        target_modules=find_all_linear_modules(model=model),
    )
    dataloader = DataLoader(dataset)
    accelerator = Accelerator()
    named_grad = estimate_gradient(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
        quant_flag=False,
    )

    with LoraGAContext(model=model, named_grad=named_grad):
        model = get_peft_model(model=model, peft_config=peft_config, adapter_name="default")

    return model


def arg_parser():
    parser = argparse.ArgumentParser(description="LoRA GA Initialization")
    parser.add_argument("--model", type=str, default="model", help="Model to be optimized")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset to be used")
    # parser.add_argument("--save_dir", type=str, default="save_dir", help="Directory to save the results")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for data loader")
    parser.add_argument("--num_iters", type=int, default=64, help="Number of iterations for LoRA GA")
    parser.add_argument("--direction", type=str, default="ArB2r", help="Direction for gradient estimation")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length for input sequences")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="Data type for computations")
    parser.add_argument("--scale", type=str, default="stable", choices=["stable", "unstable"], help="Scaling method")
    parser.add_argument("--stable_gamma", type=int, default=16, help="Gamma value for stable scaling")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    # Assuming 'model' and 'dataset' are defined elsewhere in your codebase.
    # For demonstration purposes, we'll pass None here.
    lora_ga_init(None, None,
                 batch_size=args.batch_size,
                 num_iters=args.num_iters,
                 direction=args.direction,
                 max_length=args.max_length,
                 dtype=args.dtype,
                 scale=args.scale,
                 stable_gamma=args.stable_gamma)



