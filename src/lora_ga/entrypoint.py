import argparse
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from peft import get_peft_model, PeftModel
from peft.utils import PeftType
from peft.config import PeftConfig
from peft.tuners.lora import LoraLayer
from peft.tuners import (
    AdaLoraConfig, AdaLoraModel,
    AdaptionPromptConfig,
    BOFTConfig, BOFTModel,
    FourierFTConfig, FourierFTModel,
    HRAConfig, HRAModel,
    IA3Config, IA3Model,
    LNTuningConfig, LNTuningModel,
    LoHaConfig, LoHaModel,
    LoKrConfig, LoKrModel,
    MultitaskPromptTuningConfig,
    OFTConfig, OFTModel,
    PolyConfig, PolyModel,
    PrefixTuningConfig,
    PromptEncoderConfig, PromptTuningConfig,
    VeraConfig, VeraModel,
    XLoraConfig, XLoraModel,
    AdaptionPromptModel,
    BoneModel,
    CPTEmbedding,
    PrefixEncoder,
    PromptEmbedding,
    PromptEncoder,
    VBLoRAModel,
)
from peft.tuners.tuners_utils import BaseTuner as _BaseTuner
import peft.peft_model as peft_model
import peft.mapping as mapping

from .lora_ga_utils import (estimate_gradient, LoraGAConfig, find_all_linear_modules,
                            LoraGAModel, lora_ga_layer_init, update_layer)

PEFT_TYPE_TO_TUNER_MAPPING: dict[str, type[_BaseTuner]] = {
    "LORA": LoraGAModel, # Use LoraGAModel instead of LoraModel
    "LOHA": LoHaModel,
    "LOKR": LoKrModel,
    "ADALORA": AdaLoraModel,
    "BOFT": BOFTModel,
    "IA3": IA3Model,
    "OFT": OFTModel,
    "POLY": PolyModel,
    "LN_TUNING": LNTuningModel,
    "VERA": VeraModel,
    "FOURIERFT": FourierFTModel,
    "XLORA": XLoraModel,
    "HRA": HRAModel,
}

PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraGAModel,
    PeftType.LOHA: LoHaModel,
    PeftType.LOKR: LoKrModel,
    PeftType.PROMPT_TUNING: PromptEmbedding,
    PeftType.P_TUNING: PromptEncoder,
    PeftType.PREFIX_TUNING: PrefixEncoder,
    PeftType.ADALORA: AdaLoraModel,
    PeftType.BOFT: BOFTModel,
    PeftType.ADAPTION_PROMPT: AdaptionPromptModel,
    PeftType.IA3: IA3Model,
    PeftType.OFT: OFTModel,
    PeftType.POLY: PolyModel,
    PeftType.LN_TUNING: LNTuningModel,
    PeftType.VERA: VeraModel,
    PeftType.FOURIERFT: FourierFTModel,
    PeftType.XLORA: XLoraModel,
    PeftType.HRA: HRAModel,
    PeftType.VBLORA: VBLoRAModel,
    PeftType.CPT: CPTEmbedding,
    PeftType.BONE: BoneModel,
}

PEFT_TYPE_TO_CONFIG_MAPPING: dict[str, type[PeftConfig]] = {
    "LORA": LoraGAConfig, # Use LoraGAConfig instead of LoraConfig
    "ADAPTION_PROMPT": AdaptionPromptConfig,
    "PROMPT_TUNING": PromptTuningConfig,
    "PREFIX_TUNING": PrefixTuningConfig,
    "P_TUNING": PromptEncoderConfig,
    "LOHA": LoHaConfig,
    "LOKR": LoKrConfig,
    "ADALORA": AdaLoraConfig,
    "BOFT": BOFTConfig,
    "IA3": IA3Config,
    "MULTITASK_PROMPT_TUNING": MultitaskPromptTuningConfig,
    "OFT": OFTConfig,
    "POLY": PolyConfig,
    "LN_TUNING": LNTuningConfig,
    "VERA": VeraConfig,
    "FOURIERFT": FourierFTConfig,
    "XLORA": XLoraConfig,
    "HRA": HRAConfig,
}

class LoraGAContext:
    """
    Context manager for attaching and detaching a named gradient dictionary to a model.

    This context manager allows you to temporarily attach a dictionary of named gradients
    to the model as an attribute. Upon entering the context, the `named_grad` dictionary
    is set as an attribute of the model. Upon exiting the context, the attribute is removed.

    Attributes:
        model (torch.nn.Module): The model to which the gradient dictionary will be attached.
        named_grad (dict, optional): A dictionary where keys are parameter names and values are gradients. Defaults to None.
    """

    def __init__(self,
        model: torch.nn.Module,
        named_grad: dict = None,
    ) -> None:
        self.model = model
        self.named_grad = named_grad

    def __enter__(self):
        if self.named_grad:
            setattr(self.model, "named_grad", self.named_grad)
        mapping.PEFT_TYPE_TO_CONFIG_MAPPING_origin = mapping.PEFT_TYPE_TO_CONFIG_MAPPING
        mapping.PEFT_TYPE_TO_TUNER_MAPPING_origin = mapping.PEFT_TYPE_TO_TUNER_MAPPING
        mapping.PEFT_TYPE_TO_CONFIG_MAPPING = PEFT_TYPE_TO_CONFIG_MAPPING
        mapping.PEFT_TYPE_TO_TUNER_MAPPING = PEFT_TYPE_TO_TUNER_MAPPING

        peft_model.PEFT_TYPE_TO_MODEL_MAPPING_origin = peft_model.PEFT_TYPE_TO_MODEL_MAPPING
        peft_model.PEFT_TYPE_TO_MODEL_MAPPING = PEFT_TYPE_TO_MODEL_MAPPING

        LoraLayer.update_layer_origin = LoraLayer.update_layer
        LoraLayer.update_layer = update_layer
        LoraLayer.lora_ga_layer_init = lora_ga_layer_init

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named_grad and hasattr(self.model, "named_grad"):
            delattr(self.model, "named_grad")
        mapping.PEFT_TYPE_TO_CONFIG_MAPPING = mapping.PEFT_TYPE_TO_CONFIG_MAPPING_origin
        mapping.PEFT_TYPE_TO_TUNER_MAPPING = mapping.PEFT_TYPE_TO_TUNER_MAPPING_origin
        peft_model.PEFT_TYPE_TO_MODEL_MAPPING = peft_model.PEFT_TYPE_TO_MODEL_MAPPING_origin

        LoraLayer.update_layer = LoraLayer.update_layer_origin

        del mapping.PEFT_TYPE_TO_CONFIG_MAPPING_origin
        del mapping.PEFT_TYPE_TO_TUNER_MAPPING_origin
        del peft_model.PEFT_TYPE_TO_MODEL_MAPPING_origin
        del LoraLayer.update_layer_origin
        del LoraLayer.lora_ga_layer_init

def get_lora_ga_model(model,
                      data_collator,
                      dataset,
                      batch_size: int = 2,
                      num_iters: int = 64,
                      max_length: int = 1024,
                      direction: str = "ArB2r",
                      dtype: str = "fp32",
                      scale: str = "stable",
                      stable_gamma: int = 16):

    peft_config = LoraGAConfig(
        task_type='CAUSAL_LM',
        bsz=batch_size,
        iters=num_iters,
        max_length=max_length,
        direction=direction,
        dtype=dtype,
        scale=scale,
        stable_gamma=stable_gamma,
        target_modules=find_all_linear_modules(model=model),
    )

    num_samples = batch_size * num_iters
    if len(dataset) < num_samples:
        raise ValueError(
            f"Dataset does not contain enough samples. LoRA-GA requested batch_size * num_iters = {num_samples} samples, but the dataset only has {len(dataset)} samples.")
    dataset = dataset.select(range(num_samples))
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: data_collator(x))

    accelerator = Accelerator()

    model.requires_grad_(True)
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
    parser.add_argument("--tokenizer", type=str, default="tokenizer", help="Tokenizer to be used")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset to be used")
    # parser.add_argument("--save_dir", type=str, default="save_dir", help="Directory to save the results")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for data loader")
    parser.add_argument("--num_iters", type=int, default=64, help="Number of iterations for LoRA GA")
    parser.add_argument("--direction", type=str, default="ArB2r", help="Direction for gradient estimation")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="Data type for computations")
    parser.add_argument("--scale", type=str, default="stable", choices=["stable", "unstable"], help="Scaling method")
    parser.add_argument("--stable_gamma", type=int, default=16, help="Gamma value for stable scaling")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    # Assuming 'model' and 'dataset' are defined elsewhere in your codebase.
    # For demonstration purposes, we'll pass None here.
    get_lora_ga_model(None, None,
                      batch_size=args.batch_size,
                      num_iters=args.num_iters,
                      direction=args.direction,
                      dtype=args.dtype,
                      scale=args.scale,
                      stable_gamma=args.stable_gamma)