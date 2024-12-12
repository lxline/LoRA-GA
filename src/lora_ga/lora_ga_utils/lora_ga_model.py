import re
import torch
from itertools import chain
from peft.utils.integrations import gather_params_ctx
from peft.utils import get_quantization_config
from peft.tuners.lora import LoraLayer, LoraModel

def lora_ga_layer_init(self, adapter_name):
    def get_float_weight(model: torch.nn.Module):
        model: torch.nn.Linear

        device = model.weight.device
        in_features = model.in_features
        with torch.no_grad():
            I = torch.eye(in_features).to(device)
            w = model(I)
            if hasattr(model, "bias") and isinstance(model.bias, torch.Tensor):
                w -= model.bias
            w = torch.transpose(w, 0, 1)
        w.requires_grad = model.weight.requires_grad
        return w

    if "grad" not in self.kwargs.keys():
        return

    base_layer = self.get_base_layer()
    weight = self.get_base_layer().weight
    device = weight.device
    dtype = weight.dtype
    quant_flag = False
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        """
        for quantized model, it is needed to get the floating point weights through forward, 
        which may take 1-2 minutes (7bmodel, all linear)
        """
        quant_flag = True
        weight = get_float_weight(base_layer)
        dtype = weight.dtype
    grad = self.kwargs["grad"].to(device).to(torch.float32)
    weight = weight.to(torch.float32)
    lora_r = self.r[adapter_name]
    init_config = self.kwargs["peft_config"]
    try:
        U, S, V = torch.svd_lowrank(
            grad.float(), q=min(4 * lora_r, min(grad.shape)),
            niter=4
        )
        V = V.T
    except Exception as e:
        raise ValueError("error from torch.svd_lowrank")
    # set direction
    if init_config.direction == "ArBr":
        B = U[:, 0: 2 * lora_r: 2]
        A = V[1: 2 * lora_r: 2, :]
    elif init_config.direction == "A2rBr":
        B = U[:, :lora_r]
        A = V[lora_r: 2 * lora_r, :]
    elif init_config.direction == "ArB2r":
        B = U[:, lora_r: 2 * lora_r]
        A = V[:lora_r, :]
    elif init_config.direction == "random":
        import random
        random_list = random.sample(range(2 * lora_r), 2 * lora_r)
        indexes_A = random_list[0:lora_r]
        indexes_B = random_list[lora_r:2 * lora_r]
        print(f"indexes_A={indexes_A}")
        print(f"indexes_B={indexes_B}")
        B = U[:, indexes_B]
        A = V[indexes_A, :]
    scaling_factor = self.scaling["default"]
    if init_config.scale == "gd":
        A = A / scaling_factor
        B = B / scaling_factor
    elif init_config.scale == "unit":
        # Because A,B is orthogonal, do not need to scale
        pass
    elif init_config.scale == "stable":
        m, n = grad.shape  # m: feature_out, n: feature_in
        # the scale of output is only related to the feature_out
        gamma = init_config.stable_gamma
        B = B * m ** 0.25 / gamma ** 0.5
        A = A * m ** 0.25 / gamma ** 0.5
    elif init_config.scale == "weightS":
        _, S, _ = torch.svd_lowrank(weight.data.float(), q=4 * lora_r, niter=4)
        S = S / self.scaling["default"]
        avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
        B = B * avg_s
        A = A * avg_s

    offset = B @ A
    # Training type
    # consider dtype not in init_config
    if not hasattr(init_config, "dtype"):
        pass
    elif init_config.dtype == "bf16":
        A = A.to(torch.bfloat16)
        B = B.to(torch.bfloat16)
    elif init_config.dtype == "fp32":
        A = A.to(torch.float32)
        B = B.to(torch.float32)
    scaling_factor = self.scaling["default"]
    offset *= scaling_factor
    if hasattr(init_config, "norm_clip") and init_config.norm_clip:
        # for numerical stability, offset's largest value must be less then weight's largest value
        ratio = torch.max(torch.abs(weight.data)) / torch.max(
            torch.abs(offset)
        )
        if ratio < 1:
            offset *= ratio
            A *= ratio ** 0.5
            B *= ratio ** 0.5

    weight.data -= offset

    self.lora_A[adapter_name].weight.data = A.contiguous()
    self.lora_B[adapter_name].weight.data = B.contiguous()
    if not quant_flag:
        weight = weight.data
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight
    else:
        has_bias = True if base_layer.bias is not None else False
        float_linear = torch.nn.Linear(base_layer.in_features, base_layer.out_features, has_bias)
        if has_bias and isinstance(base_layer.bias.data, torch.Tensor):
            float_linear.bias.data = base_layer.bias.data
        float_linear.weight.data = weight.data
        import bitsandbytes
        if isinstance(base_layer, bitsandbytes.nn.Linear8bitLt):
            new_base_layer = type(base_layer)(base_layer.in_features, base_layer.out_features, has_bias,
                                              has_fp16_weights=False)
        else:
            new_base_layer = type(base_layer)(base_layer.in_features, base_layer.out_features, has_bias, )
        new_base_layer.load_state_dict(float_linear.state_dict())
        new_base_layer.to(device)
        base_layer.__dict__.update(new_base_layer.__dict__)
        del new_base_layer


def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
    self.update_layer_origin(
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        False,
        use_rslora,
        use_dora,
        lora_bias,
    )
    if isinstance(init_lora_weights, str) and init_lora_weights.lower() == "lora-ga":
        with gather_params_ctx(self.get_base_layer().weight):
            self.lora_ga_layer_init(adapter_name)


class LoraGAModel(LoraModel):
    """
    Creates Low Rank Adapter (LoRA) with Gradient Approximation  model (LoRA-GA) from a pretrained transformers model.
    The method is described in detail in https://arxiv.org/abs/2407.05000
    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
    Returns:
        `torch.nn.Module`: The Lora model.
    """

    def __init__(self, model, config, adapter_name):
        named_grad_key = "named_grad"
        self.named_grad = None
        if hasattr(model, named_grad_key) and getattr(model, named_grad_key) is not None:
            self.named_grad = getattr(model, "named_grad")

        super().__init__(model, config, adapter_name)
        self.named_grad = None

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if lora_config.init_lora_weights != "lora_ga" or self.named_grad is None:
            super()._create_and_replace(
                lora_config,
                adapter_name,
                target,
                target_name,
                parent,
                current_key,
            )
            return
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        if lora_config.init_lora_weights == "lora_ga" and self.named_grad is not None:
            kwargs.update({"peft_config": self.peft_config[adapter_name], "grad": self.named_grad[current_key]})
        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)