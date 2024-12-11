import lora_ga_init
from accelerate import Accelerator
from swift.llm import EncodePreprocessor, get_model_tokenizer, get_template, load_dataset
from typing import List, Literal, Optional

if __name__ == "__main__":
    accelerator = Accelerator()
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name = "swift/self-cognition"
    split_dataset_ratio = 0.8

    lora_dtype: Literal['float16', 'bfloat16', 'float32', None] = None

    lora_ga_batch_size: int = 2
    lora_ga_iters: int = 2
    lora_ga_direction: str = 'ArB2r'
    lora_ga_scale: str = 'stable'
    lora_ga_stable_gamma: int = 16

    model, tokenizer = get_model_tokenizer(model_id, load_model=False)

    train_set, val_set, _ = load_dataset(dataset_name, split_dataset_ratio=split_dataset_ratio)
    template = get_template(tokenizer.model_meta.template, tokenizer)
    encoded_train_dataset = EncodePreprocessor(template)(train_set)
    encoded_train_dataset = encoded_train_dataset[: lora_ga_batch_size * lora_ga_iters]

    model = lora_ga_init.entrypoint.lora_ga_init(
        model=model,
        dataset=encoded_train_dataset,
        batch_size=lora_ga_batch_size,
        num_iters=lora_ga_iters,
        direction=lora_ga_direction,
        dtype=lora_dtype,
        scale=lora_ga_scale,
        stable_gamma=lora_ga_stable_gamma,
    )