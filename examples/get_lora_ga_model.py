"""
In ms-swift/swift/llm/train/tuner.py
"""
import lora_ga

model = lora_ga.entrypoint.get_lora_ga_model(
    model=model,
    data_collator=template.data_collator,
    dataset=train_dataset,
    batch_size=args.lora_ga_batch_size,
    num_iters=args.lora_ga_iters,
    max_length=args.lora_ga_max_length,
    direction=args.lora_ga_direction,
    dtype=args.lora_dtype,
    scale=args.lora_ga_scale,
    stable_gamma=args.lora_ga_stable_gamma,
)