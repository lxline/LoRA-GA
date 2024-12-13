"""
In ms-swift/swift/trainers/mixin.py
"""
from lora_ga.entrypoint import LoraGAContext

with LoraGAContext(model):
    model.save_pretrained(
        os.path.join(output_dir, 'converted', 'default'),
        path_initial_model_for_weight_conversion=os.path.join(os.path.dirname(output_dir),
                                                              'initial_model'),
    )
    model.peft_config['default'] = config