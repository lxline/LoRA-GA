from dataclasses import dataclass, field

from peft.tuners.lora import LoraConfig

@dataclass
class LoraGAConfig(LoraConfig):
    """
    Configuration for the LoraGA (Lora Gradient Alignment) approach.

    This class extends the LoraConfig base class to include additional parameters
    specific to the LoraGA method. It manages hyperparameters for training and
    gradient alignment settings.

    Attributes:
        bsz (int): The batch size for training. Default is 2.
        iters (int): The number of iterations for training. Default is 2.
        direction (str): The direction of gradient alignment. Default is "ArB2r".
        dtype (str): The data type used for computations. Default is "fp32".
        scale (str): Scaling method for gradients. Default is "stable".
        stable_gamma (int): Gamma value used for stable scaling. Default is 16.
    """

    bsz: int = field(
        default=2,
    )
    iters: int = field(
        default=2,
    )
    direction: str = field(
        default="ArB2r",
    )
    dtype: str = field(
        default="fp32",
    )
    scale: str = field(default="stable")
    stable_gamma: int = field(
        default=16,
    )

    def __post_init__(self):
        super().__post_init__()
        self.init_lora_weights = "pissa"