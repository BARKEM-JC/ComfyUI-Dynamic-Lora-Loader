from .dynamic_lora_keyword import DynamicLoraKeyword
from .dynamic_lora_offset import DynamicLoraOffset
from .dynamic_lora_block_weights import DynamicLoraBlockWeights
from .dynamic_lora_config import DynamicLoraConfig
from .dynamic_lora_loader import DynamicLoraLoader

NODE_CLASS_MAPPINGS = {
    "DynamicLoraKeyword": DynamicLoraKeyword,
    "DynamicLoraOffset": DynamicLoraOffset,
    "DynamicLoraBlockWeights": DynamicLoraBlockWeights,
    "DynamicLoraConfig": DynamicLoraConfig,
    "DynamicLoraLoader": DynamicLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicLoraKeyword": "Dynamic Lora Keyword",
    "DynamicLoraOffset": "Dynamic Lora Offset",
    "DynamicLoraBlockWeights": "Dynamic Lora Block Weights",
    "DynamicLoraConfig": "Dynamic Lora Config",
    "DynamicLoraLoader": "Dynamic Lora Loader",
}
