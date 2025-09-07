class DynamicLoraOffset:
    """Offset node: other LoRA id -> multiplier."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "other_id": ("STRING", {"default": ""}),
            "multiplier": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0}),
        }}

    RETURN_TYPES = ("DYNAMIC_LORA_OFFSET",)
    FUNCTION = "build_offset"
    CATEGORY = "conditioning"

    def build_offset(self, other_id, multiplier):
        return ({str(other_id): float(multiplier)},)
