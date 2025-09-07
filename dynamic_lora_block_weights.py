class DynamicLoraBlockWeights:
    """Provide per-block LoRA weights (12 entries)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "IN00_fine_texture": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "IN01_low_level_edges": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "IN02_detail_refinement": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "MID_global_structure": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "OUT00_object_features": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "OUT01_mid_level_semantics": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "OUT02_higher_semantics": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "OUT03_composition": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "OUT04_style_refinement": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "OUT05_global_meaning": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "OUT06_late_abstraction": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            "OUT07_final_pass": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
        }}

    RETURN_TYPES = ("DYNAMIC_LORA_BLOCK_WEIGHTS",)
    FUNCTION = "build_block_weights"
    CATEGORY = "conditioning"

    def build_block_weights(self, **kwargs):
        return (kwargs or {},)
