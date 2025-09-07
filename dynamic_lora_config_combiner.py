class DynamicLoraConfigCombiner:
    """Combines multiple LoRA configs so when one is activated (loaded via keywords), others are also activated and loaded.
    They still function normally otherwise (independent keyword matching, strength calculations, etc.)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "combine_mode": (["all_or_none", "primary_triggers_all"], {"default": "all_or_none"}),
        }
        
        # Dynamic inputs with numbered suffixes for auto-expansion
        optional = {}
        
        # Create multiple config inputs
        for i in range(1, 11):  # Start with 10 potential slots
            optional[f"config_{i}"] = ("DYNAMIC_LORA_CONFIG",)
        
        return {"required": required, "optional": optional}
    
    RETURN_TYPES = tuple(["DYNAMIC_LORA_CONFIG"] * 10)  # Return up to 10 configs
    RETURN_NAMES = tuple([f"config_{i}" for i in range(1, 11)])
    FUNCTION = "combine_configs"
    CATEGORY = "conditioning"
    
    def combine_configs(self, combine_mode="all_or_none", **kwargs):
        """Combine configs with linking behavior."""
        
        # Collect all config inputs
        configs = []
        for key, value in kwargs.items():
            if key.startswith("config_") and value is not None:
                if isinstance(value, dict):
                    configs.append(value)
                elif isinstance(value, list):
                    configs.extend([c for c in value if isinstance(c, dict)])
        
        if not configs:
            # Return None for all outputs
            return tuple([None] * 10)
        
        # Add linking metadata to each config
        group_id = f"combined_{id(self)}"  # Unique group identifier
        
        for i, config in enumerate(configs):
            # Create a copy to avoid modifying original
            linked_config = dict(config)
            
            # Add linking metadata
            linked_config["_combo_group"] = group_id
            linked_config["_combo_mode"] = combine_mode
            linked_config["_combo_members"] = [c.get("id", f"config_{j}") for j, c in enumerate(configs)]
            linked_config["_combo_index"] = i
            
            configs[i] = linked_config
        
        # Pad with None values to match return count
        while len(configs) < 10:
            configs.append(None)
        
        return tuple(configs[:10])