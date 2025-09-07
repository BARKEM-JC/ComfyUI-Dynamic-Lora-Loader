import folder_paths

class DynamicLoraConfig:
    """LoRA config node with dynamic inputs for keywords, offsets, and block weights.
    Activation tags live here as a comma-separated string."""
    
    @classmethod
    def INPUT_TYPES(cls):
        try:
            loras = folder_paths.get_filename_list("loras") or []
        except Exception:
            loras = []
            
        # Core required fields
        required = {
            "id": ("STRING", {"default": ""}),
            "lora_name": (loras,),
            "base_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0}),
            "min_strength": ("FLOAT", {"default": -2.0, "min": -100.0, "max": 100.0}),
            "max_strength": ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0}),
            "activation_tags": ("STRING", {"default": ""}),
        }
        
        # Dynamic inputs with numbered suffixes for auto-expansion
        optional = {}
        
        # Create multiple keywords inputs (ComfyUI will show more as needed)
        for i in range(1, 6):  # Start with 5 potential slots
            optional[f"keywords_{i}"] = ("DYNAMIC_LORA_KEYWORD",)
            
        # Create multiple offset inputs
        for i in range(1, 6):
            optional[f"offset_{i}"] = ("DYNAMIC_LORA_OFFSET",)
            
        # Create multiple block weight inputs
        for i in range(1, 6):
            optional[f"block_weights_{i}"] = ("DYNAMIC_LORA_BLOCK_WEIGHTS",)
        
        return {"required": required, "optional": optional}
    
    RETURN_TYPES = ("DYNAMIC_LORA_CONFIG",)
    FUNCTION = "build_config"
    CATEGORY = "conditioning"
    
    def build_config(self, id, lora_name, base_strength, min_strength, max_strength,
                     activation_tags="", **kwargs):
        """Assemble config dict for DynamicLoraLoader."""
        
        def normalize_list(x):
            if x is None: 
                return []
            return x if isinstance(x, list) else [x]
        
        # Collect all keywords inputs
        keywords_groups = []
        for key, value in kwargs.items():
            if key.startswith("keywords_") and value is not None:
                if isinstance(value, dict) and value.get("keywords"):
                    # Ensure keywords is a list and has at least one non-empty keyword
                    kw_list = value.get("keywords", [])
                    if isinstance(kw_list, list) and any(k.strip() for k in kw_list):
                        keywords_groups.append({
                            "keywords": [k.strip() for k in kw_list if k.strip()],
                            "multiplier": float(value.get("multiplier", 1.0))
                        })
        
        # Collect all offset inputs
        offsets = {}
        for key, value in kwargs.items():
            if key.startswith("offset_") and value is not None:
                if isinstance(value, dict):
                    offsets.update(value)
        
        # Collect all block weight inputs
        block_weights = {}
        for key, value in kwargs.items():
            if key.startswith("block_weights_") and value is not None:
                if isinstance(value, dict):
                    block_weights.update(value)
        
        # Parse activation tags
        tags = [t.strip() for t in (activation_tags or "").split(",") if t.strip()]
        
        return ({
            "id": str(id) if id else str(lora_name or ""),
            "path": lora_name,
            "base_strength": float(base_strength),
            "min_strength": float(min_strength),
            "max_strength": float(max_strength),
            "keywords_groups": keywords_groups,
            "offsets": offsets,
            "activation_tags": tags,
            "block_weights": block_weights,
        },)