class DynamicLoraKeyword:
    """Multiple keywords with multiplier - comma-separated keywords are treated as one group."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "keywords": ("STRING", {"default": "", "multiline": True}),
            "multiplier": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0}),
        }}

    RETURN_TYPES = ("DYNAMIC_LORA_KEYWORD",)
    FUNCTION = "build_keywords"
    CATEGORY = "conditioning"

    def build_keywords(self, keywords, multiplier):
        # Parse comma-separated keywords and strip whitespace
        keyword_list = [k.strip() for k in (keywords or "").split(",") if k.strip()]
        
        return ({
            "keywords": keyword_list, 
            "multiplier": float(multiplier)
        },)