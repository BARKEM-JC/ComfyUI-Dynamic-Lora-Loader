import folder_paths

class DynamicLoraEmbedding:
    """Single embedding with weight - path selectable from models/embeddings."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            embeddings = folder_paths.get_filename_list("embeddings") or []
        except Exception:
            embeddings = []
            
        return {"required": {
            "embedding_name": (embeddings,),
            "weight": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0}),
        }}

    RETURN_TYPES = ("DYNAMIC_LORA_EMBEDDING",)
    FUNCTION = "build_embedding"
    CATEGORY = "conditioning"

    def build_embedding(self, embedding_name, weight):
        return ({
            "embedding_name": str(embedding_name),
            "weight": float(weight)
        },)