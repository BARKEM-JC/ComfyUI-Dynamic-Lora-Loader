import os
import re
import random
from functools import reduce
from operator import mul
import folder_paths
from nodes import LoraLoader, CLIPTextEncode

class DynamicLoraLoader:
    """Takes MODEL, pos/neg prompts, optional CLIP, dynamic list of configs and embeddings.
    Supports randomizer codes like {tall:short:skinny:fat} and config combinations.
    Outputs modified (MODEL, CLIP, pos_prompt_out, neg_prompt_out)."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("MODEL",),
            "pos_prompt": ("STRING", {"multiline": True, "default": ""}),
            "neg_prompt": ("STRING", {"multiline": True, "default": ""}),
        }
        optional = {
            "clip": ("CLIP",),
        }
        
        # Create multiple config inputs for auto-expansion
        for i in range(1, 11):  # Start with 10 potential config slots
            optional[f"config_{i}"] = ("DYNAMIC_LORA_CONFIG",)
        
        # Create multiple positive embedding inputs
        for i in range(1, 6):
            optional[f"pos_embedding_{i}"] = ("DYNAMIC_LORA_EMBEDDING",)
            
        # Create multiple negative embedding inputs
        for i in range(1, 6):
            optional[f"neg_embedding_{i}"] = ("DYNAMIC_LORA_EMBEDDING",)
        
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING",)
    FUNCTION = "build_model_clip_and_prompts"
    CATEGORY = "conditioning"

    def _product(self, iterable):
        return reduce(mul, iterable, 1.0) if iterable else 1.0

    def _clamp(self, val, mn, mx):
        try: return max(float(mn), min(float(mx), float(val)))
        except: return val

    def _process_randomizer_codes(self, text):
        """Process {option1:option2:option3} randomizer codes in text."""
        if not text:
            return text
            
        def replace_randomizer(match):
            options = match.group(1).split(':')
            # Filter out empty options
            options = [opt.strip() for opt in options if opt.strip()]
            return random.choice(options) if options else ""
        
        # Find and replace all {option1:option2:...} patterns
        pattern = r'\{([^}]+)\}'
        result = re.sub(pattern, replace_randomizer, text)
        return result

    def _collect_embeddings(self, kwargs, prefix):
        """Collect embedding inputs with given prefix."""
        embeddings = []
        for key, value in kwargs.items():
            if key.startswith(prefix) and value is not None:
                if isinstance(value, dict):
                    embeddings.append(value)
                elif isinstance(value, list):
                    embeddings.extend([e for e in value if isinstance(e, dict)])
        return embeddings

    def _apply_embeddings(self, prompt, embeddings):
        """Apply embeddings to prompt."""
        if not embeddings or not prompt:
            return prompt
            
        embedding_tags = []
        for emb in embeddings:
            name = emb.get("embedding_name", "")
            weight = emb.get("weight", 1.0)
            
            if name:
                if weight == 1.0:
                    tag = name
                else:
                    tag = f"({name}:{weight})"
                embedding_tags.append(tag)
        
        # Add embedding tags to beginning of prompt
        if embedding_tags:
            return " ".join(embedding_tags) + " " + prompt
        return prompt

    def _resolve_config_combinations(self, cfgs, configs_to_apply):
        """Handle config combinations - when one config in a group is activated, activate all."""
        if not configs_to_apply:
            return configs_to_apply
            
        print(f"[DynamicLoraLoader] Resolving combinations. Original configs to apply: {len(configs_to_apply)}")
        
        # Group configs by combo group
        combo_groups = {}
        standalone_configs = []
        
        for config in cfgs:
            group_id = config.get("_combo_group")
            if group_id:
                if group_id not in combo_groups:
                    combo_groups[group_id] = []
                combo_groups[group_id].append(config)
                print(f"[DynamicLoraLoader] Config {config.get('id')} belongs to combo group {group_id}")
            else:
                standalone_configs.append(config)
        
        print(f"[DynamicLoraLoader] Found {len(combo_groups)} combo groups and {len(standalone_configs)} standalone configs")
        
        # Check which combo groups have at least one activated config
        activated_groups = set()
        applied_ids = {c.get("id") for c in configs_to_apply}
        
        for group_id, group_configs in combo_groups.items():
            group_ids = {c.get("id") for c in group_configs}
            if group_ids.intersection(applied_ids):
                activated_groups.add(group_id)
                print(f"[DynamicLoraLoader] Activating combo group {group_id} due to matching config(s)")
        
        # Build final list of configs to apply
        final_configs = []
        
        # Add standalone configs that were already selected
        for config in standalone_configs:
            if config in configs_to_apply:
                final_configs.append(config)
        
        # Add all configs from activated combo groups
        for group_id in activated_groups:
            for config in combo_groups[group_id]:
                # Calculate final strength for combo configs that weren't originally selected
                if config not in configs_to_apply:
                    print(f"[DynamicLoraLoader] Adding combo config {config.get('id')} that wasn't originally selected")
                    # Use base strength since no keyword matching occurred
                    base_strength = float(config.get("base_strength", 1.0))
                    
                    # Apply offset multipliers from other configs
                    id_map = {c.get("id"): c for c in cfgs if c.get("id")}
                    final_strength = base_strength
                    for other_id, off_mult in (config.get("offsets") or {}).items():
                        if other_id in id_map and other_id != config.get("id"):
                            try: 
                                final_strength *= float(off_mult)
                            except: 
                                pass
                    
                    final_strength = self._clamp(final_strength, 
                                               config.get("min_strength", -2.0), 
                                               config.get("max_strength", 2.0))
                    config["final_strength"] = final_strength
                
                final_configs.append(config)
        
        print(f"[DynamicLoraLoader] Final configs to apply: {len(final_configs)}")
        return final_configs

    def build_model_clip_and_prompts(self, model, pos_prompt, neg_prompt, clip=None, **kwargs):
        # Process randomizer codes first
        pos_prompt = self._process_randomizer_codes(pos_prompt or "")
        neg_prompt = self._process_randomizer_codes(neg_prompt or "")
        
        # Collect embedding inputs
        pos_embeddings = self._collect_embeddings(kwargs, "pos_embedding_")
        neg_embeddings = self._collect_embeddings(kwargs, "neg_embedding_")
        
        # Apply embeddings to prompts
        pos_prompt = self._apply_embeddings(pos_prompt, pos_embeddings)
        neg_prompt = self._apply_embeddings(neg_prompt, neg_embeddings)
        
        # Collect all config inputs from numbered parameters
        cfgs = []
        for key, value in kwargs.items():
            if key.startswith("config_") and value is not None:
                if isinstance(value, dict):
                    cfgs.append(value)
                elif isinstance(value, list):
                    cfgs.extend([c for c in value if isinstance(c, dict)])

        if not cfgs:
            return (model, clip, pos_prompt, neg_prompt)

        # Collect activation tags from all configs
        tags = []
        for c in cfgs:
            for t in c.get("activation_tags", []):
                if t and t not in tags: 
                    tags.append(t)

        # Add missing activation tags to positive prompt
        pos_lower = pos_prompt.lower()
        missing = [t for t in tags if t.lower() not in pos_lower]
        pos_out = (" ".join(missing) + " " + pos_prompt).strip() if missing else pos_prompt
        neg_out = neg_prompt

        # Calculate final strengths and filter configs that should be applied
        id_map = {c.get("id"): c for c in cfgs if c.get("id")}
        configs_to_apply = []
        
        for c in cfgs:
            base_strength = float(c.get("base_strength", 1.0))
            keywords_groups = c.get("keywords_groups") or []
            
            # Skip LoRAs that have no keywords defined - we only load LoRAs with matching keywords
            if not keywords_groups:
                final_strength = self._clamp(base_strength, c.get("min_strength", -2.0), c.get("max_strength", 2.0))
                c["final_strength"] = final_strength
                configs_to_apply.append(c)
                continue
            
            # Check if any keyword group matches the prompt
            any_group_matches = False
            keywords_adjustments = 0.0
            
            for kw_group in keywords_groups:
                keywords_list = kw_group.get("keywords", [])
                group_matches = any(kw.lower() in pos_out.lower() for kw in keywords_list if kw)
                
                if group_matches:
                    any_group_matches = True
                    # Apply multiplier only once per group, regardless of how many keywords match
                    kw_mult = float(kw_group.get("multiplier", 1.0))
                    keywords_adjustments += (kw_mult * base_strength) - base_strength
            
            # Skip this LoRA if none of its keywords match
            if not any_group_matches:
                continue
                
            final_strength = base_strength + keywords_adjustments
            
            # Apply offset multipliers from other configs
            for other_id, off_mult in (c.get("offsets") or {}).items():
                if other_id in id_map and other_id != c.get("id"):
                    try: 
                        final_strength *= float(off_mult)
                    except: 
                        pass
            
            # Clamp to min/max bounds
            final_strength = self._clamp(final_strength, c.get("min_strength", -2.0), c.get("max_strength", 2.0))
            c["final_strength"] = final_strength
            configs_to_apply.append(c)

        # Resolve config combinations
        configs_to_apply = self._resolve_config_combinations(cfgs, configs_to_apply)

        # Generate LoRA tags for prompt (only for configs that will be applied)
        lora_tags = []
        bw_order = ["IN00_fine_texture","IN01_low_level_edges","IN02_detail_refinement","MID_global_structure",
                    "OUT00_object_features","OUT01_mid_level_semantics","OUT02_higher_semantics","OUT03_composition",
                    "OUT04_style_refinement","OUT05_global_meaning","OUT06_late_abstraction","OUT07_final_pass"]

        for c in configs_to_apply:
            path = c.get("path") or c.get("id") or ""
            strength = c.get("final_strength", 1.0)
            bw = c.get("block_weights") or {}
            
            # Build block weights list in correct order
            bw_list = []
            for k in bw_order:
                if k in bw:
                    try: 
                        bw_list.append(str(float(bw.get(k))))
                    except: 
                        bw_list.append(str(bw.get(k)))
            
            # Create LoRA tag
            tag = f"<lora:{path}:{strength}" + (":" + ",".join(bw_list) if bw_list else "") + ">"
            lora_tags.append(tag)

        # Add LoRA tags to beginning of positive prompt
        for tag in lora_tags:
            if tag not in pos_out: 
                pos_out = tag + " " + pos_out

        # Initialize CLIP if not provided
        if clip is None:
            try:
                clip = CLIPTextEncode().encode(None, pos_out)[0]
            except Exception as e:
                print(f"[DynamicLoraLoader] CLIPTextEncode failed: {e}")
                clip = None

        # Apply LoRA models to the model and clip (only for configs that matched keywords or were combo-activated)
        for c in configs_to_apply:
            lora_filename = c.get("path") or c.get("id")
            if not lora_filename: 
                continue
                
            full = folder_paths.get_full_path("loras", lora_filename)
            if not full or not os.path.exists(full):
                print(f"[DynamicLoraLoader] LoRA file not found: {lora_filename}")
                continue
                
            try:
                out = LoraLoader().load_lora(model=model, clip=clip,
                                             lora_name=lora_filename,  # Use filename, not full path
                                             strength_model=c["final_strength"],
                                             strength_clip=c["final_strength"])
                if isinstance(out, (tuple,list)) and len(out)>=2:
                    model, clip = out[0], out[1]
                    combo_info = f" (combo: {c.get('_combo_group', 'none')})" if c.get("_combo_group") else ""
                    print(f"[DynamicLoraLoader] Applied LoRA {c.get('id')} with strength {c['final_strength']}{combo_info}")
            except Exception as e:
                print(f"[DynamicLoraLoader] Failed to apply LoRA {c.get('id')}: {e}")

        return (model, clip, pos_out, neg_out)