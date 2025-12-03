from diffusers.models.attention_processor import FluxAttnProcessor2_0
from safetensors import safe_open
import re
import torch

from .layers import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor
from safetensors.torch import load_file

device = "cuda"


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]


def load_checkpoint(local_path):
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    return checkpoint


def update_model_with_lora(checkpoint, lora_weights, transformer, cond_size):
    number = len(lora_weights)
    ranks = [get_lora_rank(checkpoint) for _ in range(number)]
    lora_attn_procs = {}
    double_blocks_idx = list(range(19))
    single_blocks_idx = list(range(38))
    for name, attn_processor in transformer.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:

            lora_state_dicts = {}
            for key, value in checkpoint.items():
                if re.search(r'\.(\d+)\.', key):
                    checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                    if checkpoint_layer_index == layer_index and key.startswith("transformer_blocks"):
                        lora_state_dicts[key] = value

            lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, device=device,
                dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=number
            )

            for n in range(number):
                lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(
                    f'{name}.q_loras.{n}.down.weight', None)
                lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight',
                                                                                       None)
                lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(
                    f'{name}.k_loras.{n}.down.weight', None)
                lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight',
                                                                                       None)
                lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(
                    f'{name}.v_loras.{n}.down.weight', None)
                lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight',
                                                                                       None)
                lora_attn_procs[name].proj_loras[n].down.weight.data = lora_state_dicts.get(
                    f'{name}.proj_loras.{n}.down.weight', None)
                lora_attn_procs[name].proj_loras[n].up.weight.data = lora_state_dicts.get(
                    f'{name}.proj_loras.{n}.up.weight', None)
                lora_attn_procs[name].to(device)

        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:

            lora_state_dicts = {}
            for key, value in checkpoint.items():
                if re.search(r'\.(\d+)\.', key):
                    checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                    if checkpoint_layer_index == layer_index and key.startswith("single_transformer_blocks"):
                        lora_state_dicts[key] = value

            lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, device=device,
                dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=number
            )
            for n in range(number):
                lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(
                    f'{name}.q_loras.{n}.down.weight', None)
                lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight',
                                                                                       None)
                lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(
                    f'{name}.k_loras.{n}.down.weight', None)
                lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight',
                                                                                       None)
                lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(
                    f'{name}.v_loras.{n}.down.weight', None)
                lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight',
                                                                                       None)
                lora_attn_procs[name].to(device)
        else:
            lora_attn_procs[name] = FluxAttnProcessor2_0()

    transformer.set_attn_processor(lora_attn_procs)


def update_model_with_multi_lora(checkpoints, lora_weights, transformer, cond_size):
    ck_number = len(checkpoints)
    cond_lora_number = [len(ls) for ls in lora_weights]
    cond_number = sum(cond_lora_number)
    ranks = [get_lora_rank(checkpoint) for checkpoint in checkpoints]
    multi_lora_weight = []
    for ls in lora_weights:
        for n in ls:
            multi_lora_weight.append(n)

    lora_attn_procs = {}
    double_blocks_idx = list(range(19))
    single_blocks_idx = list(range(38))
    for name, attn_processor in transformer.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            lora_state_dicts = [{} for _ in range(ck_number)]
            for idx, checkpoint in enumerate(checkpoints):
                for key, value in checkpoint.items():
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("transformer_blocks"):
                            lora_state_dicts[idx][key] = value

            lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=multi_lora_weight, device=device,
                dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=cond_number
            )

            num = 0
            for idx in range(ck_number):
                for n in range(cond_lora_number[idx]):
                    lora_attn_procs[name].q_loras[num].down.weight.data = lora_state_dicts[idx].get(
                        f'{name}.q_loras.{n}.down.weight', None)
                    lora_attn_procs[name].q_loras[num].up.weight.data = lora_state_dicts[idx].get(
                        f'{name}.q_loras.{n}.up.weight', None)
                    lora_attn_procs[name].k_loras[num].down.weight.data = lora_state_dicts[idx].get(
                        f'{name}.k_loras.{n}.down.weight', None)
                    lora_attn_procs[name].k_loras[num].up.weight.data = lora_state_dicts[idx].get(
                        f'{name}.k_loras.{n}.up.weight', None)
                    lora_attn_procs[name].v_loras[num].down.weight.data = lora_state_dicts[idx].get(
                        f'{name}.v_loras.{n}.down.weight', None)
                    lora_attn_procs[name].v_loras[num].up.weight.data = lora_state_dicts[idx].get(
                        f'{name}.v_loras.{n}.up.weight', None)
                    lora_attn_procs[name].proj_loras[num].down.weight.data = lora_state_dicts[idx].get(
                        f'{name}.proj_loras.{n}.down.weight', None)
                    lora_attn_procs[name].proj_loras[num].up.weight.data = lora_state_dicts[idx].get(
                        f'{name}.proj_loras.{n}.up.weight', None)
                    lora_attn_procs[name].to(device)
                    num += 1

        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:

            lora_state_dicts = [{} for _ in range(ck_number)]
            for idx, checkpoint in enumerate(checkpoints):
                for key, value in checkpoint.items():
                    if re.search(r'\.(\d+)\.', key):
                        checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
                        if checkpoint_layer_index == layer_index and key.startswith("single_transformer_blocks"):
                            lora_state_dicts[idx][key] = value

            lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=multi_lora_weight, device=device,
                dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size, n_loras=cond_number
            )
            num = 0
            for idx in range(ck_number):
                for n in range(cond_lora_number[idx]):
                    lora_attn_procs[name].q_loras[num].down.weight.data = lora_state_dicts[idx].get(
                        f'{name}.q_loras.{n}.down.weight', None)
                    lora_attn_procs[name].q_loras[num].up.weight.data = lora_state_dicts[idx].get(
                        f'{name}.q_loras.{n}.up.weight', None)
                    lora_attn_procs[name].k_loras[num].down.weight.data = lora_state_dicts[idx].get(
                        f'{name}.k_loras.{n}.down.weight', None)
                    lora_attn_procs[name].k_loras[num].up.weight.data = lora_state_dicts[idx].get(
                        f'{name}.k_loras.{n}.up.weight', None)
                    lora_attn_procs[name].v_loras[num].down.weight.data = lora_state_dicts[idx].get(
                        f'{name}.v_loras.{n}.down.weight', None)
                    lora_attn_procs[name].v_loras[num].up.weight.data = lora_state_dicts[idx].get(
                        f'{name}.v_loras.{n}.up.weight', None)
                    lora_attn_procs[name].to(device)
                    num += 1

        else:
            lora_attn_procs[name] = FluxAttnProcessor2_0()

    transformer.set_attn_processor(lora_attn_procs)


def set_single_lora(transformer, local_path, lora_weights=[], cond_size=512):
    checkpoint = load_checkpoint(local_path)
    update_model_with_lora(checkpoint, lora_weights, transformer, cond_size)


def set_multi_lora(transformer, local_paths, lora_weights=[[]], cond_size=512):
    checkpoints = [load_checkpoint(local_path) for local_path in local_paths]
    update_model_with_multi_lora(checkpoints, lora_weights, transformer, cond_size)


def unset_lora(transformer):
    lora_attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        lora_attn_procs[name] = FluxAttnProcessor2_0()
    transformer.set_attn_processor(lora_attn_procs)


def update_model_with_lora_and_ip_adapter3(
        checkpoint, lora_weights, ip_adapter_checkpoint, transformer, mlp_proj_model,
        cond_size, device, scale=1.0
):
    number = len(lora_weights)
    ranks = [get_lora_rank(checkpoint) for _ in range(number)]
    lora_attn_procs = {}
    double_blocks_idx = list(range(19))
    single_blocks_idx = list(range(38))

    stats = {
        "double": {
            "lora_loaded": set(),
            "lora_missing": set(),
            "ip_loaded": set(),
            "ip_missing": set()
        },
        "single": {
            "lora_loaded": set(),
            "lora_missing": set(),
            "ip_loaded": set(),
            "ip_missing": set()
        }
    }

    def print_block_stats(block_type, stats_dict):
        print(f"\n==== {block_type} Loading Details ====")
        print(f"LoRA Loaded: {len(stats_dict['lora_loaded'])}")
        print(f"LoRA Missing: {len(stats_dict['lora_missing'])} (first 5: {list(stats_dict['lora_missing'])[:5]})")
        print(f"IP-Adapter Loaded: {len(stats_dict['ip_loaded'])}")
        print(f"IP-Adapter Missing: {len(stats_dict['ip_missing'])} (first 5: {list(stats_dict['ip_missing'])[:5]})")

    for name, _ in transformer.attn_processors.items():
        layer_match = re.search(r'\.(\d+)\.', name)
        if not layer_match:
            lora_attn_procs[name] = FluxAttnProcessor2_0()
            continue
        layer_index = int(layer_match.group(1))

        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            block_type = "Double Stream Block"
            current_stats = stats["double"]

            lora_state_dicts = {}
            for key, value in checkpoint.items():
                key_layer_match = re.search(r'\.(\d+)\.', key)
                if key_layer_match and key.startswith("transformer_blocks"):
                    key_layer = int(key_layer_match.group(1))
                    if key_layer == layer_index:
                        lora_state_dicts[key] = value

            ip_state_dicts = {}
            for key, value in ip_adapter_checkpoint.items():

                cleaned_key = key[len("attn_processors."):] if key.startswith("attn_processors.") else key
                key_layer_match = re.search(r'\.(\d+)\.', cleaned_key)
                if key_layer_match and cleaned_key.startswith("transformer_blocks"):
                    key_layer = int(key_layer_match.group(1))
                    if key_layer == layer_index and (
                            "to_k_ip" in cleaned_key or "to_v_ip" in cleaned_key or "norm_added_k" in cleaned_key):
                        ip_state_dicts[cleaned_key] = value

            processor = MultiDoubleStreamBlockLoraProcessor(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights,
                device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size,
                n_loras=number, cross_attention_dim=4096, heads=24, scale=scale
            )

            lora_loaded = set()
            lora_missing = set()
            for n in range(number):
                for param_type in ["q", "k", "v", "proj"]:
                    for weight_type in ["down", "up"]:
                        param_name = f"{name}.{param_type}_loras.{n}.{weight_type}.weight"
                        try:
                            target = getattr(processor, f"{param_type}_loras")[n]
                            target_weight = getattr(target, weight_type).weight
                        except AttributeError:
                            lora_missing.add(
                                f"{block_type} {name} processor missing {param_type}_loras.{n}.{weight_type}")
                            continue

                        value = lora_state_dicts.get(param_name)
                        if value is not None:
                            target_weight.data = value.to(device)
                            lora_loaded.add(param_name)
                        else:
                            lora_missing.add(f"{block_type} {name} missing LoRA parameter: {param_name}")

            ip_loaded = set()
            ip_missing = set()
            ip_params = [("to_k_ip", "weight"), ("to_v_ip", "weight"), ("norm_added_k", "weight"),
                         ("norm_added_k", "bias")]
            for param_base, weight_type in ip_params:
                param_name = f"{name}.{param_base}.{weight_type}"
                try:
                    target_module = getattr(processor, param_base)
                    target_param = getattr(target_module, weight_type)
                except AttributeError:
                    ip_missing.add(f"{block_type} {name} processor missing {param_base}.{weight_type}")
                    continue

                value = ip_state_dicts.get(param_name)
                if value is not None:
                    target_param.data = value.to(device)
                    ip_loaded.add(param_name)
                else:
                    ip_missing.add(f"{block_type} {name} missing IP-Adapter parameter: {param_name}")

            current_stats["lora_loaded"].update(lora_loaded)
            current_stats["lora_missing"].update(lora_missing)
            current_stats["ip_loaded"].update(ip_loaded)
            current_stats["ip_missing"].update(ip_missing)

            lora_attn_procs[name] = processor
            processor.to(device)
            print(
                f"{block_type} {name} loaded (LoRA: {len(lora_loaded)}/{len(lora_loaded) + len(lora_missing)}, IP-Adapter: {len(ip_loaded)}/{len(ip_loaded) + len(ip_missing)})")

        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
            block_type = "Single Stream Block"
            current_stats = stats["single"]

            lora_state_dicts = {}
            for key, value in checkpoint.items():
                key_layer_match = re.search(r'\.(\d+)\.', key)
                if key_layer_match and key.startswith("single_transformer_blocks"):
                    key_layer = int(key_layer_match.group(1))
                    if key_layer == layer_index:
                        lora_state_dicts[key] = value

            ip_state_dicts = {}
            for key, value in ip_adapter_checkpoint.items():
                cleaned_key = key[len("attn_processors."):] if key.startswith("attn_processors.") else key
                key_layer_match = re.search(r'\.(\d+)\.', cleaned_key)
                if key_layer_match and cleaned_key.startswith("single_transformer_blocks"):
                    key_layer = int(key_layer_match.group(1))
                    if key_layer == layer_index and (
                            "to_k_ip" in cleaned_key or "to_v_ip" in cleaned_key or "norm_added_k" in cleaned_key):
                        ip_state_dicts[cleaned_key] = value

            processor = MultiSingleStreamBlockLoraProcessor(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights,
                device=device, dtype=torch.bfloat16, cond_width=cond_size, cond_height=cond_size,
                n_loras=number, cross_attention_dim=4096, heads=24, scale=scale
            )

            lora_loaded = set()
            lora_missing = set()
            for n in range(number):
                for param_type in ["q", "k", "v"]:
                    for weight_type in ["down", "up"]:
                        param_name = f"{name}.{param_type}_loras.{n}.{weight_type}.weight"
                        try:
                            target = getattr(processor, f"{param_type}_loras")[n]
                            target_weight = getattr(target, weight_type).weight
                        except AttributeError:
                            lora_missing.add(
                                f"{block_type} {name} processor missing {param_type}_loras.{n}.{weight_type}")
                            continue

                        value = lora_state_dicts.get(param_name)
                        if value is not None:
                            target_weight.data = value.to(device)
                            lora_loaded.add(param_name)
                        else:
                            lora_missing.add(f"{block_type} {name} missing LoRA parameter: {param_name}")

            ip_loaded = set()
            ip_missing = set()
            ip_params = [("to_k_ip", "weight"), ("to_v_ip", "weight"), ("norm_added_k", "weight"),
                         ("norm_added_k", "bias")]
            for param_base, weight_type in ip_params:
                param_name = f"{name}.{param_base}.{weight_type}"
                try:
                    target_module = getattr(processor, param_base)
                    target_param = getattr(target_module, weight_type)
                except AttributeError:
                    ip_missing.add(f"{block_type} {name} processor missing {param_base}.{weight_type}")
                    continue

                value = ip_state_dicts.get(param_name)
                if value is not None:
                    target_param.data = value.to(device)
                    ip_loaded.add(param_name)
                else:
                    ip_missing.add(f"{block_type} {name} missing IP-Adapter parameter: {param_name}")

            current_stats["lora_loaded"].update(lora_loaded)
            current_stats["lora_missing"].update(lora_missing)
            current_stats["ip_loaded"].update(ip_loaded)
            current_stats["ip_missing"].update(ip_missing)

            lora_attn_procs[name] = processor
            processor.to(device)
            print(
                f"{block_type} {name} loaded (LoRA: {len(lora_loaded)}/{len(lora_loaded) + len(lora_missing)}, IP-Adapter: {len(ip_loaded)}/{len(ip_loaded) + len(ip_missing)})")

        else:
            lora_attn_procs[name] = FluxAttnProcessor2_0()

    transformer.set_attn_processor(lora_attn_procs)

    print("\n" + "=" * 50)
    print("Single and Double Stream Block Loading Summary")
    print("=" * 50)
    print_block_stats("Double Stream Block", stats["double"])
    print_block_stats("Single Stream Block", stats["single"])

    print("\n" + "=" * 50)
    print("IP-Adapter Image Projection Network (mlp_proj_model) Loading Results")
    print("=" * 50)
    mlp_loaded = set()
    mlp_missing = set()
    if mlp_proj_model is not None:
        mlp_state_dicts = {}
        for key, value in ip_adapter_checkpoint.items():
            if key.startswith("image_proj."):
                cleaned_key = key[len("image_proj."):]
                mlp_state_dicts[cleaned_key] = value.to(device)

        if mlp_state_dicts:
            load_result = mlp_proj_model.load_state_dict(mlp_state_dicts, strict=False)
            mlp_loaded = set(mlp_state_dicts.keys()) - set(load_result.unexpected_keys)
            mlp_missing = set(load_result.missing_keys)
            print(f"Loaded parameters: {len(mlp_loaded)} (first 5: {list(mlp_loaded)[:5]})")
            print(f"Missing parameters: {len(mlp_missing)} (first 5: {list(mlp_missing)[:5]})")
        else:
            print("No image_proj related parameters found, loading failed")
        mlp_proj_model.to(device)
    else:
        print("mlp_proj_model not provided, skipping loading")

    total_lora_loaded = len(stats["double"]["lora_loaded"]) + len(stats["single"]["lora_loaded"])
    total_lora_missing = len(stats["double"]["lora_missing"]) + len(stats["single"]["lora_missing"])
    total_ip_loaded = len(stats["double"]["ip_loaded"]) + len(stats["single"]["ip_loaded"]) + len(mlp_loaded)
    total_ip_missing = len(stats["double"]["ip_missing"]) + len(stats["single"]["ip_missing"]) + len(mlp_missing)

    print("\n" + "=" * 50)
    print("Overall Loading Statistics")
    print("=" * 50)
    print(f"Total LoRA Loaded: {total_lora_loaded}, Total Missing: {total_lora_missing}")
    print(f"Total IP-Adapter Loaded: {total_ip_loaded}, Total Missing: {total_ip_missing}")


# def set_lora_and_ip_adapter(transformer, mlp_proj_model, lora_path, ip_adapter_path, lora_weights=[], cond_size=512,
#                             device="cuda"):
#     lora_checkpoint = load_checkpoint(lora_path)
#     ip_adapter_checkpoint = load_checkpoint(ip_adapter_path)
#
#     update_model_with_lora_and_ip_adapter(
#         checkpoint=lora_checkpoint,
#         lora_weights=lora_weights,
#         ip_adapter_checkpoint=ip_adapter_checkpoint,
#         transformer=transformer,
#         mlp_proj_model=mlp_proj_model,
#         cond_size=cond_size,
#         device=device
#     )
#
#     print(f"Successfully loaded LoRA ({lora_path}) and IP-Adapter ({ip_adapter_path})")