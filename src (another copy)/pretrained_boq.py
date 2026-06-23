from pathlib import Path
from urllib.error import URLError

import torch


OFFICIAL_PRETRAINED_BOQ_SPECS = {
    "resnet50": {
        "backbone_name": "resnet50",
        "compatible_backbone_names": {"resnet50"},
        "output_dim": 16384,
        "proj_channels": 512,
        "num_queries": 64,
        "num_layers": 2,
        "model_url": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth",
    },
    "dinov2": {
        "backbone_name": "dinov2",
        "compatible_backbone_names": {"dinov2", "dinov2_vitb14"},
        "output_dim": 12288,
        "proj_channels": 384,
        "num_queries": 64,
        "num_layers": 2,
        "model_url": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth",
    },
}


def resolve_official_pretrained_boq_backbone(backbone_name):
    backbone_name = str(backbone_name).lower()
    if "dinov2" in backbone_name:
        return "dinov2"
    if backbone_name == "resnet50":
        return "resnet50"
    raise ValueError(
        f"Official pre-trained BoQ weights are only available for {list(OFFICIAL_PRETRAINED_BOQ_SPECS.keys())}, "
        f"got {backbone_name!r}."
    )


def get_official_pretrained_boq_spec(backbone_name, output_dim=None):
    requested_backbone_name = str(backbone_name).lower()
    resolved_backbone_name = resolve_official_pretrained_boq_backbone(requested_backbone_name)
    spec = dict(OFFICIAL_PRETRAINED_BOQ_SPECS[resolved_backbone_name])
    compatible_backbone_names = spec.get("compatible_backbone_names")
    if compatible_backbone_names is not None and requested_backbone_name not in compatible_backbone_names:
        raise ValueError(
            f"Official pre-trained BoQ for {resolved_backbone_name} is only compatible with "
            f"{sorted(compatible_backbone_names)}, got {backbone_name!r}."
        )
    if output_dim is not None and int(output_dim) != spec["output_dim"]:
        raise ValueError(
            f"Official pre-trained BoQ for {resolved_backbone_name} requires output_dim={spec['output_dim']}, "
            f"got {output_dim}."
        )
    return spec


def _official_pretrained_boq_cache_path(spec):
    return Path(torch.hub.get_dir()) / "checkpoints" / Path(spec["model_url"]).name


def load_official_pretrained_boq_state_dict(backbone_name, output_dim=None, map_location="cpu", checkpoint_path=None):
    spec = get_official_pretrained_boq_spec(backbone_name, output_dim=output_dim)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Pre-trained BoQ checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        spec = dict(spec)
        spec["checkpoint_path"] = str(checkpoint_path)
        spec["checkpoint_source"] = "local"
        return state_dict, spec

    try:
        state_dict = torch.hub.load_state_dict_from_url(
            spec["model_url"],
            map_location=map_location,
        )
    except URLError as exc:
        cache_path = _official_pretrained_boq_cache_path(spec)
        raise RuntimeError(
            "Failed to download official pre-trained BoQ weights. "
            f"URL: {spec['model_url']} | cache: {cache_path} | error: {exc}. "
            "If you are offline, rerun with --scratch_boq to train from scratch, "
            "or pass --pretrained_boq_path with a local checkpoint file."
        ) from exc
    return state_dict, spec


def _extract_submodule_state_dict(state_dict, prefix):
    prefix = prefix.rstrip(".") + "."
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _filter_state_dict_for_module(module, state_dict):
    module_state = module.state_dict()
    filtered_state_dict = {}
    skipped_keys = {}

    for key, value in state_dict.items():
        if key not in module_state:
            skipped_keys[key] = "missing_in_module"
            continue

        expected_value = module_state[key]
        if expected_value.shape != value.shape:
            skipped_keys[key] = (
                f"shape_mismatch(checkpoint={tuple(value.shape)}, module={tuple(expected_value.shape)})"
            )
            continue

        filtered_state_dict[key] = value

    return filtered_state_dict, skipped_keys


def _load_module_state(module, state_dict, strict=True):
    filtered_state_dict, skipped_keys = _filter_state_dict_for_module(module, state_dict)
    incompatible = module.load_state_dict(filtered_state_dict, strict=False)
    ignored_missing_keys = []

    if hasattr(module, "get_missing_state_dict_keys_to_ignore"):
        ignored_key_set = set(module.get_missing_state_dict_keys_to_ignore())
        ignored_missing_keys = [key for key in incompatible.missing_keys if key in ignored_key_set]

    missing_keys = [key for key in incompatible.missing_keys if key not in set(ignored_missing_keys)]

    report = {
        "loaded_keys": len(filtered_state_dict),
        "missing_keys": missing_keys,
        "unexpected_keys": list(incompatible.unexpected_keys),
        "skipped_keys": skipped_keys,
    }
    if ignored_missing_keys:
        report["initialized_keys"] = ignored_missing_keys

    if strict and (report["missing_keys"] or report["unexpected_keys"] or report["skipped_keys"]):
        problems = []
        if report["missing_keys"]:
            problems.append(f"missing_keys={report['missing_keys']}")
        if report["unexpected_keys"]:
            problems.append(f"unexpected_keys={report['unexpected_keys']}")
        if report["skipped_keys"]:
            problems.append(f"skipped_keys={report['skipped_keys']}")
        raise RuntimeError(
            f"Failed to strictly load official pre-trained BoQ weights into {module.__class__.__name__}: "
            + ", ".join(problems)
        )

    return report


def _build_initialized_module_report(parameter_keys):
    return {
        "loaded_keys": 0,
        "missing_keys": [],
        "unexpected_keys": [],
        "skipped_keys": {},
        "initialized_keys": list(parameter_keys),
    }


def load_pretrained_boq_weights(
    backbone,
    aggregator,
    backbone_name,
    output_dim=None,
    map_location="cpu",
    strict=True,
    checkpoint_path=None,
):
    state_dict, spec = load_official_pretrained_boq_state_dict(
        backbone_name,
        output_dim=output_dim,
        map_location=map_location,
        checkpoint_path=checkpoint_path,
    )
    backbone_state_dict = _extract_submodule_state_dict(state_dict, "backbone")
    aggregator_state_dict = _extract_submodule_state_dict(state_dict, "aggregator")

    return {
        "spec": spec,
        "backbone": _load_module_state(backbone, backbone_state_dict, strict=strict),
        "aggregator": _load_module_state(aggregator, aggregator_state_dict, strict=strict),
    }


def load_pretrained_boq_into_dual_branch_encoder(
    image_encoder,
    backbone_name,
    output_dim=None,
    map_location="cpu",
    strict=True,
    checkpoint_path=None,
):
    state_dict, spec = load_official_pretrained_boq_state_dict(
        backbone_name,
        output_dim=output_dim,
        map_location=map_location,
        checkpoint_path=checkpoint_path,
    )
    backbone_state_dict = _extract_submodule_state_dict(state_dict, "backbone")
    aggregator_state_dict = _extract_submodule_state_dict(state_dict, "aggregator")

    return {
        "spec": spec,
        "q_backbone": _load_module_state(image_encoder.q_backbone, backbone_state_dict, strict=strict),
        "db_backbone": _load_module_state(image_encoder.db_backbone, backbone_state_dict, strict=strict),
        "shared_aggregator": _load_module_state(image_encoder.shared_aggregator, aggregator_state_dict, strict=strict),
    }


def load_pretrained_boq_into_shared_query_encoder(
    image_encoder,
    backbone_name,
    output_dim=None,
    map_location="cpu",
    strict=True,
    checkpoint_path=None,
):
    state_dict, spec = load_official_pretrained_boq_state_dict(
        backbone_name,
        output_dim=output_dim,
        map_location=map_location,
        checkpoint_path=checkpoint_path,
    )
    backbone_state_dict = _extract_submodule_state_dict(state_dict, "backbone")
    aggregator_state_dict = _extract_submodule_state_dict(state_dict, "aggregator")

    branch_boq_state_dict = {}
    shared_head_state_dict = {}
    shared_query_bank_state_dict = {}
    for key, value in aggregator_state_dict.items():
        if key.startswith("fc."):
            shared_head_state_dict[key] = value
            continue

        if key.startswith("boqs.") and key.endswith(".queries"):
            layer_idx = key.split(".")[1]
            shared_query_bank_state_dict[f"queries.{layer_idx}"] = value
            continue

        branch_boq_state_dict[key] = value

    return {
        "spec": spec,
        "ground_backbone": _load_module_state(image_encoder.ground_backbone, backbone_state_dict, strict=strict),
        "satellite_backbone": _load_module_state(image_encoder.satellite_backbone, backbone_state_dict, strict=strict),
        "ground_boq": _load_module_state(image_encoder.ground_boq, branch_boq_state_dict, strict=strict),
        "satellite_boq": _load_module_state(image_encoder.satellite_boq, branch_boq_state_dict, strict=strict),
        "shared_query_bank": _load_module_state(image_encoder.shared_query_bank, shared_query_bank_state_dict, strict=strict),
        "shared_head": _load_module_state(image_encoder.shared_head, shared_head_state_dict, strict=strict),
    }


def load_pretrained_boq_into_view_adapter_encoder(
    image_encoder,
    backbone_name,
    output_dim=None,
    map_location="cpu",
    strict=False,
    checkpoint_path=None,
):
    state_dict, spec = load_official_pretrained_boq_state_dict(
        backbone_name,
        output_dim=output_dim,
        map_location=map_location,
        checkpoint_path=checkpoint_path,
    )
    backbone_state_dict = _extract_submodule_state_dict(state_dict, "backbone")
    aggregator_state_dict = _extract_submodule_state_dict(state_dict, "aggregator")

    return {
        "spec": spec,
        "shared_backbone": _load_module_state(image_encoder.backbone, backbone_state_dict, strict=strict),
        "shared_aggregator": _load_module_state(image_encoder.aggregator, aggregator_state_dict, strict=strict),
        "ground_adapters": _build_initialized_module_report(image_encoder.get_adapter_state_keys("ground")),
        "satellite_adapters": _build_initialized_module_report(image_encoder.get_adapter_state_keys("satellite")),
    }
