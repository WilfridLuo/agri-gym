"""Runtime construction helpers for the main benchmark entry point."""

import json
import os
from pathlib import Path

from data import AGMMULoader
from tools import (
    CropImageTool,
    DrawBoundingBoxTool,
    EmbeddingRetrieveTool,
    GetManagementAdviceTool,
    IdentifyDiseaseTool,
    ImageMetadataTool,
    RetrieveMultimodalEvidenceTool,
    SearchKnowledgeBaseTool,
    ToolRegistry,
    TriageAgriIssueTool,
    VisualRetrieveTool,
    ZoomImageTool,
)
from vlm import BaseVLM, VLMConfig, VLMFactory, VLMType


def create_tool_registry(agmmu_loader: AGMMULoader, emb_dir: str | None = None, device: str = "cuda") -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ImageMetadataTool())
    registry.register(CropImageTool())
    registry.register(ZoomImageTool())
    registry.register(DrawBoundingBoxTool())
    registry.register(TriageAgriIssueTool())

    shared_embedding_tool = None
    if emb_dir:
        emb_path = Path(emb_dir)
        if not emb_path.is_absolute():
            emb_path = Path(__file__).parent.parent.parent / emb_path
        if emb_path.exists():
            shared_embedding_tool = EmbeddingRetrieveTool(str(emb_path), device=device)
            registry.shared_resources["embedding_retriever"] = shared_embedding_tool

    if agmmu_loader.items or getattr(agmmu_loader, "facts", []):
        registry.register(SearchKnowledgeBaseTool(agmmu_loader))
        registry.register(VisualRetrieveTool(agmmu_loader, emb_dir=emb_dir, device=device, embedding_tool=shared_embedding_tool))
        registry.register(RetrieveMultimodalEvidenceTool(agmmu_loader, emb_dir=emb_dir, device=device, embedding_tool=shared_embedding_tool))
        registry.register(IdentifyDiseaseTool(agmmu_loader))
        registry.register(GetManagementAdviceTool(agmmu_loader))
    if shared_embedding_tool is not None:
        registry.register(shared_embedding_tool)
    return registry


def _default_generation_max_tokens(model_name: str) -> int:
    model_key = (model_name or "").lower().replace("_", "-")
    if "deepseek-vl2" in model_key:
        return 768
    return 1024


def create_vlm(
    backend: str,
    model_type: str,
    model_name: str,
    api_key: str = None,
    api_base: str = None,
    model_path: str = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    local_files_only: bool = False,
    max_tokens: int | None = None,
) -> BaseVLM:
    local_type_map = {
        "qwen3": VLMType.QWEN3_VL,
        "qwen2.5": VLMType.QWEN2_5_VL,
        "gemma": VLMType.GEMMA_VL,
    }
    api_type_map = {
        "openai": VLMType.GPT4V_API,
        "gpt4v": VLMType.GPT4V_API,
        "claude": VLMType.CLAUDE_API,
        "gemini": VLMType.GEMINI_API,
    }
    api_key_envs = {
        "openai": ("OPENAI_API_KEY",),
        "gpt4v": ("OPENAI_API_KEY",),
        "claude": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }
    generation_max_tokens = int(max_tokens) if max_tokens is not None else _default_generation_max_tokens(model_name)

    if backend == "local":
        if model_type not in local_type_map:
            raise ValueError(f"Unsupported local model_type: {model_type}")
        resolved_path = model_path or model_name
        if not resolved_path:
            raise ValueError("Local backend requires model_name or model_path")

        config = VLMConfig(
            model_type=local_type_map[model_type],
            model_name=model_name,
            model_path=resolved_path,
            api_key=None,
            device=device,
            dtype=dtype,
            temperature=0.2,
            max_tokens=generation_max_tokens,
            local_files_only=local_files_only,
        )
        return VLMFactory.create(config)

    if backend == "api":
        if model_type not in api_type_map:
            raise ValueError(f"Unsupported API model_type: {model_type}")
        resolved_api_key = None if str(api_key or "").strip().upper() == "EMPTY" else api_key
        if not resolved_api_key:
            for env_name in api_key_envs.get(model_type, ()):
                resolved_api_key = os.environ.get(env_name)
                if resolved_api_key:
                    break
        if not resolved_api_key:
            env_hint = ", ".join(api_key_envs.get(model_type, ("API key env var",)))
            raise ValueError(
                f"API backend for {model_type} requires --api-key/--judge-api-key or one of: {env_hint}"
            )

        config = VLMConfig(
            model_type=api_type_map[model_type],
            model_name=model_name,
            model_path=None,
            api_key=resolved_api_key,
            api_base=api_base,
            temperature=0.2,
            max_tokens=generation_max_tokens,
        )
        return VLMFactory.create(config)

    if backend == "vllm":
        config = VLMConfig(
            model_type=VLMType.VLLM_API,
            model_name=model_name,
            model_path=None,
            api_key=api_key,
            api_base=api_base,
            device=device,
            dtype=dtype,
            temperature=0.2,
            max_tokens=generation_max_tokens,
        )
        return VLMFactory.create(config)

    raise ValueError(f"Unsupported backend: {backend}")


def resolve_processed_kb_from_manifest(manifest_path: Path) -> Path | None:
    if not manifest_path.exists() or manifest_path.suffix != ".json":
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    processed = data.get("processed_kb")
    if not processed:
        return None

    p = Path(str(processed))
    if not p.is_absolute():
        p = manifest_path.parent / p
    elif not p.exists():
        sibling = manifest_path.parent / p.name
        if sibling.exists():
            p = sibling
    return p


def cached_repo_exists(model_name: str) -> bool:
    if not model_name or "/" not in model_name:
        return False

    hub_root = os.environ.get("HF_HUB_CACHE")
    if not hub_root:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            hub_root = str(Path(hf_home) / "hub")

    if not hub_root:
        return False

    org, repo = model_name.split("/", 1)
    repo_dir = Path(hub_root) / f"models--{org}--{repo}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists() or not snapshots.is_dir():
        return False

    snapshot_dirs = [p for p in snapshots.iterdir() if p.is_dir()]
    if not snapshot_dirs:
        return False

    required_any = ["config.json", "tokenizer_config.json", "preprocessor_config.json", "processor_config.json"]
    for snap in snapshot_dirs:
        if any((snap / f).exists() for f in required_any):
            return True
    return False


def decide_local_files_only(
    backend: str,
    model_name: str,
    model_path: str,
    force_local_files_only: bool,
) -> bool:
    if backend != "local":
        return False

    if force_local_files_only:
        return True

    if model_path:
        return True

    return cached_repo_exists(model_name)
