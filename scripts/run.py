#!/usr/bin/env python3
"""
Agricultural VLM Agent Framework - Main Entry Point

"""
import os

# Optional runtime configuration is read from the shell environment or a local .env file.
# Do not hard-code credentials, cache paths, or machine-specific directories in source.

import sys
import argparse
import atexit
import json
import random
import re
import shutil
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks import MMSTDatasetLoader
from agent import LLMAgent
from data import AGMMULoader
from env import AgriEnv, AgriTask
from evaluator import (
    AgriRolloutRunner,
    LLMJudge,
    create_tool_registry,
    create_vlm,
    decide_local_files_only,
)
from tools import clear_in_memory_images, register_in_memory_image


def load_local_api_keys(path_value: str | None = None) -> list[str]:
    """Load local API keys from untracked env files without overriding shell env."""
    candidates = []
    if path_value:
        candidates.append(Path(path_value))
    candidates.extend([Path(".env")])
    seen = set()
    loaded = []
    for path in candidates:
        if not path.is_absolute():
            path = Path.cwd() / path
        path_key = str(path)
        if path_key in seen:
            continue
        seen.add(path_key)
        if not path.exists() or not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
        loaded.append(str(path))
    return loaded


def _api_key_available(model_type: str, explicit_key: str | None = None, extra_env: tuple[str, ...] = ()) -> bool:
    if explicit_key and str(explicit_key).strip().upper() != "EMPTY":
        return True
    if any(os.environ.get(name) for name in extra_env):
        return True
    env_names = {
        "openai": ("OPENAI_API_KEY",),
        "gpt4v": ("OPENAI_API_KEY",),
        "claude": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }.get(model_type, ())
    return any(os.environ.get(name) for name in env_names)


DEFAULT_CONFIG = {
    "backend": "vllm", # "api" or "vllm"
    "model_type": "qwen", # "openai", "claude", "gemini", or "qwen" (for local vLLM)
    "model_name": "Qwen/Qwen3-VL-2B-Instruct", # Qwen/Qwen3-VL-8B-Instruct
    "model_path": None,
    "api_key": "EMPTY",
    "api_base": None,
    "api_key_file": ".env",
    "agent_max_tokens": None,
    "launch_vllm": True,
    "vllm_model_name": None,
    "vllm_host": "127.0.0.1",
    "vllm_port": 8000,
    "vllm_dtype": "bfloat16",
    "vllm_gpu_memory_utilization": 0.4,
    "vllm_max_model_len": 24576,
    "vllm_enforce_eager": True,
    "vllm_enable_chunked_prefill": False,
    "vllm_enable_prefix_caching": False,
    "vllm_extra_args": "",
    "vllm_startup_timeout": 240,
    "vllm_log_dir": None,
    "device": "cuda",
    "dtype": "bfloat16",
    "local_files_only": False,
    "judge_backend": "api", # "api, vllm"
    "judge_model_type": "openai", # "openai", "claude", "gemini", or "qwen" (for local vLLM)
    "judge_model_name": "gpt-4o",# Qwen/Qwen3-VL-2B-Instruct
    "judge_model_path": None,
    "judge_api_key": None,
    "judge_api_base": None,
    "judge_vllm_host": "127.0.0.1",
    "judge_vllm_port": 8001,
    "judge_vllm_model_name": None,
    "judge_vllm_gpu_memory_utilization": 0.3,
    "judge_vllm_max_model_len": 16384,
    "judge_vllm_enforce_eager": True,
    "judge_vllm_enable_chunked_prefill": False,
    "judge_vllm_enable_prefix_caching": False,
    "judge_vllm_extra_args": "",
    "separate_judge_model": True,
    "run_judge": True,
    "judge_pass_score": 7.0,
    "judge_mode": "score_only",
    "limit": None,
    "sample_seed": 42,
    "sample_mode": "seeded",
    "batch_size": 32,
    "concurrency": 16,
    "judge_concurrency": 16,
    "serialize_vlm": None,
    "serialize_judge_vlm": None,
    "serialize_tools": True,
    "preload_embedding": True,
    "task": "eval",
    "query": None,
    "query_image": None,
    "gt_answer": None,
    "show_trajectory": False,
    "split": "test",
    "mmst_subset": "MMST_Standard",
    "mmst_json": None,
    "agmmu_path": "data/agmmu_ft_hf1.json",
    "kb_processed_json": "data/kb/emb_full_8b/agmmu_kb_45096.json",
    "kb_emb_dir": "data/kb/emb_full_8b",
    "kb_image_root": None,
    "start_index": 0,
    "skip_errors": True,
    "resume_skip_completed": False,
    "save_interval": 10,
    "append_output": False,
    "output_dir": "mirage_mmst_output",
    "visual_output_dir": None,
    "max_steps": 5,
    "max_prompt_images": -1,
    "max_history_messages": 4,
    "agent_id": "001",
}


_VLLM_PROCESSES: list[subprocess.Popen] = []
_VLLM_LOG_HANDLES = []


def _vllm_api_base(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def _vllm_healthcheck(api_base: str, api_key: str, timeout: float = 5.0) -> bool:
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _vllm_model_ids(api_base: str, api_key: str, timeout: float = 5.0) -> list[str]:
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    model_ids = []
    for item in payload.get("data", []):
        if isinstance(item, dict) and item.get("id"):
            model_ids.append(str(item["id"]))
    return model_ids


def validate_vllm_model_names(args) -> None:
    """Fail fast when a vLLM endpoint is serving a different model than requested."""
    if args.backend == "vllm" and args.api_base:
        served = _vllm_model_ids(args.api_base, args.api_key)
        if served and args.model_name not in served:
            raise RuntimeError(
                f"Agent model {args.model_name!r} is not served by {args.api_base}. "
                f"Available models: {served}"
            )

    if not (args.run_judge and args.judge_backend == "vllm" and args.judge_api_base):
        return

    judge_served = _vllm_model_ids(args.judge_api_base, args.api_key)
    if not judge_served:
        return

    if args.judge_model_name in judge_served:
        return

    raise RuntimeError(
        f"Judge model {args.judge_model_name!r} is not served by {args.judge_api_base}. "
        f"Available models: {judge_served}"
    )


def _cleanup_vllm_process() -> None:
    global _VLLM_PROCESSES, _VLLM_LOG_HANDLES
    for process in _VLLM_PROCESSES:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
    _VLLM_PROCESSES = []
    for handle in _VLLM_LOG_HANDLES:
        try:
            handle.close()
        except Exception:
            pass
    _VLLM_LOG_HANDLES = []


def _vllm_extra_arg_tokens(extra_args: str | None) -> list[str]:
    if not extra_args:
        return []
    try:
        return shlex.split(str(extra_args))
    except ValueError:
        return str(extra_args).split()


def _has_vllm_option(extra_args: str | None, option: str) -> bool:
    tokens = _vllm_extra_arg_tokens(extra_args)
    return any(token == option or token.startswith(f"{option}=") for token in tokens)


def _append_vllm_extra_args(existing: str | None, addition: str) -> str:
    existing = (existing or "").strip()
    addition = addition.strip()
    if not addition:
        return existing
    if not existing:
        return addition
    return f"{existing} {addition}"


def _vllm_config_path(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "vllm" / name


def _deepseek_vl2_small_vllm_config_path() -> Path:
    return _vllm_config_path("deepseek-vl2-small")


def _deepseek_vl2_vllm_config_path() -> Path:
    return _vllm_config_path("deepseek-vl2")


def _is_deepseek_vl2_base_model(model_key: str) -> bool:
    return model_key in {"deepseek-vl2", "deepseek-ai/deepseek-vl2"}


def _is_gemma3_12b_model(model_key: str) -> bool:
    return "gemma-3-12b" in model_key


def _apply_vllm_model_family_defaults(
    args,
    *,
    label: str,
    model_name: str,
    max_model_len_attr: str,
    extra_args_attr: str,
    gpu_memory_utilization_attr: str | None = None,
) -> None:
    """Apply narrowly scoped vLLM defaults required by specific model families."""
    model_key = (model_name or "").lower().replace("_", "-")
    is_deepseek_vl2 = "deepseek-vl2" in model_key
    is_gemma3_12b = _is_gemma3_12b_model(model_key)
    if not is_deepseek_vl2 and not is_gemma3_12b:
        return

    if is_gemma3_12b:
        # Gemma 3 12B loads about 23 GiB of weights in bf16. The project-wide
        # default 0.3 GPU memory budget leaves no KV cache room on a 96 GB GPU,
        # especially with the default 24k context and multimodal encoder cache.
        max_len = getattr(args, max_model_len_attr, None)
        try:
            if max_len is not None and int(max_len) > 12288:
                print(
                    f"{label} Gemma 3 12B detected: lowering {max_model_len_attr} "
                    f"from {max_len} to 12288 to keep vLLM cache allocation stable."
                )
                setattr(args, max_model_len_attr, 12288)
        except (TypeError, ValueError):
            pass

        if gpu_memory_utilization_attr:
            gpu_util = getattr(args, gpu_memory_utilization_attr, None)
            try:
                if gpu_util is not None and float(gpu_util) < 0.55:
                    print(
                        f"{label} Gemma 3 12B detected: increasing {gpu_memory_utilization_attr} "
                        f"from {gpu_util} to 0.55 for KV cache blocks on a 96 GB GPU."
                    )
                    setattr(args, gpu_memory_utilization_attr, 0.55)
            except (TypeError, ValueError):
                pass

    if not is_deepseek_vl2:
        return

    # DeepSeek-VL2 config exposes max_position_embeddings=4096. vLLM rejects
    # larger max_model_len values unless VLLM_ALLOW_LONG_MAX_MODEL_LEN is used,
    # which is unsafe here. Keep the server context at the model limit.
    max_len = getattr(args, max_model_len_attr, None)
    try:
        if max_len is not None and int(max_len) > 4096:
            print(
                f"{label} DeepSeek-VL2 detected: lowering {max_model_len_attr} "
                f"from {max_len} to 4096 to match model config."
            )
            setattr(args, max_model_len_attr, 4096)
    except (TypeError, ValueError):
        pass

    if _is_deepseek_vl2_base_model(model_key) and gpu_memory_utilization_attr:
        gpu_util = getattr(args, gpu_memory_utilization_attr, None)
        try:
            if gpu_util is not None and float(gpu_util) < 0.70:
                print(
                    f"{label} DeepSeek-VL2 base detected: increasing {gpu_memory_utilization_attr} "
                    f"from {gpu_util} to 0.70 so the 51 GiB bf16 checkpoint leaves KV cache room."
                )
                setattr(args, gpu_memory_utilization_attr, 0.70)
        except (TypeError, ValueError):
            pass

    extra_args = getattr(args, extra_args_attr, "")

    if not _has_vllm_option(extra_args, "--hf-config-path"):
        if "deepseek-vl2-small" in model_key:
            config_path = _deepseek_vl2_small_vllm_config_path()
            config_label = "DeepSeek-VL2-Small"
        elif _is_deepseek_vl2_base_model(model_key):
            config_path = _deepseek_vl2_vllm_config_path()
            config_label = "DeepSeek-VL2"
        else:
            config_path = None
            config_label = None

        if config_path is not None:
            if config_path.exists():
                extra_args = _append_vllm_extra_args(
                    extra_args,
                    f"--hf-config-path {shlex.quote(str(config_path))}",
                )
                print(f"{label} {config_label} detected: using patched vLLM config {config_path}")
            else:
                print(f"Warning: {config_label} patched vLLM config not found: {config_path}")

    if not _has_vllm_option(extra_args, "--moe-backend"):
        # On sm120/Blackwell, vLLM 0.20.1 may auto-select FlashInfer CUTLASS
        # MoE for DeepSeek-VL2 and crash during startup because the installed
        # FlashInfer wheel cannot JIT that fused MoE kernel. Triton is slower
        # but stable and keeps the benchmark runnable.
        extra_args = _append_vllm_extra_args(extra_args, "--moe-backend triton")

    setattr(args, extra_args_attr, extra_args)


def _ensure_one_vllm_server(
    *,
    label: str,
    api_base: str,
    model_name: str,
    api_key: str,
    host: str,
    port: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
    enable_chunked_prefill: bool | None,
    enable_prefix_caching: bool | None,
    startup_timeout: int,
    log_dir: Path,
    launch_vllm: bool,
    extra_args: str | None = None,
) -> None:
    global _VLLM_PROCESSES, _VLLM_LOG_HANDLES

    if _vllm_healthcheck(api_base, api_key):
        served = _vllm_model_ids(api_base, api_key)
        if served and model_name not in served:
            raise RuntimeError(
                f"{label} vLLM endpoint {api_base} is reachable but does not serve {model_name!r}. "
                f"Available models: {served}"
            )
        print(f"✓ {label} vLLM server already available: {api_base}")
        print(
            "  Note: existing server config is reused; memory/max-len arguments only apply when this script launches it."
        )
        return

    if not launch_vllm:
        raise RuntimeError(
            f"{label} vLLM backend selected but server is not reachable at {api_base}. "
            "Start it manually or run with --launch-vllm."
        )

    vllm_bin = shutil.which("vllm")
    if not vllm_bin:
        raise RuntimeError(
            "vLLM backend selected but `vllm` CLI was not found in this environment. "
            "Install with: python -m pip install vllm"
        )

    cmd = [
        vllm_bin,
        "serve",
        model_name,
        "--dtype",
        dtype,
        "--api-key",
        api_key or "EMPTY",
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if enable_chunked_prefill is not None:
        cmd.append("--enable-chunked-prefill" if enable_chunked_prefill else "--no-enable-chunked-prefill")
    if enable_prefix_caching is not None:
        cmd.append("--enable-prefix-caching" if enable_prefix_caching else "--no-enable-prefix-caching")
    if extra_args:
        cmd.extend(shlex.split(str(extra_args)))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"vllm_{label.lower()}_server.log"
    print(f"Starting {label} vLLM server:")
    print("  " + " ".join(cmd))
    print(f"  {label} vLLM server logs: {log_path}")
    log_handle = open(log_path, "a", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _VLLM_PROCESSES.append(process)
    _VLLM_LOG_HANDLES.append(log_handle)
    atexit.register(_cleanup_vllm_process)

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{label} vLLM server exited early with code {process.returncode}. "
                f"Check {log_path}. If the log says no available memory for cache blocks, increase "
                f"--{'judge-vllm' if label == 'JUDGE' else 'vllm'}-gpu-memory-utilization or reduce "
                f"--{'judge-vllm' if label == 'JUDGE' else 'vllm'}-max-model-len."
            )
        if _vllm_healthcheck(api_base, api_key):
            validate_served = _vllm_model_ids(api_base, api_key)
            if validate_served and model_name not in validate_served:
                raise RuntimeError(
                    f"{label} vLLM server started but does not serve {model_name!r}. "
                    f"Available models: {validate_served}"
                )
            print(f"✓ {label} vLLM server ready: {api_base}")
            return
        time.sleep(2)

    raise RuntimeError(
        f"Timed out waiting for {label} vLLM server at {api_base}. "
        "Try lowering --vllm-gpu-memory-utilization or --vllm-max-model-len."
    )


def ensure_vllm_servers(args) -> None:
    """Start only the vLLM servers required by the selected agent/judge backends."""
    log_dir = Path(args.vllm_log_dir) if args.vllm_log_dir else _configured_output_dir(args) / "logs"
    if not log_dir.is_absolute():
        log_dir = Path.cwd() / log_dir

    if args.backend == "vllm":
        if not args.api_base:
            args.api_base = _vllm_api_base(args.vllm_host, args.vllm_port)
        agent_model = args.vllm_model_name or args.model_name
        args.model_name = agent_model
        _apply_vllm_model_family_defaults(
            args,
            label="AGENT",
            model_name=agent_model,
            max_model_len_attr="vllm_max_model_len",
            extra_args_attr="vllm_extra_args",
            gpu_memory_utilization_attr="vllm_gpu_memory_utilization",
        )
        _ensure_one_vllm_server(
            label="AGENT",
            api_base=args.api_base,
            model_name=agent_model,
            api_key=args.api_key,
            host=args.vllm_host,
            port=args.vllm_port,
            dtype=args.vllm_dtype,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            enforce_eager=args.vllm_enforce_eager,
            enable_chunked_prefill=args.vllm_enable_chunked_prefill,
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            startup_timeout=args.vllm_startup_timeout,
            log_dir=log_dir,
            launch_vllm=args.launch_vllm,
            extra_args=args.vllm_extra_args,
        )

    needs_judge_server = (
        args.run_judge
        and args.separate_judge_model
        and args.judge_backend == "vllm"
    )
    if needs_judge_server:
        if not args.judge_api_base:
            args.judge_api_base = _vllm_api_base(args.judge_vllm_host, args.judge_vllm_port)
        judge_model = args.judge_vllm_model_name or args.judge_model_name
        args.judge_model_name = judge_model
        _apply_vllm_model_family_defaults(
            args,
            label="JUDGE",
            model_name=judge_model,
            max_model_len_attr="judge_vllm_max_model_len",
            extra_args_attr="judge_vllm_extra_args",
            gpu_memory_utilization_attr="judge_vllm_gpu_memory_utilization",
        )
        _ensure_one_vllm_server(
            label="JUDGE",
            api_base=args.judge_api_base,
            model_name=judge_model,
            api_key=args.api_key,
            host=args.judge_vllm_host,
            port=args.judge_vllm_port,
            dtype=args.vllm_dtype,
            gpu_memory_utilization=args.judge_vllm_gpu_memory_utilization,
            max_model_len=args.judge_vllm_max_model_len,
            enforce_eager=args.judge_vllm_enforce_eager,
            enable_chunked_prefill=args.judge_vllm_enable_chunked_prefill,
            enable_prefix_caching=args.judge_vllm_enable_prefix_caching,
            startup_timeout=args.vllm_startup_timeout,
            log_dir=log_dir,
            launch_vllm=args.launch_vllm,
            extra_args=args.judge_vllm_extra_args,
        )


def print_effective_config(args) -> None:
    judge_api_key_present = bool(args.judge_api_key)
    if args.run_judge and args.judge_backend == "api" and not judge_api_key_present:
        env_names = {
            "openai": ("OPENAI_API_KEY",),
            "gpt4v": ("OPENAI_API_KEY",),
            "claude": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
            "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        }.get(args.judge_model_type, ())
        judge_api_key_present = any(os.environ.get(name) for name in env_names)

    config = {
        "agent": {
            "backend": args.backend,
            "model_name": args.model_name,
            "api_base": args.api_base if args.backend == "vllm" else None,
            "has_api_key": _api_key_available(args.model_type, args.api_key, ("AGRI_AGENT_API_KEY",)) if args.backend == "api" else None,
            "serialize_vlm": args.serialize_vlm,
            "max_tokens": args.agent_max_tokens,
        },
        "judge": {
            "enabled": args.run_judge,
            "backend": args.judge_backend if args.run_judge else None,
            "model_type": args.judge_model_type if args.run_judge else None,
            "model_name": args.judge_model_name if args.run_judge else None,
            "api_base": args.judge_api_base if args.run_judge and args.judge_backend in {"api", "vllm"} else None,
            "has_api_key": judge_api_key_present if args.run_judge and args.judge_backend == "api" else None,
            "serialize_judge_vlm": args.serialize_judge_vlm if args.run_judge else None,
        },
        "vllm_server": {
            "launch_vllm": args.launch_vllm,
            "host": args.vllm_host,
            "port": args.vllm_port,
            "dtype": args.vllm_dtype,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "max_model_len": args.vllm_max_model_len,
            "enforce_eager": args.vllm_enforce_eager,
            "enable_chunked_prefill": args.vllm_enable_chunked_prefill,
            "enable_prefix_caching": args.vllm_enable_prefix_caching,
            "extra_args": args.vllm_extra_args,
            "startup_timeout": args.vllm_startup_timeout,
            "log_dir": args.vllm_log_dir or str(Path(args.output_dir) / "logs"),
        },
        "judge_vllm_server": {
            "enabled": bool(args.run_judge and args.separate_judge_model and args.judge_backend == "vllm"),
            "host": args.judge_vllm_host,
            "port": args.judge_vllm_port,
            "model_name": args.judge_vllm_model_name or args.judge_model_name,
            "gpu_memory_utilization": args.judge_vllm_gpu_memory_utilization,
            "max_model_len": args.judge_vllm_max_model_len,
            "enforce_eager": args.judge_vllm_enforce_eager,
            "enable_chunked_prefill": args.judge_vllm_enable_chunked_prefill,
            "enable_prefix_caching": args.judge_vllm_enable_prefix_caching,
            "extra_args": args.judge_vllm_extra_args,
        },
        "scheduler": {
            "type": "threadpool_async",
            "batch_size": args.batch_size,
            "agent_concurrency": args.concurrency,
            "judge_concurrency": args.judge_concurrency,
            "serialize_tools": args.serialize_tools,
            "preload_embedding": args.preload_embedding,
        },
        "agent_prompt": {
            "max_prompt_images": args.max_prompt_images,
            "max_history_messages": args.max_history_messages,
            "max_tokens": args.agent_max_tokens,
        },
        "dataset": {
            "name": "MIRAGE-Benchmark/MIRAGE",
            "subset": args.mmst_subset,
            "split": args.split,
            "limit": args.limit,
            "start_index": args.start_index,
        },
        "kb": {
            "processed_json": args.kb_processed_json,
            "emb_dir": args.kb_emb_dir,
        },
        "output": {
            "output_dir": args.output_dir,
            "visual_output_dir": args.visual_output_dir,
        },
    }
    print("\nEffective main-run config:")
    print(json.dumps(config, ensure_ascii=False, indent=2))


def preload_embedding_retriever(tool_registry) -> None:
    retriever = tool_registry.shared_resources.get("embedding_retriever")
    if retriever is None:
        return
    print("Preloading shared embedding retriever once...")
    if hasattr(retriever, "preload"):
        retriever.preload()
    else:
        retriever._ensure_loaded()
    print(
        "✓ Embedding retriever mounted: "
        f"model={retriever.model_name}, instance_id={id(retriever)}"
    )


class LockedVLM:
    """Serialize calls to a shared local Transformers VLM when needed."""

    def __init__(self, vlm, lock=None):
        self._vlm = vlm
        self._lock = lock or threading.Lock()

    def generate(self, *args, **kwargs):
        with self._lock:
            return self._vlm.generate(*args, **kwargs)

    def encode_image(self, *args, **kwargs):
        return self._vlm.encode_image(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._vlm, name)


class LockedToolRegistry:
    """Serialize shared tool execution when tools hold non-thread-safe model state."""

    def __init__(self, registry, lock=None):
        self._registry = registry
        self._lock = lock or threading.Lock()

    @property
    def tools(self):
        return self._registry.tools

    @property
    def shared_resources(self):
        return self._registry.shared_resources

    def execute(self, *args, **kwargs):
        with self._lock:
            return self._registry.execute(*args, **kwargs)

    def list_tools(self):
        return self._registry.list_tools()

    def get_tools_description(self):
        return self._registry.get_tools_description()


def _resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    p = Path(path_value)
    if not p.is_absolute():
        p = Path(__file__).parent.parent / p
    return p


def load_agmmu_knowledge_base(agmmu_path: str | None, kb_processed_json: str | None = None) -> AGMMULoader:
    """加载 AGMMU 知识库"""
    print(f"\nLoading AGMMU knowledge base...")
    
    loader = AGMMULoader()

    processed_path = _resolve_path(kb_processed_json)
    raw_path = _resolve_path(agmmu_path)

    if processed_path and processed_path.exists():
        print(f"Loading processed KB: {processed_path}")
        loader.load_processed_json(str(processed_path))
    elif raw_path and raw_path.exists():
        print(f"Loading raw KB: {raw_path}")
        loader.load_json(str(raw_path))
    else:
        # 尝试从默认位置加载
        default_paths = [
            Path(__file__).parent.parent / "data" / "kb" / "emb_full_8b" / "agmmu_kb_45096.json",
            Path(__file__).parent.parent.parent / "agri" / "agmmu_ft_hf1.json",
            Path.home() / "agmmu_ft_hf1.json",
        ]
        
        found = False
        for default_path in default_paths:
            if default_path.exists():
                if "emb_full_8b" in str(default_path):
                    loader.load_processed_json(str(default_path))
                else:
                    loader.load_json(str(default_path))
                found = True
                break
        
        if not found:
            print("⚠ AGMMU knowledge base not found (optional)")
    
    if loader.items:
        stats = loader.stats()
        print(f"✓ Knowledge base stats:")
        print(f"  - Total items: {stats['total_items']}")
        print(f"  - Species: {stats['total_species']}")
    
    return loader


def _build_tasks_from_items(items, subset: str = "MMST_Standard"):
    tasks = []
    for item in items:
        image_refs = list(item.image_paths or ([item.image_path] if item.image_path else []))
        image_inputs = list(getattr(item, "image_inputs", []) or [])
        for ref, image in zip(image_refs, image_inputs):
            register_in_memory_image(ref, image)
        tasks.append(
            AgriTask(
                task_id=str(item.id),
                query=item.question,
                gt_answer=item.reference_answer,
                image_path=item.image_path,
                image_paths=image_refs,
                image_inputs=image_inputs,
                metadata={
                    **(item.metadata or {}),
                    "source": "MIRAGE",
                    "subset": subset,
                    "image_paths": image_refs,
                },
            )
        )
    return tasks


def _build_tasks_from_mmst(loader, limit=None, subset: str = "MMST_Standard", start_index: int = 0):
    """将 MMST 样本转为 AgriEnv 任务。"""
    if start_index:
        total = len(loader.hf_dataset) if getattr(loader, "hf_dataset", None) is not None else len(loader.data)
        items = loader.get_batch(batch_size=limit or total, offset=start_index)
    else:
        items = loader.get_items(limit=limit)
    return _build_tasks_from_items(items, subset=subset)


def _dataset_size(loader) -> int:
    if getattr(loader, "hf_dataset", None) is not None:
        return len(loader.hf_dataset)
    return len(loader.data)


def _select_dataset_indices(total_available: int, start_index: int, requested: int, sample_seed: int | None, sample_mode: str) -> list[int]:
    population = list(range(start_index, total_available))
    if requested <= 0:
        return []
    requested = min(requested, len(population))
    if sample_mode == "sequential" or sample_seed is None:
        return population[:requested]
    rng = random.Random(int(sample_seed))
    if sample_mode == "seeded_sorted":
        return sorted(rng.sample(population, requested))
    return rng.sample(population, requested)


def _get_items_by_indices(loader, indices: list[int]):
    items = []
    for idx in indices:
        batch = loader.get_batch(batch_size=1, offset=idx)
        if batch:
            items.append(batch[0])
    return items


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _drop_error_records_from_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    kept: list[str] = []
    dropped = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                kept.append(line.rstrip("\n"))
                continue
            if isinstance(row, dict) and "error_record" in row:
                dropped += 1
                continue
            kept.append(json.dumps(row, ensure_ascii=False))
    if dropped:
        tmp = path.with_suffix(path.suffix + ".resume_clean_tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            for row_line in kept:
                handle.write(row_line + "\n")
        tmp.replace(path)
    return dropped


def _successful_dataset_indices_from_jsonl(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict) or "error_record" in row:
                continue
            value = row.get("dataset_index")
            try:
                completed.add(int(value))
            except (TypeError, ValueError):
                continue
    return completed


def _short_text(value, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    return text[:limit]


def _safe_tag(value: str | None) -> str:
    text = str(value or "none").strip().lower()
    text = text.replace("/", "_").replace("\\", "_").replace(":", "_")
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "none"


def _model_pair_tag(args) -> str:
    agent_name = args.vllm_model_name if getattr(args, "backend", None) == "vllm" and args.vllm_model_name else args.model_name
    judge_name = (
        (args.judge_vllm_model_name or args.judge_model_name)
        if args.run_judge and args.separate_judge_model and getattr(args, "judge_backend", None) == "vllm"
        else (
            args.judge_model_name
            if args.run_judge and args.separate_judge_model
            else (agent_name if args.run_judge else "nojudge")
        )
    )
    return f"agent-{_safe_tag(agent_name)}__judge-{_safe_tag(judge_name)}"


def _configured_output_dir(args) -> Path:
    base_output_dir = Path(args.output_dir)
    model_tag = _model_pair_tag(args)
    if base_output_dir.name == model_tag:
        return base_output_dir
    return base_output_dir / model_tag


def _compact_obj(value, max_str: int = 240, max_list: int = 3, max_dict: int = 8):
    if isinstance(value, str):
        return value[:max_str]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_compact_obj(v, max_str=max_str, max_list=max_list, max_dict=max_dict) for v in value[:max_list]]
    if isinstance(value, dict):
        out = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_dict:
                break
            out[str(key)] = _compact_obj(item, max_str=max_str, max_list=max_list, max_dict=max_dict)
        return out
    return str(value)[:max_str]


def _extract_core_evidence(env_message) -> dict:
    if not env_message:
        return {}
    try:
        payload = json.loads(env_message) if isinstance(env_message, str) else env_message
    except Exception:
        return {"message_excerpt": _short_text(env_message, 300)}
    if not isinstance(payload, dict):
        return {"message_excerpt": _short_text(payload, 300)}

    evidence = {}
    for key in ("query", "species", "field", "tool_name", "result_count", "advice_count", "operation"):
        if key in payload:
            evidence[key] = _compact_obj(payload.get(key), max_str=160, max_list=2, max_dict=4)

    rows = payload.get("candidates") or payload.get("results") or payload.get("answer_evidence") or []
    if isinstance(rows, list) and rows:
        top = []
        for row in rows[:3]:
            if not isinstance(row, dict):
                continue
            facts = row.get("facts") or row.get("all_facts") or row.get("matched_facts") or []
            relevance = row.get("relevance") or {}
            fact_bits = []
            if isinstance(facts, list):
                for fact in facts[:2]:
                    if isinstance(fact, dict):
                        fact_bits.append(
                            {
                                "field": fact.get("field"),
                                "answer": _short_text(fact.get("answer"), 140),
                            }
                        )
            top.append(
                {
                    "faq_id": row.get("faq_id"),
                    "species": _short_text(row.get("species"), 120),
                    "issue_type": _short_text(row.get("issue_type"), 120),
                    "score": row.get("fused_score", row.get("image_score", row.get("text_score"))),
                    "relevance": _compact_obj(relevance, max_str=120, max_list=2, max_dict=6) if relevance else {},
                    "facts": fact_bits,
                }
            )
        evidence["top_evidence"] = top

    advices = payload.get("advices") or []
    if isinstance(advices, list) and advices:
        evidence["top_advices"] = [
            {
                "faq_id": a.get("faq_id"),
                "species": _short_text(a.get("species"), 120),
                "advice": _short_text(a.get("advice"), 180),
            }
            for a in advices[:2]
            if isinstance(a, dict)
        ]

    if payload.get("generated_image_path"):
        evidence["visual_artifact"] = {
            "generated_image_path": payload.get("generated_image_path"),
            "source_image_path": payload.get("source_image_path"),
            "operation": payload.get("operation"),
            "bbox_pixels": payload.get("bbox_pixels") or payload.get("crop_bbox_pixels") or payload.get("boxes_pixels"),
        }
    return evidence


def _build_core_trajectory(result: dict, judge_res: dict | None = None) -> dict:
    judge_context = result.get("judge_context") if isinstance(result.get("judge_context"), dict) else {}
    if judge_context.get("compact_trajectory"):
        return _build_core_trajectory_from_judge_context(result, judge_context, judge_res)

    decisions = {str(d.get("step")): d for d in result.get("decision_trace", []) if isinstance(d, dict)}
    judge_steps = {}
    if judge_res:
        for step_score in judge_res.get("step_scores") or []:
            if isinstance(step_score, dict):
                judge_steps[str(step_score.get("step"))] = step_score

    steps = []
    for step in result.get("trajectory", []):
        if not isinstance(step, dict):
            continue
        step_id = step.get("step")
        decision = decisions.get(str(step_id), {})
        observation = step.get("observation") or {}
        params = step.get("params") or {}
        judge_step = judge_steps.get(str(step_id), {})
        steps.append(
            {
                "step": step_id,
                "action": step.get("action"),
                "params": _compact_obj(params, max_str=220, max_list=4, max_dict=8),
                "agent_reason": {
                    key: _compact_obj((decision.get("decision_metadata") or {}).get(key), max_str=260, max_list=2, max_dict=4)
                    for key in ("working_hypothesis", "evidence_gap", "why_this_tool", "expected_contribution")
                    if (decision.get("decision_metadata") or {}).get(key) not in (None, "", [], {})
                },
                "accepted_evidence": _compact_obj((decision.get("decision_metadata") or {}).get("accepted_evidence"), max_str=220, max_list=3, max_dict=4),
                "rejected_evidence": _compact_obj((decision.get("decision_metadata") or {}).get("rejected_evidence"), max_str=220, max_list=3, max_dict=4),
                "tool_evidence_ledger": {
                    "accepted": _compact_obj(decision.get("agent_accepted_evidence"), max_str=220, max_list=4, max_dict=5),
                    "rejected": _compact_obj(decision.get("agent_rejected_evidence"), max_str=220, max_list=4, max_dict=5),
                },
                "tool_feedback": _extract_core_evidence(observation.get("env_message", "")),
                "visual_artifact": decision.get("visual_artifact") or {},
                "judge": {
                    "score": judge_step.get("score"),
                    "reason": judge_step.get("reason"),
                    "positive_evidence": judge_step.get("positive_evidence", []),
                    "issues": judge_step.get("issues", []),
                } if judge_step else None,
            }
        )

    return {
        "task_id": result.get("task_id"),
        "dataset_index": result.get("dataset_index"),
        "query": result.get("query"),
        "gt_answer": result.get("gt_answer"),
        "final_answer": result.get("final_answer"),
        "image_paths": result.get("image_paths"),
        "metadata": _compact_obj(result.get("metadata"), max_str=180, max_list=4, max_dict=8),
        "visual_artifacts": result.get("visual_artifacts"),
        "judge_summary": {
            "total_score": judge_res.get("total_score"),
            "answer_score": judge_res.get("answer_score"),
            "process_score": judge_res.get("process_score"),
            "verdict": judge_res.get("verdict"),
            "reason": judge_res.get("reason"),
            "flags": judge_res.get("flags"),
            "subscores": judge_res.get("subscores"),
            "step_score_summary": judge_res.get("step_score_summary"),
        } if judge_res else None,
        "steps": steps,
    }


def _build_core_trajectory_from_judge_context(
    result: dict,
    judge_context: dict,
    judge_res: dict | None = None,
) -> dict:
    decisions = {
        str(d.get("step")): d
        for d in judge_context.get("agent_decision_metadata", [])
        if isinstance(d, dict)
    }
    judge_steps = {}
    if judge_res:
        for step_score in judge_res.get("step_scores") or []:
            if isinstance(step_score, dict):
                judge_steps[str(step_score.get("step"))] = step_score

    steps = []
    for step in judge_context.get("compact_trajectory", []):
        if not isinstance(step, dict):
            continue
        step_id = step.get("step")
        decision = decisions.get(str(step_id), {})
        metadata = decision.get("decision_metadata") or {}
        judge_step = judge_steps.get(str(step_id), {})
        steps.append(
            {
                "step": step_id,
                "action": step.get("action"),
                "params": step.get("params") or {},
                "agent_reason": {
                    key: _compact_obj(metadata.get(key), max_str=260, max_list=2, max_dict=4)
                    for key in ("working_hypothesis", "evidence_gap", "why_this_tool", "expected_contribution")
                    if metadata.get(key) not in (None, "", [], {})
                },
                "accepted_evidence": metadata.get("accepted_evidence", []),
                "rejected_evidence": metadata.get("rejected_evidence", []),
                "tool_evidence_ledger": decision.get("tool_evidence_ledger") or {},
                "tool_feedback": step.get("evidence") or {},
                "visual_artifact": decision.get("visual_artifact") or {},
                "judge": {
                    "score": judge_step.get("score"),
                    "reason": judge_step.get("reason"),
                    "positive_evidence": judge_step.get("positive_evidence", []),
                    "issues": judge_step.get("issues", []),
                } if judge_step else None,
            }
        )

    task_context = judge_context.get("task") or {}
    final_context = judge_context.get("final") or {}
    return {
        "task_id": result.get("task_id") or task_context.get("task_id"),
        "dataset_index": result.get("dataset_index"),
        "query": result.get("query") or task_context.get("query"),
        "gt_answer": result.get("gt_answer") or task_context.get("gt_answer"),
        "final_answer": result.get("final_answer") or final_context.get("pred_answer"),
        "image_paths": result.get("image_paths") or task_context.get("image_paths"),
        "metadata": _compact_obj(result.get("metadata") or task_context.get("metadata"), max_str=180, max_list=4, max_dict=8),
        "visual_artifacts": result.get("visual_artifacts") or judge_context.get("visual_artifacts"),
        "judge_context_version": judge_context.get("version"),
        "final_evidence_used": final_context.get("evidence_used"),
        "final_cited_evidence_facts": judge_context.get("final_cited_evidence_facts"),
        "judge_summary": {
            "total_score": judge_res.get("total_score"),
            "answer_score": judge_res.get("answer_score"),
            "process_score": judge_res.get("process_score"),
            "verdict": judge_res.get("verdict"),
            "reason": judge_res.get("reason"),
            "flags": judge_res.get("flags"),
            "subscores": judge_res.get("subscores"),
            "step_score_summary": judge_res.get("step_score_summary"),
        } if judge_res else None,
        "steps": steps,
    }


def _result_for_storage(result: dict) -> dict:
    judge_context = result.get("judge_context") if isinstance(result.get("judge_context"), dict) else {}
    if not judge_context:
        return result
    return {
        "task_id": result.get("task_id"),
        "query": result.get("query"),
        "gt_answer": result.get("gt_answer"),
        "final_answer": result.get("final_answer"),
        "reward": result.get("reward"),
        "success": result.get("success"),
        "exact_match": result.get("exact_match"),
        "token_f1": result.get("token_f1"),
        "steps": result.get("steps"),
        "trajectory": judge_context.get("compact_trajectory", []),
        "decision_trace": judge_context.get("agent_decision_metadata", []),
        "judge_context": judge_context,
        "image_path": result.get("image_path"),
        "image_paths": result.get("image_paths"),
        "metadata": result.get("metadata"),
        "visual_artifacts": result.get("visual_artifacts") or judge_context.get("visual_artifacts"),
        "timestamp": result.get("timestamp"),
        "dataset_index": result.get("dataset_index"),
        "batch_start": result.get("batch_start"),
        "raw_trajectory_omitted": True,
        "raw_env_message_omitted": True,
    }


def _judge_one(judge: LLMJudge, result: dict, tool_names: list[str], args) -> dict:
    judge_res = judge.judge(result, tool_names)
    judge_score = float(judge_res.get("total_score", 0.0))
    judge_res["pass_threshold"] = args.judge_pass_score
    judge_res["verdict"] = "pass" if judge_score >= args.judge_pass_score else "fail"
    judge_res["verdict_source"] = "threshold"
    return {
        "task_id": result.get("task_id"),
        "query": result.get("query"),
        "gt_answer": result.get("gt_answer"),
        "final_answer": result.get("final_answer"),
        "image_path": result.get("image_path"),
        "image_paths": result.get("image_paths"),
        "metadata": _compact_obj(result.get("metadata"), max_str=180, max_list=4, max_dict=8),
        "visual_artifacts": result.get("visual_artifacts"),
        "agent_steps": result.get("steps"),
        "agent_actions": [s.get("action") for s in result.get("trajectory", [])],
        "core_trajectory": _build_core_trajectory(result, judge_res),
        "judge": judge_res,
    }


def _init_tool_usage(tool_names: list[str]) -> dict:
    action_counts = {name: 0 for name in tool_names}
    action_counts["submit_answer"] = 0
    action_counts["request_info"] = 0
    return {
        "tasks_with_any_tool": 0,
        "action_counts": action_counts,
    }


def _update_tool_usage(accum: dict, result: dict, tool_names: list[str]) -> None:
    actions = [step.get("action") for step in result.get("trajectory", [])]
    has_tool = False
    for action in actions:
        if action in accum["action_counts"]:
            accum["action_counts"][action] += 1
        if action in tool_names:
            has_tool = True
    if has_tool:
        accum["tasks_with_any_tool"] += 1


def _finalize_tool_usage(accum: dict, total: int) -> dict:
    return {
        "tasks_with_any_tool": accum["tasks_with_any_tool"],
        "tool_call_task_ratio": accum["tasks_with_any_tool"] / max(1, total),
        "action_counts": accum["action_counts"],
    }


def _init_benchmark_metrics() -> dict:
    return {
        "completed_with_submit": 0,
        "total_steps": 0,
        "step_count_histogram": {},
        "terminal_action_counts": {},
        "tool_call_count_histogram": {},
        "total_visual_artifacts": 0,
        "tool_errors": 0,
        "total_tool_calls": 0,
        "visual_artifact_tasks": 0,
        "visual_artifact_integrated_tasks": 0,
        "grounded_answer_tasks": 0,
        "answer_score_sum": 0.0,
        "process_score_sum": 0.0,
        "answer_score_count": 0,
        "process_score_count": 0,
        "subscore_sums": {
            "answer_correctness": 0.0,
            "evidence_grounding": 0.0,
            "trajectory_quality": 0.0,
            "tool_efficiency": 0.0,
            "visual_reasoning": 0.0,
        },
        "subscore_count": 0,
        "task_id_counts": {},
        "content_key_counts": {},
    }


def _update_rollout_metrics(accum: dict, result: dict, tool_names: list[str]) -> None:
    trajectory = result.get("trajectory") or []
    decisions = {
        str(item.get("step")): item
        for item in (result.get("decision_trace") or [])
        if isinstance(item, dict)
    }
    actions = [step.get("action") for step in trajectory if isinstance(step, dict)]
    final_answer = str(result.get("final_answer") or "").strip()
    if final_answer and actions and actions[-1] == "submit_answer":
        accum["completed_with_submit"] += 1
    step_count = len(trajectory)
    accum["total_steps"] += step_count
    step_key = str(step_count)
    accum["step_count_histogram"][step_key] = accum["step_count_histogram"].get(step_key, 0) + 1
    terminal_action = str(actions[-1] if actions else "none")
    accum["terminal_action_counts"][terminal_action] = accum["terminal_action_counts"].get(terminal_action, 0) + 1

    tool_set = set(tool_names)
    tool_call_count = sum(1 for action in actions if action in tool_set)
    tool_call_key = str(tool_call_count)
    accum["tool_call_count_histogram"][tool_call_key] = accum["tool_call_count_histogram"].get(tool_call_key, 0) + 1
    accum["total_visual_artifacts"] += len(result.get("visual_artifacts") or [])

    for step in trajectory:
        if not isinstance(step, dict):
            continue
        if step.get("action") in tool_set:
            accum["total_tool_calls"] += 1
            observation = step.get("observation") or {}
            try:
                payload = json.loads(observation.get("env_message") or "{}")
            except Exception:
                payload = {}
            decision = decisions.get(str(step.get("step"))) or {}
            compact_error = bool(decision.get("tool_error")) or decision.get("tool_success") is False
            raw_error = isinstance(payload, dict) and (
                payload.get("error") or payload.get("retrieval_diagnostics", {}).get("status") == "error"
            )
            if raw_error or compact_error:
                accum["tool_errors"] += 1

    if result.get("visual_artifacts"):
        accum["visual_artifact_tasks"] += 1
        if _visual_artifact_integrated(result):
            accum["visual_artifact_integrated_tasks"] += 1

    task_id = str(result.get("task_id") or "")
    if task_id:
        accum["task_id_counts"][task_id] = accum["task_id_counts"].get(task_id, 0) + 1
    content_key = json.dumps(
        {
            "task_id": result.get("task_id"),
            "query": result.get("query"),
            "image_paths": result.get("image_paths") or ([result.get("image_path")] if result.get("image_path") else []),
            "gt_answer": result.get("gt_answer"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    accum["content_key_counts"][content_key] = accum["content_key_counts"].get(content_key, 0) + 1


def _visual_artifact_integrated(result: dict) -> bool:
    artifacts = result.get("visual_artifacts") or []
    paths = [
        str(item.get("generated_image_path") or "")
        for item in artifacts
        if isinstance(item, dict) and item.get("generated_image_path")
    ]
    if not paths:
        return False
    trajectory = result.get("trajectory") or []
    decision_trace = result.get("decision_trace") or []
    for path in paths:
        basename = Path(path).name
        for step in trajectory:
            if basename and basename in json.dumps(step.get("params") or {}, ensure_ascii=False):
                return True
        for decision in decision_trace:
            if basename and basename in json.dumps(decision, ensure_ascii=False):
                return True
    return False


def _update_judge_metrics(accum: dict, judge_res: dict) -> None:
    subscores = judge_res.get("subscores") or {}
    if isinstance(subscores, dict):
        for key in accum["subscore_sums"]:
            try:
                accum["subscore_sums"][key] += float(subscores.get(key, 0.0))
            except Exception:
                pass
        accum["subscore_count"] += 1
        try:
            if float(subscores.get("evidence_grounding", 0.0)) >= 1.2:
                accum["grounded_answer_tasks"] += 1
        except Exception:
            pass
    try:
        accum["answer_score_sum"] += float(judge_res.get("answer_score", (subscores or {}).get("answer_correctness", 0.0)))
        accum["answer_score_count"] += 1
    except Exception:
        pass
    try:
        process_score = judge_res.get("process_score")
        if process_score is None and isinstance(subscores, dict):
            process_score = (
                float(subscores.get("evidence_grounding", 0.0))
                + float(subscores.get("trajectory_quality", 0.0))
                + float(subscores.get("tool_efficiency", 0.0))
                + float(subscores.get("visual_reasoning", 0.0))
            )
        accum["process_score_sum"] += float(process_score or 0.0)
        accum["process_score_count"] += 1
    except Exception:
        pass


def _judge_score_breakdown(accum: dict) -> dict:
    subscore_count = max(1, accum.get("subscore_count", 0))
    means = {
        key: value / subscore_count
        for key, value in (accum.get("subscore_sums") or {}).items()
    }
    answer_correctness = means.get(
        "answer_correctness",
        accum.get("answer_score_sum", 0.0) / max(1, accum.get("answer_score_count", 0)),
    )
    process_components = {
        "evidence_grounding": means.get("evidence_grounding", 0.0),
        "trajectory_quality": means.get("trajectory_quality", 0.0),
        "tool_efficiency": means.get("tool_efficiency", 0.0),
        "visual_reasoning": means.get("visual_reasoning", 0.0),
    }
    process_total = sum(process_components.values())
    return {
        "score_source": "code_from_judge_subscores",
        "answer_correctness": answer_correctness,
        "answer_score": answer_correctness,
        "process_components": process_components,
        "process_score": process_total,
        "total_score": answer_correctness + process_total,
        "ranges": {
            "answer_correctness": "0-4",
            "evidence_grounding": "0-2",
            "trajectory_quality": "0-2",
            "tool_efficiency": "0-1",
            "visual_reasoning": "0-1",
            "process_score": "0-6",
            "total_score": "0-10",
        },
    }


def _finalize_benchmark_metrics(accum: dict, total: int, judge_scores: list[float], pass_threshold: float) -> dict:
    judged = len(judge_scores)
    subscore_count = max(1, accum["subscore_count"])
    duplicate_task_ids = {
        key: count
        for key, count in accum.get("task_id_counts", {}).items()
        if count > 1
    }
    duplicate_content_count = sum(
        count - 1
        for count in accum.get("content_key_counts", {}).values()
        if count > 1
    )
    return {
        "pass_at_1": sum(1 for score in judge_scores if score >= pass_threshold) / max(1, judged),
        "avg_answer_score": accum["answer_score_sum"] / max(1, accum["answer_score_count"]),
        "avg_process_score": accum["process_score_sum"] / max(1, accum["process_score_count"]),
        "avg_answer_score_normalized": (accum["answer_score_sum"] / max(1, accum["answer_score_count"])) / 4.0,
        "avg_process_score_normalized": (accum["process_score_sum"] / max(1, accum["process_score_count"])) / 6.0,
        "completion_rate": accum["completed_with_submit"] / max(1, total),
        "avg_steps": accum["total_steps"] / max(1, total),
        "tool_error_rate": accum["tool_errors"] / max(1, accum["total_tool_calls"]),
        "visual_artifact_integration_rate": (
            accum["visual_artifact_integrated_tasks"] / max(1, accum["visual_artifact_tasks"])
        ),
        "grounded_answer_rate": accum["grounded_answer_tasks"] / max(1, judged),
        "unique_task_id_count": len(accum.get("task_id_counts", {})),
        "duplicate_task_id_count": sum(count - 1 for count in duplicate_task_ids.values()),
        "duplicate_task_id_rate": sum(count - 1 for count in duplicate_task_ids.values()) / max(1, total),
        "duplicate_content_count": duplicate_content_count,
        "duplicate_content_rate": duplicate_content_count / max(1, total),
        "duplicate_task_ids": dict(sorted(duplicate_task_ids.items())[:20]),
        "subscore_means": {
            key: value / subscore_count
            for key, value in accum["subscore_sums"].items()
        },
        "step_stats": {
            "total_steps": accum["total_steps"],
            "avg_steps": accum["total_steps"] / max(1, total),
            "step_count_histogram": dict(sorted(accum.get("step_count_histogram", {}).items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 999999)),
            "terminal_action_counts": dict(sorted(accum.get("terminal_action_counts", {}).items())),
            "total_tool_calls": accum["total_tool_calls"],
            "avg_tool_calls_per_task": accum["total_tool_calls"] / max(1, total),
            "tool_call_count_histogram": dict(sorted(accum.get("tool_call_count_histogram", {}).items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 999999)),
            "total_visual_artifacts": accum.get("total_visual_artifacts", 0),
            "avg_visual_artifacts_per_task": accum.get("total_visual_artifacts", 0) / max(1, total),
        },
        "definitions": {
            "pass_at_1": "code-computed total_score >= judge_pass_score",
            "avg_answer_score": "mean code-computed answer_score, equal to answer_correctness, range 0-4",
            "avg_process_score": "mean code-computed process_score from process subscores, range 0-6",
            "completion_rate": "non-empty final_answer submitted by terminal submit_answer action",
            "avg_steps": "mean trajectory length",
            "tool_error_rate": "tool calls with explicit tool error divided by total tool calls",
            "visual_artifact_integration_rate": "tasks with visual artifacts where artifact path is reused in later params/decision trace",
            "grounded_answer_rate": "judge evidence_grounding subscore >= 1.2",
            "duplicate_task_id_rate": "extra repeated task_id records divided by evaluated tasks",
            "duplicate_content_rate": "extra exact repeated task_id/query/image_paths/gt records divided by evaluated tasks",
            "subscore_means": "mean judge subscores over judged tasks",
        },
    }


def _drop_metric_definitions(metrics: dict | None) -> dict:
    if not isinstance(metrics, dict):
        return {}
    return {key: value for key, value in metrics.items() if key != "definitions"}


def _flatten_numeric_metrics(prefix: str, value, output: dict) -> None:
    if isinstance(value, bool):
        output[prefix] = int(value)
    elif isinstance(value, (int, float)):
        output[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == "definitions":
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric_metrics(child_prefix, item, output)


def _build_short_summary(summary: dict) -> dict:
    """Build a compact metric-only summary for tables and quick analysis."""
    evaluation = summary.get("evaluation") or {}
    benchmark_metrics = _drop_metric_definitions(summary.get("benchmark_metrics"))
    tool_usage = summary.get("tool_usage") or {}
    agent_rollout = summary.get("agent_rollout") or {}
    batch_api = summary.get("batch_api") or {}
    dataset = summary.get("dataset") or {}
    files = summary.get("files") or {}
    models = summary.get("models") or {}
    failures = summary.get("failures") or []
    if not isinstance(failures, list):
        failures = []

    total_items = (
        agent_rollout.get("total")
        or batch_api.get("selected_count")
        or dataset.get("requested")
        or evaluation.get("total_judged")
        or 0
    )
    judged = evaluation.get("total_judged", 0)
    flat_metrics = {}
    _flatten_numeric_metrics("evaluation", evaluation, flat_metrics)
    _flatten_numeric_metrics("benchmark", benchmark_metrics, flat_metrics)
    _flatten_numeric_metrics("tool_usage", tool_usage, flat_metrics)
    _flatten_numeric_metrics("agent_rollout", agent_rollout, flat_metrics)

    return {
        "schema_version": "agri_judge_short_summary_v1",
        "models": {
            "agent_model": models.get("agent_model"),
            "judge_model": models.get("judge_model") or evaluation.get("model_name"),
            "model_output_tag": models.get("model_output_tag"),
            "model_output_dir": models.get("model_output_dir"),
        },
        "dataset": {
            "name": dataset.get("name"),
            "subset": dataset.get("subset"),
            "split": dataset.get("split"),
            "sample_mode": dataset.get("sample_mode"),
            "sample_seed": dataset.get("sample_seed"),
            "start_index": dataset.get("start_index"),
            "requested": dataset.get("requested") or batch_api.get("selected_count"),
            "total_available": dataset.get("total_available"),
        },
        "counts": {
            "total_items": total_items,
            "judged": judged,
            "failures": len(failures),
            "pass_threshold": evaluation.get("pass_threshold"),
        },
        "metrics": {
            "evaluation": evaluation,
            "benchmark": benchmark_metrics,
            "subscore_means": benchmark_metrics.get("subscore_means") or {},
            "score_breakdown": evaluation.get("score_breakdown") or {},
        },
        "steps": {
            "step_stats": benchmark_metrics.get("step_stats") or {},
            "action_counts": tool_usage.get("action_counts") or {},
            "tasks_with_any_tool": tool_usage.get("tasks_with_any_tool"),
            "tool_call_task_ratio": tool_usage.get("tool_call_task_ratio"),
        },
        "diagnostics": {
            "judge_context": summary.get("judge_context_diagnostics") or {},
            "batch_api": batch_api if batch_api else None,
            "failure_examples": failures[:5],
        },
        "files": {
            "judge_jsonl": files.get("judge_jsonl"),
            "core_trajectory_jsonl": files.get("core_trajectory_jsonl"),
            "summary_file": files.get("summary_file"),
            "short_summary_file": files.get("short_summary_file"),
        },
        "flat_metrics": flat_metrics,
    }


def _run_one_task(task: AgriTask, dataset_index: int, batch_start: int, vlm, tool_registry, args) -> dict:
    env = AgriEnv(tasks=[task], tool_registry=tool_registry, max_steps=args.max_steps)
    agent = LLMAgent(
        vlm=vlm,
        agent_id=f"agri_agent_{args.agent_id}_{dataset_index}",
        max_steps=args.max_steps,
        max_prompt_images=args.max_prompt_images,
        max_history_messages=args.max_history_messages,
    )
    runner = AgriRolloutRunner(env=env, agent=agent, max_steps=args.max_steps)
    result = runner.run_task(0).to_dict()
    result["dataset_index"] = dataset_index
    result["batch_start"] = batch_start
    return result


def run_evaluation(vlm, loader, tool_registry, args):
    """运行评估（Env-Action Loop）。"""
    print("\n" + "="*60)
    print("Starting Batch Env-Agent Evaluation")
    print("="*60)

    base_output_dir = Path(args.output_dir)
    model_tag = _model_pair_tag(args)
    output_dir = _configured_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_output_dir = Path(args.visual_output_dir) if args.visual_output_dir else output_dir / "visual_tools"
    if not visual_output_dir.is_absolute():
        visual_output_dir = Path.cwd() / visual_output_dir
    visual_output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AGRI_VISUAL_TOOL_DIR"] = str(visual_output_dir)
    sample_tag = f"seed{args.sample_seed}_{args.sample_mode}" if args.sample_mode != "sequential" and args.sample_seed is not None else f"start{args.start_index}"
    tag = f"{args.mmst_subset}_{args.split}_{sample_tag}_{model_tag}".lower()
    result_jsonl = output_dir / f"agent_results_{tag}.jsonl"
    trajectory_jsonl = output_dir / f"agent_trajectories_{tag}.jsonl"
    judge_jsonl = output_dir / f"judge_eval_{tag}.jsonl"
    core_trajectory_jsonl = output_dir / f"core_trajectories_{tag}.jsonl"
    summary_file = output_dir / f"batch_workflow_summary_{tag}.json"
    short_summary_file = output_dir / f"short_summary_{tag}.json"

    if args.resume_skip_completed and not args.append_output:
        raise ValueError("--resume-skip-completed requires --append-output so existing successful rows are preserved.")

    resume_completed_indices: set[int] = set()
    resume_cleaned_errors = 0
    if args.resume_skip_completed:
        resume_cleaned_errors = _drop_error_records_from_jsonl(result_jsonl)
        resume_completed_indices = _successful_dataset_indices_from_jsonl(result_jsonl)
        print(
            "Resume mode: cleaned "
            f"{resume_cleaned_errors} error_record row(s); "
            f"will skip {len(resume_completed_indices)} completed dataset_index value(s)."
        )

    for path in (result_jsonl, trajectory_jsonl, judge_jsonl, core_trajectory_jsonl, summary_file, short_summary_file):
        if path.exists() and not args.append_output:
            path.unlink()

    total_available = _dataset_size(loader)
    if args.start_index >= total_available:
        raise ValueError(f"--start-index {args.start_index} is outside dataset size {total_available}")
    total_requested = args.limit if args.limit is not None else total_available - args.start_index
    total_requested = max(0, min(total_requested, total_available - args.start_index))
    selected_indices = _select_dataset_indices(
        total_available=total_available,
        start_index=args.start_index,
        requested=total_requested,
        sample_seed=args.sample_seed,
        sample_mode=args.sample_mode,
    )
    total_requested = len(selected_indices)
    batch_size = args.batch_size or total_requested
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0 when provided")
    concurrency = args.concurrency or batch_size
    if concurrency <= 0:
        raise ValueError("--concurrency must be > 0 when provided")
    judge_concurrency = args.judge_concurrency or (1 if args.serialize_judge_vlm else concurrency)
    if judge_concurrency <= 0:
        raise ValueError("--judge-concurrency must be > 0 when provided")
    num_batches = (total_requested + batch_size - 1) // batch_size
    print(
        f"Batch inference config: total_available={total_available}, "
        f"start_index={args.start_index}, requested={total_requested}, "
        f"sample_mode={args.sample_mode}, sample_seed={args.sample_seed}, "
        f"batch_size={batch_size}, num_batches={num_batches}, concurrency={concurrency}, "
        f"judge_concurrency={judge_concurrency}, serialize_vlm={args.serialize_vlm}, "
        f"serialize_judge_vlm={args.serialize_judge_vlm}, serialize_tools={args.serialize_tools}"
    )
    print(
        "Async scheduler: ThreadPoolExecutor advances multiple independent task trajectories "
        "within each batch; vLLM/API backends run without VLM serialization by default."
    )

    tool_names = list(tool_registry.tools.keys())
    worker_vlm = LockedVLM(vlm) if args.serialize_vlm else vlm
    worker_tool_registry = LockedToolRegistry(tool_registry) if args.serialize_tools else tool_registry
    judge = None
    judge_model_name = None
    if args.run_judge:
        judge_vlm = vlm
        if args.separate_judge_model:
            print(f"Creating JUDGE model ({args.judge_backend}): {args.judge_model_name}")
            judge_local_files_only = decide_local_files_only(
                backend=args.judge_backend,
                model_name=args.judge_model_name,
                model_path=args.judge_model_path,
                force_local_files_only=args.local_files_only,
            )
            if args.judge_backend in {"local", "vllm"}:
                print(f"Judge loading mode: {'cache-only' if judge_local_files_only else 'cache+download'}")
            else:
                print("Judge loading mode: remote API")
            judge_api_key = args.judge_api_key
            if not judge_api_key and args.judge_backend == "vllm":
                judge_api_key = args.api_key
            judge_vlm = create_vlm(
                backend=args.judge_backend,
                model_type=args.judge_model_type,
                model_name=args.judge_model_name,
                api_key=judge_api_key,
                api_base=args.judge_api_base,
                model_path=args.judge_model_path,
                device=args.device,
                dtype=args.dtype,
                local_files_only=judge_local_files_only,
            )
        if args.serialize_judge_vlm:
            judge_vlm = worker_vlm if judge_vlm is vlm else LockedVLM(judge_vlm)
        judge = LLMJudge(judge_vlm, judge_mode=args.judge_mode)
        judge_model_name = args.judge_model_name if args.separate_judge_model else args.model_name

    completed_count = 0
    reward_sum = 0.0
    success_count = 0
    judge_scores = []
    judge_answer_scores = []
    judge_process_scores = []
    failures = []
    tool_usage_accum = _init_tool_usage(tool_names)
    benchmark_metric_accum = _init_benchmark_metrics()

    batch_positions = range(0, total_requested, batch_size)
    batch_bar = tqdm(
        batch_positions,
        total=num_batches,
        desc="MMST batch chunks",
        unit="batch",
        dynamic_ncols=True,
    )
    for batch_pos in batch_bar:
        original_batch_indices = selected_indices[batch_pos:batch_pos + batch_size]
        if resume_completed_indices:
            batch_indices = [idx for idx in original_batch_indices if idx not in resume_completed_indices]
        else:
            batch_indices = original_batch_indices
        batch_count = len(batch_indices)
        batch_label = f"{original_batch_indices[0]}-{original_batch_indices[-1]}" if original_batch_indices else str(batch_pos)
        batch_bar.set_postfix(
            completed=completed_count,
            failures=len(failures),
            judge_avg=(sum(judge_scores) / len(judge_scores)) if judge_scores else None,
        )
        clear_in_memory_images()
        if resume_completed_indices and not batch_indices:
            batch_bar.set_postfix(
                completed=completed_count,
                failures=len(failures),
                skipped_existing=len(original_batch_indices),
            )
            continue
        items = _get_items_by_indices(loader, batch_indices)
        tasks = _build_tasks_from_items(items, subset=args.mmst_subset)
        task_items = list(zip(batch_indices, tasks))
        judge_futures = {}
        task_bar = tqdm(
            total=len(task_items),
            desc=f"Tasks in batch@{batch_label}",
            unit="task",
            leave=False,
            dynamic_ncols=True,
        )
        judge_executor = (
            ThreadPoolExecutor(max_workers=max(1, min(judge_concurrency, len(task_items))))
            if judge is not None and task_items
            else None
        )
        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(task_items)))) as executor:
            futures = {
                executor.submit(
                    _run_one_task,
                    task,
                    dataset_idx,
                    batch_pos,
                    worker_vlm,
                    worker_tool_registry,
                    args,
                ): (dataset_idx, task)
                for dataset_idx, task in task_items
            }
            for future in as_completed(futures):
                global_idx, task = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    failure = {
                        "dataset_index": global_idx,
                        "task_id": getattr(task, "task_id", None),
                        "batch_start": batch_pos,
                        "error": str(e),
                    }
                    failures.append(failure)
                    _append_jsonl(result_jsonl, {"error_record": failure})
                    task_bar.update(1)
                    task_bar.set_postfix(
                        task_id=str(getattr(task, "task_id", global_idx))[:24],
                        error=str(e)[:30],
                    )
                    if not args.skip_errors:
                        raise
                    continue

                completed_count += 1
                reward_sum += float(result.get("reward", 0.0))
                success_count += 1 if result.get("success") else 0
                _update_tool_usage(tool_usage_accum, result, tool_names)
                _update_rollout_metrics(benchmark_metric_accum, result, tool_names)
                _append_jsonl(result_jsonl, _result_for_storage(result))
                _append_jsonl(
                    trajectory_jsonl,
                    {
                        "task_id": result.get("task_id"),
                        "dataset_index": global_idx,
                        "image_path": result.get("image_path"),
                        "image_paths": result.get("image_paths"),
                        "judge_context": result.get("judge_context"),
                        "trajectory": (result.get("judge_context") or {}).get("compact_trajectory", []),
                        "decision_trace": (result.get("judge_context") or {}).get("agent_decision_metadata", []),
                        "visual_artifacts": result.get("visual_artifacts") or (result.get("judge_context") or {}).get("visual_artifacts"),
                        "raw_trajectory_omitted": True,
                        "raw_env_message_omitted": True,
                    },
                )

                if judge is not None:
                    judge_futures[
                        judge_executor.submit(_judge_one, judge, result, tool_names, args)
                    ] = (global_idx, batch_pos, task)
                    task_bar.set_postfix(
                        task_id=str(getattr(task, "task_id", global_idx))[:24],
                        judge="queued",
                    )
                else:
                    task_bar.set_postfix(task_id=str(getattr(task, "task_id", global_idx))[:24])
                task_bar.update(1)
        task_bar.close()

        if judge_futures:
            judge_bar = tqdm(
                total=len(judge_futures),
                desc=f"Judge in batch@{batch_label}",
                unit="task",
                leave=False,
                dynamic_ncols=True,
            )
            for judge_future in as_completed(judge_futures):
                global_idx, judge_batch_start, task = judge_futures[judge_future]
                try:
                    judged = judge_future.result()
                    judged["dataset_index"] = global_idx
                    judged["batch_start"] = judge_batch_start
                    judge_res = judged.get("judge", {})
                    judge_scores.append(float(judge_res.get("total_score", 0.0)))
                    judge_answer_scores.append(float(judge_res.get("answer_score", 0.0)))
                    judge_process_scores.append(float(judge_res.get("process_score", 0.0)))
                    _update_judge_metrics(benchmark_metric_accum, judge_res)
                    _append_jsonl(judge_jsonl, judged)
                    _append_jsonl(core_trajectory_jsonl, judged.get("core_trajectory", {}))
                    judge_bar.set_postfix(
                        task_id=str(getattr(task, "task_id", global_idx))[:24],
                        judge=round(judge_scores[-1], 3),
                    )
                except Exception as e:
                    failure = {
                        "dataset_index": global_idx,
                        "task_id": getattr(task, "task_id", None),
                        "batch_start": judge_batch_start,
                        "stage": "judge",
                        "error": str(e),
                    }
                    failures.append(failure)
                    _append_jsonl(judge_jsonl, {"error_record": failure})
                    judge_bar.set_postfix(
                        task_id=str(getattr(task, "task_id", global_idx))[:24],
                        error=str(e)[:30],
                    )
                    if not args.skip_errors:
                        if judge_executor is not None:
                            judge_executor.shutdown(wait=False, cancel_futures=True)
                        raise
                finally:
                    judge_bar.update(1)
            judge_bar.close()
        if judge_executor is not None:
            judge_executor.shutdown(wait=True)

        if args.save_interval and completed_count % args.save_interval == 0:
            tqdm.write(f"[Progress] completed={completed_count}/{total_requested}, failures={len(failures)}")

        clear_in_memory_images()
        batch_bar.set_postfix(
            completed=completed_count,
            failures=len(failures),
            judge_avg=round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None,
        )

    stats = {
        "total": completed_count,
        "avg_reward": reward_sum / completed_count if completed_count else 0.0,
        "success_rate": success_count / completed_count if completed_count else 0.0,
        "failures": len(failures),
    }
    judge_summary = None
    if judge_scores:
        judge_summary = {
            "model_name": judge_model_name,
            "avg_score": sum(judge_scores) / max(1, len(judge_scores)),
            "avg_answer_score": sum(judge_answer_scores) / max(1, len(judge_answer_scores)),
            "avg_process_score": sum(judge_process_scores) / max(1, len(judge_process_scores)),
            "avg_answer_score_normalized": (sum(judge_answer_scores) / max(1, len(judge_answer_scores))) / 4.0,
            "avg_process_score_normalized": (sum(judge_process_scores) / max(1, len(judge_process_scores))) / 6.0,
            "pass_at_1": sum(1 for s in judge_scores if s >= args.judge_pass_score) / max(1, len(judge_scores)),
            "pass_rate": sum(1 for s in judge_scores if s >= args.judge_pass_score) / max(1, len(judge_scores)),
            "pass_threshold": args.judge_pass_score,
            "answer_score_range": "0-4",
            "process_score_range": "0-6",
            "score_source": "code_from_judge_subscores",
            "score_breakdown": _judge_score_breakdown(benchmark_metric_accum),
            "total_judged": len(judge_scores),
            "judge_mode": args.judge_mode,
        }

    output_files = {
        "results_jsonl": str(result_jsonl),
        "trajectory_jsonl": str(trajectory_jsonl),
        "core_trajectory_jsonl": str(core_trajectory_jsonl) if args.run_judge else None,
        "judge_jsonl": str(judge_jsonl) if args.run_judge else None,
        "summary_file": str(summary_file),
        "short_summary_file": str(short_summary_file),
        "storage": {
            "results_jsonl": "compact runtime judge_context; raw trajectory/env_message omitted",
            "trajectory_jsonl": "runtime judge_context plus compact trajectory and decision metadata",
            "core_trajectory_jsonl": "judge-readable trajectory with per-step judge scores",
        },
    }
    summary = {
        "dataset": {
            "name": "MIRAGE-Benchmark/MIRAGE",
            "subset": args.mmst_subset,
            "split": args.split,
            "total_available": total_available,
            "start_index": args.start_index,
            "sample_mode": args.sample_mode,
            "sample_seed": args.sample_seed,
            "selected_indices": selected_indices,
            "requested": total_requested,
            "resume_skip_completed": args.resume_skip_completed,
            "resume_cleaned_error_records": resume_cleaned_errors,
            "resume_completed_existing": len(resume_completed_indices),
            "batch_size": batch_size,
            "concurrency": concurrency,
            "judge_concurrency": judge_concurrency if args.run_judge else 0,
            "scheduler": "threadpool_async",
            "serialize_vlm": args.serialize_vlm,
            "serialize_judge_vlm": args.serialize_judge_vlm,
            "serialize_tools": args.serialize_tools,
        },
        "agent_rollout": stats,
        "evaluation": judge_summary,
        "benchmark_metrics": _finalize_benchmark_metrics(
            benchmark_metric_accum,
            completed_count,
            judge_scores,
            args.judge_pass_score,
        ),
        "tool_usage": _finalize_tool_usage(tool_usage_accum, completed_count),
        "failures": failures,
        "files": output_files,
        "models": {
            "agent_model": args.model_name,
            "judge_model": judge_model_name,
            "kb_emb_dir": args.kb_emb_dir,
            "model_output_tag": model_tag,
            "base_output_dir": str(base_output_dir),
            "model_output_dir": str(output_dir),
        },
        "visual_tools": {
            "output_dir": str(visual_output_dir),
        },
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(short_summary_file, "w", encoding="utf-8") as f:
        json.dump(_build_short_summary(summary), f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("Evaluation Complete")
    print("="*60)
    print(f"Total evaluated: {stats['total']}")
    print(f"Avg reward: {stats['avg_reward']:.4f}")
    print(f"Success rate: {stats['success_rate']:.4f}")
    print(f"Agent results JSONL: {result_jsonl}")
    print(f"Agent trajectories JSONL: {trajectory_jsonl}")
    if judge_summary:
        print(f"Core trajectories JSONL: {core_trajectory_jsonl}")
        print(f"Judge avg score: {judge_summary['avg_score']:.3f}")
        print(
            "Judge avg answer/process: "
            f"{judge_summary['avg_answer_score']:.3f}/4, {judge_summary['avg_process_score']:.3f}/6"
        )
        components = (judge_summary.get("score_breakdown") or {}).get("process_components") or {}
        print(
            "Judge avg components: "
            f"answer_correctness={judge_summary['score_breakdown']['answer_correctness']:.3f}/4, "
            f"evidence_grounding={components.get('evidence_grounding', 0.0):.3f}/2, "
            f"trajectory_quality={components.get('trajectory_quality', 0.0):.3f}/2, "
            f"tool_efficiency={components.get('tool_efficiency', 0.0):.3f}/1, "
            f"visual_reasoning={components.get('visual_reasoning', 0.0):.3f}/1"
        )
        print(f"Judge pass rate: {judge_summary['pass_rate']:.3f}")
        print(f"Judge eval JSONL: {judge_jsonl}")
    print(f"Summary saved to: {summary_file}")
    print(f"Short summary saved to: {short_summary_file}")

    return {
        "total_evaluated": stats["total"],
        "avg_reward": stats["avg_reward"],
        "success_rate": stats["success_rate"],
        "results_file": str(result_jsonl),
        "trajectory_file": str(trajectory_jsonl),
        "core_trajectory_file": str(core_trajectory_jsonl) if args.run_judge else None,
        "judge_file": str(judge_jsonl) if args.run_judge else None,
        "summary_file": output_files["summary_file"],
    }


def run_single_query(vlm, tool_registry, args):
    """运行单个查询（Env-Action Loop）。"""
    print("\n" + "="*60)
    print("Running Single Query")
    print("="*60)

    task = AgriTask(
        task_id="query_0",
        query=args.query,
        gt_answer=args.gt_answer or "",
        image_path=args.query_image,
        image_paths=[args.query_image] if args.query_image else [],
        metadata={"source": "interactive_query", "image_paths": [args.query_image] if args.query_image else []},
    )
    env = AgriEnv(tasks=[task], tool_registry=tool_registry, max_steps=args.max_steps)
    agent = LLMAgent(
        vlm=vlm,
        max_steps=args.max_steps,
        max_prompt_images=args.max_prompt_images,
        max_history_messages=args.max_history_messages,
    )
    runner = AgriRolloutRunner(env=env, agent=agent, max_steps=args.max_steps)

    result = runner.run_task(0).to_dict()

    print(f"\nQuery: {result['query']}")
    print(f"\nAnswer: {result['final_answer']}")
    print(f"\nReward: {result['reward']:.4f}")
    print(f"\nSteps taken: {result['steps']}")

    if args.show_trajectory:
        print("\nFull Trajectory:")
        print(json.dumps(result["trajectory"], indent=2, ensure_ascii=False))

    return result


def build_parser() -> argparse.ArgumentParser:
    c = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(
        description="MIRAGE MMST agricultural agent batch runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    agent = parser.add_argument_group("agent model")
    agent.add_argument("--backend", default=c["backend"], choices=["local", "api", "vllm"])
    agent.add_argument("--model-type", default=c["model_type"], choices=["qwen3", "qwen2.5", "gemma", "openai", "gpt4v", "claude", "gemini"])
    agent.add_argument("--model-name", default=c["model_name"])
    agent.add_argument("--model-path", default=c["model_path"])
    agent.add_argument("--api-key", default=c["api_key"])
    agent.add_argument("--api-base", default=c["api_base"])
    agent.add_argument("--api-key-file", default=c["api_key_file"])
    agent.add_argument("--agent-max-tokens", type=int, default=c["agent_max_tokens"], help="Agent generation max_tokens. Defaults to a model-family safe value.")
    agent.add_argument("--device", default=c["device"], choices=["cuda", "cpu"])
    agent.add_argument("--dtype", default=c["dtype"], choices=["float32", "float16", "bfloat16"])
    agent.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=c["local_files_only"])

    judge = parser.add_argument_group("judge model")
    judge.add_argument("--run-judge", action=argparse.BooleanOptionalAction, default=c["run_judge"])
    judge.add_argument("--separate-judge-model", action=argparse.BooleanOptionalAction, default=c["separate_judge_model"])
    judge.add_argument("--judge-backend", default=c["judge_backend"], choices=["local", "api", "vllm"])
    judge.add_argument("--judge-model-type", default=c["judge_model_type"], choices=["qwen3", "qwen2.5", "gemma", "openai", "gpt4v", "claude", "gemini"])
    judge.add_argument("--judge-model-name", default=c["judge_model_name"])
    judge.add_argument("--judge-model-path", default=c["judge_model_path"])
    judge.add_argument("--judge-api-key", default=c["judge_api_key"])
    judge.add_argument("--judge-api-base", default=c["judge_api_base"])
    judge.add_argument("--judge-vllm-host", default=c["judge_vllm_host"])
    judge.add_argument("--judge-vllm-port", type=int, default=c["judge_vllm_port"])
    judge.add_argument("--judge-vllm-model-name", default=c["judge_vllm_model_name"])
    judge.add_argument("--judge-vllm-gpu-memory-utilization", type=float, default=c["judge_vllm_gpu_memory_utilization"])
    judge.add_argument("--judge-vllm-max-model-len", type=int, default=c["judge_vllm_max_model_len"])
    judge.add_argument("--judge-vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=c["judge_vllm_enforce_eager"])
    judge.add_argument("--judge-vllm-enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=c["judge_vllm_enable_chunked_prefill"])
    judge.add_argument("--judge-vllm-enable-prefix-caching", action=argparse.BooleanOptionalAction, default=c["judge_vllm_enable_prefix_caching"])
    judge.add_argument("--judge-vllm-extra-args", default=c["judge_vllm_extra_args"], help="Extra raw arguments appended to judge vLLM serve, e.g. --trust-remote-code")
    judge.add_argument("--judge-concurrency", type=int, default=c["judge_concurrency"])
    judge.add_argument("--judge-pass-score", type=float, default=c["judge_pass_score"])
    judge.add_argument("--judge-mode", default=c["judge_mode"], choices=["score_only", "diagnostic"], help="score_only outputs only numeric subscores/step scores; diagnostic also asks for rationales.")
    judge.add_argument("--serialize-judge-vlm", action=argparse.BooleanOptionalAction, default=c["serialize_judge_vlm"])

    serving = parser.add_argument_group("vllm serving")
    serving.add_argument("--launch-vllm", action=argparse.BooleanOptionalAction, default=c["launch_vllm"])
    serving.add_argument("--vllm-model-name", default=c["vllm_model_name"])
    serving.add_argument("--vllm-host", default=c["vllm_host"])
    serving.add_argument("--vllm-port", type=int, default=c["vllm_port"])
    serving.add_argument("--vllm-dtype", default=c["vllm_dtype"], choices=["auto", "float16", "bfloat16", "float32"])
    serving.add_argument("--vllm-gpu-memory-utilization", type=float, default=c["vllm_gpu_memory_utilization"])
    serving.add_argument("--vllm-max-model-len", type=int, default=c["vllm_max_model_len"])
    serving.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=c["vllm_enforce_eager"])
    serving.add_argument("--vllm-enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=c["vllm_enable_chunked_prefill"])
    serving.add_argument("--vllm-enable-prefix-caching", action=argparse.BooleanOptionalAction, default=c["vllm_enable_prefix_caching"])
    serving.add_argument("--vllm-extra-args", default=c["vllm_extra_args"], help="Extra raw arguments appended to agent vLLM serve, e.g. --trust-remote-code")
    serving.add_argument("--vllm-startup-timeout", type=int, default=c["vllm_startup_timeout"])
    serving.add_argument("--vllm-log-dir", default=c["vllm_log_dir"])

    data = parser.add_argument_group("data and kb")
    data.add_argument("--task", default=c["task"], choices=["eval", "query"])
    data.add_argument("--query", default=c["query"])
    data.add_argument("--query-image", default=c["query_image"])
    data.add_argument("--gt-answer", default=c["gt_answer"])
    data.add_argument("--show-trajectory", action=argparse.BooleanOptionalAction, default=c["show_trajectory"])
    data.add_argument("--split", default=c["split"], choices=["train", "test"])
    data.add_argument("--mmst-subset", default=c["mmst_subset"])
    data.add_argument("--mmst-json", default=c["mmst_json"])
    data.add_argument("--limit", type=int, default=c["limit"])
    data.add_argument("--start-index", type=int, default=c["start_index"])
    data.add_argument("--sample-seed", type=int, default=c["sample_seed"])
    data.add_argument("--sample-mode", default=c["sample_mode"], choices=["seeded", "seeded_sorted", "sequential"])
    data.add_argument("--agmmu-path", default=c["agmmu_path"])
    data.add_argument("--kb-processed-json", default=c["kb_processed_json"])
    data.add_argument("--kb-emb-dir", default=c["kb_emb_dir"])
    data.add_argument("--kb-image-root", default=c["kb_image_root"])

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--batch-size", type=int, default=c["batch_size"])
    runtime.add_argument("--concurrency", type=int, default=c["concurrency"])
    runtime.add_argument("--max-steps", type=int, default=c["max_steps"])
    runtime.add_argument("--max-prompt-images", type=int, default=c["max_prompt_images"])
    runtime.add_argument("--max-history-messages", type=int, default=c["max_history_messages"])
    runtime.add_argument("--serialize-vlm", action=argparse.BooleanOptionalAction, default=c["serialize_vlm"])
    runtime.add_argument("--serialize-tools", action=argparse.BooleanOptionalAction, default=c["serialize_tools"])
    runtime.add_argument("--preload-embedding", action=argparse.BooleanOptionalAction, default=c["preload_embedding"])
    runtime.add_argument("--skip-errors", action=argparse.BooleanOptionalAction, default=c["skip_errors"])
    runtime.add_argument("--resume-skip-completed", action=argparse.BooleanOptionalAction, default=c["resume_skip_completed"], help="Append-only resume: remove error_record rows and skip dataset_index values already completed in agent_results.")
    runtime.add_argument("--save-interval", type=int, default=c["save_interval"])

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", default=c["output_dir"])
    output.add_argument("--visual-output-dir", default=c["visual_output_dir"])
    output.add_argument("--append-output", action=argparse.BooleanOptionalAction, default=c["append_output"])
    output.add_argument("--agent-id", default=c["agent_id"])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    loaded_key_files = load_local_api_keys(args.api_key_file)
    if loaded_key_files:
        print(f"Loaded local API key env file(s): {', '.join(loaded_key_files)}")
    if args.backend == "api" and str(args.api_key or "").strip().upper() == "EMPTY":
        args.api_key = os.environ.get("AGRI_AGENT_API_KEY") or args.api_key
    if args.run_judge and args.judge_backend == "api" and not args.judge_api_key:
        args.judge_api_key = os.environ.get("AGRI_JUDGE_API_KEY")
    
    # 验证参数
    if args.task == "query" and not args.query:
        parser.error("query mode requires --query argument")
    if args.backend == "vllm" and args.api_base is None:
        args.api_base = _vllm_api_base(args.vllm_host, args.vllm_port)
    if args.serialize_vlm is None:
        args.serialize_vlm = args.backend == "local"
    if args.serialize_judge_vlm is None:
        args.serialize_judge_vlm = args.judge_backend == "local"
    if args.run_judge and args.separate_judge_model and args.judge_backend == "vllm" and args.judge_api_base is None:
        args.judge_api_base = _vllm_api_base(args.judge_vllm_host, args.judge_vllm_port)
    try:
        ensure_vllm_servers(args)
        validate_vllm_model_names(args)
        print_effective_config(args)

        # 加载 AGMMU 知识库
        agmmu_loader = load_agmmu_knowledge_base(args.agmmu_path, args.kb_processed_json)
        if args.kb_image_root:
            agmmu_loader.kb_root = Path(args.kb_image_root)
            print(f"Visual retrieval image root override: {agmmu_loader.kb_root}")
        
        # 创建工具注册表
        tool_registry = create_tool_registry(agmmu_loader, emb_dir=args.kb_emb_dir, device=args.device)
        if tool_registry.tools:
            print(f"✓ Tools registered: {', '.join([t.name for t in tool_registry.tools.values()])}")
        else:
            print("⚠ No tools registered (AGMMU not available)")
        if args.preload_embedding:
            preload_embedding_retriever(tool_registry)
        
        # 创建 VLM
        print(f"Creating AGENT model ({args.backend}): {args.model_name}")
        agent_local_files_only = decide_local_files_only(
            backend=args.backend,
            model_name=args.model_name,
            model_path=args.model_path,
            force_local_files_only=args.local_files_only,
        )
        if args.backend == "local":
            print(f"Agent loading mode: {'cache-only' if agent_local_files_only else 'cache+download'}")
        elif args.backend == "vllm":
            print("Agent loading mode: vLLM OpenAI-compatible endpoint")
        else:
            print("Agent loading mode: remote API")
        vlm = create_vlm(
            backend=args.backend,
            model_type=args.model_type,
            model_name=args.model_name,
            api_key=args.api_key,
            api_base=args.api_base,
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            local_files_only=agent_local_files_only,
            max_tokens=args.agent_max_tokens,
        )
        
        # 执行任务
        if args.task == "eval":
            # 加载数据集
            print(f"\nLoading MMST dataset ({args.split})...")
            loader = MMSTDatasetLoader(split=args.split)
            
            if args.mmst_json:
                loader.load_from_json(args.mmst_json)
            else:
                loader.load_from_huggingface(subset=args.mmst_subset)
            
            stats = loader.stats()
            print(f"✓ Dataset loaded: {stats['total_items']} items")
            
            # 运行评估
            run_evaluation(vlm, loader, tool_registry, args)
        else:  # query
            # 运行单个查询
            run_single_query(vlm, tool_registry, args)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
