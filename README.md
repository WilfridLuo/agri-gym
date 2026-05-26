# AgriGym

AgriGym is an executable benchmark framework for evidence-grounded agricultural consultation agents. It evaluates an agent through a complete trajectory: the agent receives an agricultural query with optional images, interacts with an environment through tools, submits a final answer, and is then evaluated by a trajectory-aware judge.

This anonymous source release contains only the main inference-and-evaluation workflow. It intentionally excludes ablation scripts, training code, knowledge-base construction scripts, experiment-specific shell launchers, generated outputs, local data, model checkpoints, and private credentials.

## What Is Included

- `run.py`: repository-root entry point.
- `scripts/run.py`: main batch workflow for agent rollout and judge evaluation.
- `src/agent`: agent prompt construction and tool-call parsing.
- `src/env`: executable environment and task interface.
- `src/tools`: visual tools and knowledge-base retrieval tools.
- `src/evaluator`: rollout runner, judge context construction, and judge scoring.
- `src/benchmarks`: dataset loader for benchmark samples.
- `src/vlm`: local/API vision-language model backends.
- `src/data`: processed knowledge-base loader.

## What Is Not Included

- Training or RL code.
- Ablation variants.
- Knowledge-base construction scripts.
- Experiment-specific shell scripts.
- Generated trajectories, judge outputs, figures, logs, checkpoints, or cache files.
- API keys or local machine paths.

## Setup

Create an environment with the provided dependency files, then configure credentials through environment variables or a local `.env` file. Do not commit `.env`.

Common variables:

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
HF_TOKEN=...
```

If using a local vLLM backend, install and launch through the main workflow options. If using an API backend, set `--backend api` and the corresponding model type/name.

## Required External Assets

The source release does not include benchmark data, processed knowledge-base files, or embedding indexes. Provide them with command-line arguments, for example:

```bash
python run.py \
  --kb-processed-json /path/to/processed_kb.json \
  --kb-emb-dir /path/to/embedding_index \
  --output-dir /path/to/output_dir
```

## Main Workflow

The standard command runs agent rollouts and judge evaluation in one process:

```bash
python run.py
```

Useful options:

```bash
python run.py \
  --backend vllm \
  --model-type qwen \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --judge-backend api \
  --judge-model-type openai \
  --judge-model-name gpt-4o \
  --batch-size 16 \
  --max-steps 5 \
  --limit 128
```

The workflow writes agent results, trajectories, core trajectories, judge records, and a summary JSON into the configured output directory.

## Notes For Anonymous Review

This repository contains method code only. All private outputs, local paths, local credentials, training experiments, and auxiliary ablation scripts have been removed from the anonymous source tree.
