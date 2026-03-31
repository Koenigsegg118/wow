# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM-powered air combat simulation control system. It integrates two LLM agents (Planner + Executor) via LangGraph to generate real-time control decisions for aircraft in a 2v2 aerial combat scenario running in AFSIM (Warlock simulation engine).

## Running the System

```bash
# Activate conda environment
conda activate wow

# Start the Python socket server (waits for AFSIM to connect on localhost:65432)
python llm-llm_with_connection.py

# Optional: run local LLM inference servers (if not using remote servers)
python llm_server.py --model ./models/Qwen3-4B-Instruct-2507 --port 8000
python llm_server.py --model ./models/gemma-3-1b-it --port 8001
```

AFSIM (Warlock) must be started separately and will connect to the Python server. The Python server must be started first.

## Basic Testing

```bash
python test.py        # Tests socket protocol
python test_server.py # Tests server connection
```

## Environment Setup

```bash
conda env create -f environment.yml
conda activate wow
```

Requires CUDA 12.6+ for local GPU inference.

## Architecture

### Data Flow

```
AFSIM (C++) ──TCP──► StateReceiver.recv_frame()
                          │
                          ├──► send_status_data(last_action)  [immediate, ~0ms]
                          │
                          └──► state_queue ──► llm_worker thread
                                                    │
                                              translate_sim_data_to_llm_context()
                                                    │
                                              LangGraph app.invoke()
                                                    │
                                          ┌─── Planner Node (Qwen 4B) ───┐
                                          │                               │
                                          └─── Executor Node (Gemma 1B) ─┘
                                                    │
                                          apply_llm_decision_to_sim()
                                                    │
                                              last_action updated
```

### Key Design: Dual-Thread Real-Time Loop

`realtime_server.py` uses two threads to satisfy AFSIM's fast response requirement (~100ms) while LLM inference takes ~5s:
- **Main thread**: Receives state frames → immediately returns `last_action` → pushes state to queue
- **LLM worker thread**: Consumes queue → runs inference → updates `last_action` (thread-safe via lock)
- The queue holds only the latest frame (drops outdated states via put_nowait/discard pattern)

### TCP Protocol

**State from AFSIM** (text format):
```
simTime 640 [float0] [float1] ... [float639]
```
640 values = 80 platforms × 8 values each: `[live, lat, lon, alt_m, v_north, v_east, v_down, heading_deg]`

**Actions to AFSIM** (binary format):
```
"STATUS" (6 bytes) + 640 × float32 (little-endian)
```
Only platforms 0–3 (red_1, red_2, blue_1, blue_2) are used. Per-platform action values:
- `[0]` = heading delta / 45° → normalized to [-1, 1]
- `[1]` = altitude delta / 2000m → normalized to [-1, 1]
- `[2]` = speed delta / 150 m/s → normalized to [-1, 1]

### LangGraph Workflow (`llm_with_connection/graph.py`)

`AgentState` TypedDict: `{task, plan, result, dynamic_context, planner_decision}`

1. **Planner node** (Qwen 4B): Receives `dynamic_context` → outputs a Chinese tactical plan as JSON `{action, plan}`
2. **Router node**: Always routes to executor (if `planner_decision == "execute"`)
3. **Executor node** (Gemma 1B): Receives plan + context → outputs JSON control increments:
   ```json
   {"red_1": {"turn_deg": float, "up_m": float, "dspeed_mps": float},
    "red_2": {"turn_deg": float, "up_m": float, "dspeed_mps": float}}
   ```

### Module Responsibilities (`llm_with_connection/`)

| File | Responsibility |
|------|---------------|
| `config.py` | LLM server URLs, model names, socket host/port, reset threshold |
| `clients.py` | Create and validate OpenAI-compatible client connections |
| `graph.py` | Define LangGraph nodes (Planner, Executor) and compile workflow |
| `realtime_server.py` | TCP socket server with dual-thread real-time loop |
| `socket_protocol.py` | `StateReceiver` class, `send_status_data()`, `send_reset_instruction()` |
| `sim_translation.py` | Convert 640-element float array → readable Chinese LLM context string |
| `action_mapping.py` | Convert Executor JSON output → normalized 640-element float action array |

## Configuration

Edit `llm_with_connection/config.py` to change:
- `PLANNER_API_BASE` / `EXECUTOR_API_BASE`: LLM server endpoints (default: `http://10.134.114.3:5000/v1` and `:5001/v1`)
- `PLANNER_MODEL_NAME` / `EXECUTOR_MODEL_NAME`: Model identifiers passed to the API
- `PORT`: AFSIM connection port (default: 65432)
- `RESET_TIME_THRESHOLD`: Auto-reset simulation when `simTime` exceeds this value (seconds)

## Other Scripts

- `LLM-Rules.py`: Legacy rules-based (non-LLM) controller
- `LLM-LLM.py`: Earlier version of the dual-LLM controller (predates the modular `llm_with_connection/` package)
- `acmi.py`: Parse ACMI/Tacview files for post-simulation analysis
- `sft_data.py`, `sft.py`, `preprocess_cpt_data.py`: Training data pipeline for fine-tuning models on combat telemetry
- `decision.py`: Standalone local Transformers inference (no server required)
- `bidir_pipe_win.py`: Windows bidirectional pipe communication utility

## Semantic Data Contract — Schema 变更规则 (强制)

`semantic_data_contract_v1/` 是项目语义标注数据的唯一规范。**任何对字段名、枚举值、粒度的变更都必须按以下顺序执行，不得跳步：**

1. 先改 `semantic_record_schema_v1.json`
2. 再改 `enum_registry_v1.json`
3. 再改对照表 / guideline / prompt / export spec（`semantic_record_template_readable.md`, `annotation_guideline_v1.md`, `acmi_annotation_prompt_v1.txt`, `sft_export_spec_v1.md`, `acmi_auto_label_io_examples.json`）
4. 最后 bump 一个 patch 版本（更新 `V1_1_PATCH_SUMMARY.md` 或创建新的 patch summary）

**禁止：**
- 边做数据边改字段名/枚举/粒度
- 在单个文件中私自新增枚举值而不更新 enum_registry
- 三路数据（manual / acmi_ai / afsim_rule）各自定义不同字段名

违反此规则会导致三路数据不对齐，修复成本极高。
