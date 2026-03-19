# wow

Air-combat project refactored into a 4-layer boundary:

- `sim/`: AFSIM adapters, replay/scenario interfaces, sim-time control mapping
- `inference/`: model runtime layer (artifact loading, preprocess/infer/postprocess)
- `ml/`: dataset build, training, evaluation, artifact export
- `shared/`: stable contracts (`shared/schema`), units and coordinate conventions
- `artifacts/`: deployable policy artifacts (`policy.yaml`, model file, stats)
- `tools/`: architecture/contract guard scripts

## Quick Layout

```text
wow/
  sim/
  inference/
    interface/
    backends/
    config/
  ml/
  shared/
    schema/
      VERSION
      episode_schema.py
      action_schema.py
      units.md
      coordinate_frames.md
  artifacts/
    policy_demo/v0.1.1/
  tools/
```

## Sim Side

### 1) Export ACMI to episode contract

```bash
python sim/export_episode.py \
  --acmi "tra_data/104th vs inSky - Rd 1.acmi" \
  --out "datasets/episodes/sample_episode.json" \
  --dt 0.5 \
  --max_entities 4
```

### 2) Run BC socket server (artifact-first)

```bash
python sim/bc_policy_socket_server.py \
  --artifact "artifacts/policy_demo/v0.1.1" \
  --control_indices "0,1,2,3"
```

Legacy compatibility (deprecated, still available):

```bash
python sim/bc_policy_socket_server.py --model "bc_transformer_smooth.pt"
```

## ML Side

### 1) Build training dataset

```bash
python ml/build_training_dataset.py --out datasets/dt2hz_H2s_fighteronly.npz
```

### 2) Train

```bash
python ml/train_bc_transformer_smooth.py \
  --data datasets/dt2hz_H2s_fighteronly.npz \
  --save datasets/bc_transformer_fighteronly.pt
```

### 3) Export deployable artifact

ONNX runtime artifact:

```bash
python ml/export_artifact.py \
  --checkpoint datasets/bc_transformer_fighteronly.pt \
  --backend onnxruntime \
  --out artifacts/policy_fighter/v0.1.1
```

Minimal numpy smoke artifact:

```bash
python ml/export_artifact.py \
  --backend numpy_linear \
  --out artifacts/policy_demo/v0.1.1
```

### 4) Smoke-read episode in ML side

```bash
python ml/episode_smoke_test.py --episode datasets/episodes/sample_episode.json
```

## Data Contract

See:
- `data_contract.md`
- `shared/schema/units.md`
- `shared/schema/coordinate_frames.md`

## Guards (CI/local)

```bash
python tools/check_dependency_direction.py
python tools/check_schema_guard.py
python tools/check_artifact_schema_version.py
```

GitHub Actions runs the same guards in:
- `.github/workflows/architecture_contract_checks.yml`

## Backward-Compatible Entrypoints

Old paths remain callable through wrappers, for example:
- `python acmi.py` -> `sim/acmi.py`
- `python training/build_training_dataset.py` -> `ml/build_training_dataset.py`
- `python tools/acmi2tspi.py` -> `sim/tools/acmi2tspi.py`
