# Data Contract

This project uses a versioned episode/action contract under `shared/schema`.

- Version source of truth: `shared/schema/VERSION`
- Current version: `0.1.1`

## Top-Level Episode Fields

| field | type | unit | description |
|---|---|---|---|
| `schema_version` | `str` | - | Must match `shared/schema/VERSION` |
| `episode_id` | `str` | - | Unique episode identifier |
| `source` | `dict` | - | Source metadata (e.g. ACMI path, reference lat/lon) |
| `time_step_s` | `float` | `s` | Fixed frame/action interval |
| `coordinate_frame` | `str` | - | `ENU` (east, north, up) |
| `entities` | `list[Entity]` | - | Entity registry |
| `frames` | `list[Frame]` | - | Time-ordered state snapshots |
| `actions` | `list[ActionFrame]` | - | Time-ordered action commands |

## Entity

| field | type | unit | description |
|---|---|---|---|
| `entity_id` | `str` | - | Stable id in one episode |
| `name` | `str` | - | Human-readable entity name |
| `type` | `str` | - | Platform type |
| `side` | `str` | - | `blue` / `red` / `unknown` |

## State (`frames[i].states[j]`)

| field | type | unit | description |
|---|---|---|---|
| `entity_id` | `str` | - | Entity id |
| `alive` | `bool` | - | Is platform alive at this frame |
| `x_e_m` | `float` | `m` | East position |
| `y_n_m` | `float` | `m` | North position |
| `z_u_m` | `float` | `m` | Up position |
| `vx_e_mps` | `float` | `m/s` | East velocity |
| `vy_n_mps` | `float` | `m/s` | North velocity |
| `vz_u_mps` | `float` | `m/s` | Up velocity |
| `heading_rad` | `float` | `rad` | Heading: `atan2(vx_east, vy_north)` |

## Action (`actions[i].commands[j]`)

| field | type | unit | description |
|---|---|---|---|
| `entity_id` | `str` | - | Entity id |
| `dpsi_rad` | `float` | `rad` | Relative heading delta command |
| `alt_sp_m` | `float` | `m` | Altitude setpoint |
| `spd_sp_mps` | `float` | `m/s` | Speed setpoint |

## Example

```json
{
  "schema_version": "0.1.1",
  "episode_id": "sample_episode",
  "source": {
    "kind": "acmi",
    "file": "tra_data/example.acmi"
  },
  "time_step_s": 0.5,
  "coordinate_frame": "ENU",
  "entities": [
    {
      "entity_id": "1001",
      "name": "F-16C",
      "type": "Air+FixedWing",
      "side": "blue"
    }
  ],
  "frames": [
    {
      "t_s": 0.0,
      "states": [
        {
          "entity_id": "1001",
          "alive": true,
          "x_e_m": 0.0,
          "y_n_m": 0.0,
          "z_u_m": 5000.0,
          "vx_e_mps": 20.0,
          "vy_n_mps": 220.0,
          "vz_u_mps": 0.0,
          "heading_rad": 0.0906598872
        }
      ]
    }
  ],
  "actions": [
    {
      "t_s": 0.0,
      "commands": [
        {
          "entity_id": "1001",
          "dpsi_rad": 0.0,
          "alt_sp_m": 5000.0,
          "spd_sp_mps": 221.0
        }
      ]
    }
  ]
}
```

## Alignment Rules

- `policy.yaml.schema_version` in `artifacts/**` must equal `shared/schema/VERSION`.
- `sim` and `ml` both consume/produce through this contract; runtime inference goes through `inference/`.
