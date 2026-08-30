# Canonical Tracking Schema

## Schema Version
0.2.0

## Purpose
The canonical tracking schema defines the common internal representation for all tracking data within the Ball-Free Game State Reconstruction project. By standardizing the format, we ensure that downstream modules (such as noise analysis, tactical validation, and coordinate transformation) can seamlessly consume data originating from diverse sources, including Metrica Sports, SkillCorner, SoccerNet-GSR, and synthetically generated tracking data.

## Required Fields

The schema utilizes a "tall" (long) format where each row represents a single entity (player) at a specific point in time. 

| Field | Data Type | Meaning | Allowed/Expected Values | Null Allowed? |
| :--- | :--- | :--- | :--- | :--- |
| `match_id` | String | A unique identifier for the match or tracking sequence. Must be unique across all datasets. | Alpha-numeric strings (e.g., `"metrica_1"`). Passed explicitly during ingestion. | No |
| `frame` | Integer | The discrete temporal index of the observation. | `> 0`. Strictly monotonically increasing sequence. | No |
| `timestamp` | Float | The elapsed time in seconds from a reference point (e.g., start of half or recording). | `>= 0.0`. Represents continuous time. | No |
| `player_id` | String | A unique identifier for the player. | Alpha-numeric strings (e.g., `"11"`, `"home_11"`). Must be stable within a match. | No |
| `team` | String | The team to which the player belongs. | Controlled vocabulary: `"home"`, `"away"`, or `"referee"`. | No |
| `x` | Float | The entity's spatial X coordinate. | Floating-point values. Normalized pitch coordinates are expected (e.g., `[0, 1]`), though out-of-bounds values are valid. | Yes (if `visible=False`) |
| `y` | Float | The entity's spatial Y coordinate. | Floating-point values. Normalized pitch coordinates are expected (e.g., `[0, 1]`), though out-of-bounds values are valid. | Yes (if `visible=False`) |
| `confidence` | Float | The confidence score of the tracking detection. | `[0.0, 1.0]`. When source provides no confidence, a convention of `1.0` is used for visible detections and `0.0` for missing. | No |
| `visible` | Boolean | Indicates whether the entity is actively tracked in this frame. | `True` or `False`. | No |

## Coordinate Conventions
- **X-Axis:** `x` increases from left to right in the canonical representation.
- **Y-Axis:** `y` increases top-to-bottom OR bottom-to-top, but this orientation must be explicitly supported by the source conventions. We do not assume silently.
- **Normalization:** Source coordinate orientation and scaling must be explicitly normalized during the ingestion pipeline. Canonical coordinates are represented as normalized pitch coordinates (typically `[0, 1]`) unless a later implementation determines that a metric/world-coordinate representation is preferable.
- **Pitch Dimensions:** We do not invent pitch dimensions. If absolute metric coordinates are required, they will be derived from explicit dataset metadata.

## Identity Conventions
- **Player IDs:** `player_id` must remain stable within a given match. If a player leaves and re-enters, or if the tracker maintains identity, the ID should match. 
- **Team Names:** The `team` field uses a consistent, controlled vocabulary (`"home"`, `"away"`).
- **The Ball:** The ball is **NOT** represented as a player row in the canonical tracking dataframe unless explicitly needed by a downstream task. If ball data exists, it should be represented separately (e.g., in a dedicated `ball_tracking` dataframe sharing the same temporal indices, or as distinct match-level metadata).

## Visibility and Confidence
- **Visible:** `visible` is a strict boolean flag. It allows downstream tasks to quickly filter out missing detections without performing null checks on coordinate floats.
- **Confidence:** `confidence` is a continuous numeric value in `[0, 1]`. 
  - When the source tracker provides a confidence or probability score (e.g., from an object detection model), that exact score is mapped to `confidence`.
  - When the source (such as Metrica) does not provide confidence, the chosen convention is to hardcode `confidence = 1.0` when coordinates exist, and `confidence = 0.0` when they do not. We do not silently invent an arbitrary model confidence score.

## Missing-Data Semantics
- **Player Not on Field vs. Missing Detection:** A player who is substituted off the field or hasn't entered yet should generally have `visible=False` and `x=NaN, y=NaN` for those frames, rather than omitting their rows entirely, to maintain a consistent rectangular shape for temporal processing if desired.
- **Unavailable vs. Invalid Coordinates:** `NaN` will be used exclusively to represent unavailable or invalid coordinates. Downstream processors should never use arbitrary magic numbers (like `-999` or `0.0` when `0.0` is a valid pitch location) to represent missing spatial data.

## Example Canonical Table

| match_id | frame | timestamp | player_id | team | x | y | confidence | visible |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metrica_1 | 1 | 0.04 | 11 | home | 0.0008 | 0.4823 | 1.0 | True |
| metrica_1 | 1 | 0.04 | 1 | home | 0.4431 | 0.5401 | 1.0 | True |
| metrica_1 | 1 | 0.04 | 25 | away | 0.9050 | 0.4746 | 1.0 | True |
| metrica_1 | 1 | 0.04 | 14 | home | NaN | NaN | 0.0 | False |
| metrica_1 | 2 | 0.08 | 11 | home | 0.0009 | 0.4823 | 1.0 | True |
| metrica_1 | 2 | 0.08 | 1 | home | 0.4432 | 0.5401 | 1.0 | True |
| metrica_1 | 2 | 0.08 | 25 | away | 0.9049 | 0.4746 | 1.0 | True |
| metrica_1 | 2 | 0.08 | 14 | home | NaN | NaN | 0.0 | False |
| metrica_1 | 3 | 0.12 | 11 | home | 0.0011 | 0.4823 | 1.0 | True |
| metrica_1 | 3 | 0.12 | 1 | home | 0.4434 | 0.5401 | 1.0 | True |

## Source-specific ingestion mapping

### Metrica
- **`match_id`**: Provided explicitly to the parser during ingestion rather than guessing it from the filename.
- **`frame` / `timestamp`**: Mapped directly from the `Frame` and `Time [s]` columns.
- **`player_id`**: Extracted from the wide-format column headers (e.g., `"Player11"` becomes `"11"`).
- **`team`**: Mapped statically based on which source file the tracking row originates from (Home or Away dataset).
- **`x` / `y`**: Mapped directly from the `[0, 1]` normalized coordinate pairs.
- **`confidence`**: Metrica does not provide confidence. Mapped as `1.0` when coordinates are present, and `0.0` when `NaN`.
- **`visible`**: `True` if `x` and `y` are present, `False` if `NaN`.
- **Ball Handling**: The `Ball_X` and `Ball_Y` columns are extracted into a separate ball tracking structure and excluded from this canonical player tracking table.

### SkillCorner
TBD — pending dataset inspection

### SoccerNet-GSR
TBD — pending dataset inspection

### Synthetic degradation
TBD — pending dataset inspection

## Design decisions and rationale

**Why a "Tall" Format?**
Raw tracking data (like Metrica) frequently uses a "wide" format where each column represents a single spatial dimension of a specific player (e.g., 28 columns for 14 players' X/Y coordinates). While wide formats can be highly compressed and easy to write to CSV, a **tall format** (one row per player per frame) is vastly superior for downstream analytical processing because:

1. **Vectorized Operations:** Modern data analysis libraries (Pandas, Polars) and machine learning frameworks (PyTorch) operate more naturally on feature columns (`x`, `y`) rather than dynamic column names (`Player1_X`, `Player2_X`).
2. **Variable Roster Sizes:** Tall formats naturally accommodate variable numbers of players on the pitch (e.g., due to red cards or tracking failures) without needing a dynamic number of sparse columns.
3. **Group-by Capabilities:** Aggregating data by `team`, `player_id`, or `frame` is trivial in a tall format but requires complex unpivoting in a wide format.
4. **Consistency:** All data sources can be mapped to exactly 8 columns regardless of how many players they track or how they name them.
