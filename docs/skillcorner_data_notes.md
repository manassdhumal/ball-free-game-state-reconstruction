# SkillCorner Open Data Notes

## 1. Official Repository & Source
- **Repository URL:** `https://github.com/SkillCorner/opendata`
- **License:** Open Data under SkillCorner terms & attribution guidelines (joint initiative with PySport).
- **Source Nature:** Broadcast-derived tracking data collected via computer vision, object detection, and trajectory extrapolation from single-camera broadcast video feeds.

## 2. Repository Commit & Version
- **Cloned Commit:** `c1e17a0` ("Align viz tools docs page with the README and fix feature wording")
- **Repository Branch:** `master`
- **Release Status:** Active 2024/2025 Open Data release.

## 3. Dataset Scope
- **Domain:** Professional association football tracking and contextual analytics.
- **Coverage:** 10 full matches from the Australian A-League (AUS 1).
- **Season:** 2024/2025 Regular Season.
- **Provided Data Layers per Match:**
  1. `{id}_match.json`: Match metadata, lineups, referee, pitch dimensions, period frame bounds, player roles.
  2. `{id}_tracking_extrapolated.jsonl`: 10 FPS tracking data for all active on-pitch players (detected + extrapolated) and ball.
  3. `{id}_dynamic_events.csv`: Contextual Game Intelligence event annotations.
  4. `{id}_phases_of_play.csv`: Concurrent in-possession and out-of-possession tactical phase segmentation.
- **Aggregates:** Season-level physical aggregates (`aus1league_physicalaggregates_20242025.csv`), off-ball runs (`aus1league_obraggregates_20242025.csv`), and passing aggregates (`aus1league_passingaggregates_20242025.csv`).

## 4. Match Inventory Summary
Across the 10 matches in `data/raw/skillcorner/data/matches.json`:

| Match ID | Date / Time (UTC) | Competition | Season | Home Team | Away Team | Score | Pitch Size (m) | Home Attack (P1, P2) | Tracked Players | Total Frames | Active Frames | Detection Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1886347** | 2024-11-30 04:00 | A-League | 2024/2025 | Auckland FC | Newcastle Jets | 2 - 0 | 104 × 68 | R→L, L→R | 29 | 59,061 | 43,458 | 59.28% |
| **1899585** | 2024-12-07 04:00 | A-League | 2024/2025 | Auckland FC | Wellington Phoenix | 2 - 1 | 104 × 68 | R→L, L→R | 30 | 60,530 | 40,681 | 55.19% |
| **1925299** | 2024-12-21 06:00 | A-League | 2024/2025 | Brisbane Roar | Perth Glory | 0 - 1 | 105 × 68 | R→L, L→R | 32 | 61,301 | 47,942 | 57.88% |
| **1953632** | 2024-12-31 08:00 | A-League | 2024/2025 | CC Mariners | Melbourne City | 1 - 1 | 105 × 68 | L→R, R→L | 30 | 59,250 | 43,912 | 62.11% |
| **1996435** | 2025-02-01 06:00 | A-League | 2024/2025 | Sydney FC | Adelaide United | 4 - 1 | 105 × 68 | L→R, R→L | 30 | 57,621 | 43,531 | 65.80% |
| **2006229** | 2025-03-07 08:35 | A-League | 2024/2025 | Melbourne City | Macarthur FC | 2 - 0 | 105 × 68 | R→L, L→R | 32 | 59,270 | 44,171 | 62.92% |
| **2011166** | 2025-04-12 05:00 | A-League | 2024/2025 | Wellington Phoenix | Melbourne Victory | 2 - 3 | 105 × 68 | R→L, L→R | 32 | 71,851 | 39,015 | 52.42% |
| **2013725** | 2025-04-27 07:00 | A-League | 2024/2025 | Western United | Sydney FC | 1 - 0 | 106 × 68 | L→R, R→L | 30 | 70,251 | 44,992 | 53.95% |
| **2015213** | 2025-05-03 08:00 | A-League | 2024/2025 | Western United | Auckland FC | 4 - 2 | 106 × 68 | L→R, R→L | 32 | 72,101 | 48,542 | 52.54% |
| **2017461** | 2025-05-17 09:35 | A-League | 2024/2025 | Melbourne Victory | Auckland FC | 0 - 1 | 105 × 68 | L→R, R→L | 32 | 71,451 | 40,404 | 51.16% |

## 5. Native File & Tracking JSONL Structure
Each `{id}_tracking_extrapolated.jsonl` contains line-delimited JSON records. Each line represents one frame at 10.0 Hz with the following top-level schema:

```json
{
  "frame": 10,
  "timestamp": "00:00:00.00",
  "period": 1,
  "ball_data": {
    "x": 0.32,
    "y": 0.38,
    "z": 0.13,
    "is_detected": true
  },
  "possession": {
    "player_id": null,
    "group": null
  },
  "image_corners_projection": {
    "x_top_left": -52.52,
    "y_top_left": 39.0,
    "x_bottom_left": -23.21,
    "y_bottom_left": -37.05,
    "x_bottom_right": 22.76,
    "y_bottom_right": -36.88,
    "x_top_right": 50.99,
    "y_top_right": 39.0
  },
  "player_data": [
    {
      "x": -39.63,
      "y": -0.08,
      "player_id": 51009,
      "is_detected": false
    }
  ]
}
```

## 6. Frame & Temporal Conventions
- **Sampling Rate:** Fixed 10.0 Hz ($dt = 0.10\text{ s}$ per frame).
- **Frame Indexing:** Integer sequence starting from frame `0` or `10`.
- **Timestamp Formatting:** Native string format `"HH:MM:SS.ss"` (or `"MM:SS.ss"`), representing match clock elapsed time.
- **Period Segmentation:**
  - `period = 1`: First half.
  - `period = 2`: Second half (match clock restarts at `"00:45:00.00"`).
  - `period = null`: Pre-match, half-time interval, and post-match dead-time frames (contains empty `player_data: []`).

## 7. Player & Team Identity Semantics
- **Player ID:** Integer `player_id` in `player_data` (e.g., `51009`) matches `player['id']` in the match metadata `{id}_match.json`.
- **Team Assignment:** Mapped deterministically from `{id}_match.json` using `player['team_id']`:
  - If `player['team_id'] == home_team['id']` $\rightarrow$ `team = "home"`
  - If `player['team_id'] == away_team['id']` $\rightarrow$ `team = "away"`
  - If a player is missing from metadata, a clear `ValueError` is raised (no silent assignment).
- **Canonical Player ID Formatting:** `canonical player_id = str(player_id)` (e.g., `"51009"`). Source-faithful player ID string; jersey numbers are not used as player IDs to avoid collision. Cross-match uniqueness is provided by `match_id`.

## 8. Detection vs. Extrapolation Semantics
SkillCorner uses a hybrid computer vision + tracking model:
- `is_detected = true` (`visible = True`, `confidence = 1.0`): The player was directly observed on camera in the broadcast video frame.
- `is_detected = false` (`visible = False`, `confidence = 0.0`): The player was off-screen (outside the camera frustum) or temporarily occluded; their $(x, y)$ coordinates were estimated by SkillCorner's trajectory extrapolation model.
- **Valid Extrapolated Positions:** `visible = False` indicates lack of direct visual detection on camera, but does **NOT** mean missing coordinates. Extrapolated coordinates are valid, populated estimates.
- **Completeness Invariant:** In all 10 matches, every active play frame contains **exactly 22 player positions** (on-screen detected + off-screen extrapolated).
- **Detection Rate:** Across all 10 matches, on average **57.7%** of player observations are directly detected on camera, while **42.3%** are extrapolated.

## 9. Ball & Possession Representation
- **Ball Tracking:** Stored in `ball_data` with keys `x`, `y`, `z`, and `is_detected`.
- **Ball Separation:** The ball is **NOT** included in the canonical player tracking DataFrame and is extracted separately.
- **Possession Metadata:** `possession` dictionary contains `player_id` (ID of player in possession, or `null`) and `group` (`"home team"`, `"away team"`, or `null`).

## 10. Coordinate Conventions & Transformation Rationale

### Source Coordinate System (SkillCorner Native)
- **Coordinate Units:** Meters (metric).
- **Coordinate Origin $(0, 0)$:** The **center spot** of the football pitch.
- **X-Axis (Long Dimension):** Longitudinal pitch axis along the broadcast camera view ($[-L/2, +L/2]$), where $L$ is the physical pitch length ($104\text{m}$, $105\text{m}$, or $106\text{m}$). Left is negative, right is positive.
- **Y-Axis (Short Dimension):** Lateral pitch axis ($[-W/2, +W/2]$), where $W$ is the physical pitch width ($68\text{m}$). Near touchline (bottom of screen) is negative; far touchline (top of screen) is positive.
- **Out-of-Bounds Range:** Player coordinates extend slightly beyond physical lines (e.g., $X \in [-60.1, +63.5]$, $Y \in [-40.5, +43.3]$) for players taking throw-ins, corner kicks, or running off-pitch. Out-of-bounds coordinates are not clipped.

### Pitch Dimensions per Match
Pitch length varies by venue and is explicitly recorded in `{id}_match.json`:
- `1886347`, `1899585`: $104\text{ m} \times 68\text{ m}$ (Mount Smart Stadium)
- `1925299`, `1953632`, `1996435`, `2006229`, `2011166`, `2017461`: $105\text{ m} \times 68\text{ m}$
- `2013725`, `2015213`: $106\text{ m} \times 68\text{ m}$ (GMHBA Stadium / Ironbark Fields)

### Canonical Normalization Transformation
To achieve full compatibility with the canonical schema (v0.2.0) and allow cross-dataset comparison with Metrica Sports:
$$x_{\text{norm}} = \frac{x_{\text{metric}} + L/2}{L}$$
$$y_{\text{norm}} = \frac{y_{\text{metric}} + W/2}{W}$$

Where $L = \text{pitch\_length}$ and $W = \text{pitch\_width}$.
- Within pitch boundaries: $x_{\text{norm}} \in [0.0, 1.0]$ and $y_{\text{norm}} \in [0.0, 1.0]$.
- Center spot $(0, 0)$ maps to $(0.5, 0.5)$.
- Bottom-left corner $(-L/2, -W/2)$ maps to $(0.0, 0.0)$.
- Top-right corner $(+L/2, +W/2)$ maps to $(1.0, 1.0)$.
- The parser provides `normalize_coords=True` (default) for canonical $[0, 1]$ normalized coordinates, and `normalize_coords=False` to preserve raw metric meters.

## 11. Canonical Schema Mapping Decisions

| SkillCorner Native Source | Canonical Field | Data Type | Canonical Mapping Decision & Rationale |
| :--- | :--- | :--- | :--- |
| `match_id` argument / filename | `match_id` | String | Unique string identifier (e.g., `"skillcorner_1886347"`). |
| `frame` | `frame` | Integer | Preserved directly as integer frame index. |
| `timestamp` (`"HH:MM:SS.ss"`) | `timestamp` | Float | Converted from timestamp string into total elapsed seconds ($H \times 3600 + M \times 60 + S$). |
| `player_id` | `player_id` | String | Source-faithful string: `str(player_id)` (e.g. `"51009"`). |
| `player['team_id']` vs `home_team['id']` | `team` | String | Controlled vocabulary: `"home"` or `"away"`. |
| `x` (meters) | `x` | Float | Normalized to $[0, 1]$ via $(x + L/2)/L$ using match pitch length $L$. |
| `y` (meters) | `y` | Float | Normalized to $[0, 1]$ via $(y + W/2)/W$ using match pitch width $W$. |
| `is_detected` | `confidence` | Float | SkillCorner does not output continuous detection probabilities. When `is_detected == True`, mapped to `1.0` (direct visual observation). When `is_detected == False` (extrapolated), mapped to `0.0` to avoid manufacturing artificial probabilistic confidence. |
| `is_detected` | `visible` | Boolean | Mapped as `bool(is_detected)`: `True` indicates the player is directly visible on broadcast camera; `False` indicates the player is off-camera / occluded and extrapolated (coordinates are still populated). |

## 12. Known Limitations & Source Caveats
1. **Identity Accuracy:** SkillCorner notes that $\approx 97\%$ of player identities are accurate; rare ID swaps occur during dense penalty-box scuffles.
2. **Trajectory Smoothness:** Raw extrapolated positions can exhibit minor velocity discontinuities upon camera re-entry when switching from extrapolation to visual detection.
3. **Variable Pitch Lengths:** Normalization must use match-specific metadata pitch dimensions rather than assuming a universal $105\text{m}$ pitch length.
4. **Camera Viewport Constraints:** On average, only 14–18 players are in camera view at any moment; 4–8 players are continuously extrapolated.

## 13. Open Questions & TBD Items
- **Attacking Direction Inversion:** SkillCorner records `home_team_side` (e.g. `['right_to_left', 'left_to_right']`). Ingestion preserves physical pitch coordinates without flipping halves; half-normalization will be handled in downstream tactical feature extractors.
- **Referees:** Referees are not currently tracked in the Open Data tracking stream.
