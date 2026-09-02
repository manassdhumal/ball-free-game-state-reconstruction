# Ball-Free Possession Baseline

## 1. Problem Definition

In many real-world tracking scenarios — broadcast camera feeds, partial GPS data, SkillCorner extrapolated tracks — the ball position is either unavailable, unreliable, or delayed. This module addresses the question:

**Can we infer which player/team has possession using only player tracking geometry?**

This is a *baseline* heuristic, not the final research model. It establishes a performance floor and reveals where player geometry alone is insufficient.

## 2. Input / Output

### Input
- **Canonical player tracking DataFrame** (schema v0.2.0)
  - Required columns: `match_id`, `frame`, `timestamp`, `player_id`, `team`, `x`, `y`, `visible`
  - Ball columns (`Ball_X`, `Ball_Y`) are **never accessed** even if present

### Output
- **Possession DataFrame**: one row per frame with `possessor_id`, `team`, `possession_score`, `confidence`
- **Candidate events DataFrame**: `possession_change`, `pass_candidate`, `turnover_candidate`, `recovery_candidate`
- **Evaluation metrics**: precision, recall, F1 at configurable temporal tolerances

## 3. Anti-Leakage Contract

The inference pipeline (stages 1–4) **never** accesses:
- Ball coordinates (`Ball_X`, `Ball_Y`)
- Event labels
- Future frames
- Future ground-truth possession

Event labels are consumed **only** in stage 5 (evaluation), **after** all predictions are finalised.

## 4. Spatial Graph Definition

For each frame, we construct a lightweight `FrameGraph` containing:

### 4.1 Node Attributes
| Attribute | Source |
|---|---|
| `player_id` | Canonical tracking |
| `team` | Canonical tracking |
| `x`, `y` | Canonical tracking (visible players only) |

### 4.2 Computed Features
| Feature | Description |
|---|---|
| `pairwise_distances` | N×N Euclidean distance matrix (diagonal = ∞, no self-edges) |
| `nearest_teammate_dist` | Distance to closest same-team player |
| `nearest_teammate_id` | ID of closest same-team player |
| `nearest_opponent_dist` | Distance to closest opposing-team player |
| `nearest_opponent_id` | ID of closest opposing-team player |
| `local_density` | Count of players within configurable radius (default 0.05 ≈ 5m) |

### 4.3 Optional KNN Edges
- k-nearest-neighbour edge list (default k=5)
- Each edge stores `(source_id, target_id, distance)`
- Directed (each node emits k edges)

### 4.4 Design Decisions
- **Lightweight dataclass**, not NetworkX — minimal dependencies
- **Invisible players excluded** — `visible=False` rows are filtered out
- **Normalised coordinates** — distances in [0, 1] pitch space
- All distances computed via vectorised NumPy broadcasting

## 5. Heuristic Possession Score

Each visible player receives a weighted score from 7 features:

| Feature | Weight | Description |
|---|---|---|
| Opponent proximity (inverse) | 0.25 | Players closer to opponents are more likely contesting the ball |
| Teammate proximity (inverse) | 0.15 | Tight teammate clusters suggest build-up zones |
| Local density | 0.15 | Dense areas indicate active play zones |
| Centrality | 0.10 | Distance from pitch centre (inverted); peripheral GKs score lower |
| Movement continuity | 0.15 | Low acceleration = smooth, deliberate movement (ball carrier signature) |
| Forward progress | 0.10 | Velocity toward attacking goal |
| Team compactness | 0.10 | Possessing teams tend to be more compact |

### 5.1 Normalisation
Each feature is min-max normalised to [0, 1] within the frame before weighting. This prevents scale differences from dominating.

### 5.2 Weight Configuration
Weights are configurable via a dictionary parameter. The defaults above are chosen based on sports analytics intuition, not optimised on data.

## 6. Temporal Continuity Rule

To prevent possession from switching randomly every frame:

1. **Switch margin** (default: 0.15): A new candidate must exceed the current possessor's score by at least this margin to trigger a switch
2. **Persistence window** (default: 5 frames = 0.2s at 25 FPS): Minimum consecutive frames a possessor must hold before any switch is allowed
3. **Minimum score threshold** (default: 0.10): If no player exceeds this threshold, the frame is marked as no-possession (`possessor_id = None`)

### 6.1 Rationale
- In real football, possession changes are relatively rare events (typically 200–400 per match)
- At 25 FPS, a 90-minute match has ~135,000 frames
- Without temporal smoothing, noisy heuristic scores would produce thousands of spurious switches

## 7. Event Candidate Logic

Candidate events are derived from possession transitions:

| Candidate Event | Condition |
|---|---|
| `possession_change` | `possessor_id` changes between consecutive frames |
| `pass_candidate` | `possessor_id` changes but `team` remains the same |
| `turnover_candidate` | `possessor_id` changes and `team` changes |
| `recovery_candidate` | Team regains possession after losing it (A→B→A pattern) |

These are **predicted candidates**, not ground truth.

## 8. Evaluation Methodology

### 8.1 Matching Rule
- **Greedy 1-to-1 temporal matching**: candidate pairs sorted by absolute frame difference
- A predicted event at frame $f_{pred}$ matches a reference event at frame $f_{ref}$ if $|f_{pred} - f_{ref}| \leq \text{tolerance\_frames}$
- Each reference event matches at most one predicted event

### 8.2 Tolerances
| Tolerance (seconds) | Tolerance (frames at 25 FPS) |
|---|---|
| ±0.5s | ±12 frames |
| ±1.0s | ±25 frames |
| ±2.0s | ±50 frames |

### 8.3 Event Type Mapping
| Predicted Type | Reference Types (Metrica) |
|---|---|
| `pass_candidate` | PASS |
| `turnover_candidate` | BALL LOST, BALL OUT |
| `recovery_candidate` | RECOVERY |

### 8.4 Metrics
- **Precision**: fraction of predicted events that match a reference event
- **Recall**: fraction of reference events that match a predicted event
- **F1**: harmonic mean of precision and recall

## 9. Limitations

### 9.1 Fundamental Limitations
- **Player geometry is not sufficient** for reliable possession inference. The ball's position is the primary determinant of possession — this baseline deliberately ignores it.
- **Goalkeepers** often score low on opponent-proximity features even when clearly in possession.
- **Set pieces** (corners, free kicks, throw-ins) create dense player clusters that confuse proximity-based heuristics.
- **Long passes** create temporal gaps where no player appears to have possession geometrically.

### 9.2 Heuristic Limitations
- Feature weights are hand-tuned, not learned from data.
- Min-max normalisation is frame-local, so absolute feature magnitudes are lost.
- The "forward progress" feature assumes home attacks right and away attacks left — this may not hold after half-time unless tracking coordinates are flipped.

### 9.3 Evaluation Limitations
- Metrica event data is itself not perfect ground truth for possession.
- The event type mapping (e.g., `turnover_candidate` → `BALL LOST`) is approximate.
- Precision/recall depend heavily on tolerance — ±0.5s is strict for a heuristic baseline.

## 10. Sensitivity to Parameters

| Parameter | Effect of Increasing |
|---|---|
| `switch_margin` | Fewer switches, higher temporal stability, potentially missed real changes |
| `persistence_window` | Longer minimum tenure, smoother but laggier |
| `min_score_threshold` | More no-possession frames |
| `density_radius` | Higher density counts, less discriminative |
| `opponent_proximity` weight | Favours players in contested zones |

## 11. Future Upgrade Path

This baseline establishes a performance floor. Planned upgrades include:

1. **Conditional Random Field (CRF)**: Model temporal dependencies as a structured sequence labelling problem, replacing the rule-based temporal continuity.
2. **Graph Neural Network (GNN)**: Replace the hand-crafted spatial features with learned node embeddings on the player graph.
3. **Velocity field features**: Use team-level velocity fields and formation templates.
4. **Multi-scale temporal context**: Use sliding windows of multiple sizes rather than frame-by-frame scoring.

These upgrades are **not** implemented in this baseline.

## 12. Module API Reference

### `src/possession/spatial_graph.py`
- `FrameGraph` — dataclass for per-frame spatial graph
- `build_frame_graph(frame_df, frame, timestamp, ...)` → `FrameGraph`
- `build_spatial_features(tracking_df, ...)` → `list[FrameGraph]`

### `src/possession/possession_baseline.py`
- `score_frame(graph, ...)` → `pd.DataFrame` — per-player scores for one frame
- `predict_possession(tracking_df, ...)` → `pd.DataFrame` — full possession timeline
- `infer_events(possession_df)` → `pd.DataFrame` — candidate events
- `evaluate_predictions(predicted, reference, ...)` → `pd.DataFrame` — P/R/F1
- `possession_summary(possession_df)` → `dict` — high-level statistics
