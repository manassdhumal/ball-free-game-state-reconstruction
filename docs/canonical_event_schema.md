# Canonical Event Schema

## 1. Purpose
This document defines the Canonical Event Schema for the Ball-Free Game State Reconstruction project. The schema provides a unified representation of event data, bridging provider-specific formats (like Metrica) into a standard structure. This enables downstream tasks such as possession inference, pass/turnover detection, and synchronization with canonical tracking data.

## 2. Schema Version
**Version:** 0.1.0

## 3. Data Dictionary

### Required Fields
These fields must be present and non-null for every event.

* `match_id` (str): Unique identifier for the match.
* `event_id` (int): A unique sequential identifier for the event within the match. Since raw providers (like Metrica) may lack explicit event IDs, these are generated sequentially based on chronological order.
* `period` (int): The half or period of the match (e.g., 1, 2).
* `start_frame` (int): The video/tracking frame number when the event began.
* `start_timestamp` (float): The time in seconds from the start of the period when the event began.
* `event_type` (str): The primary category of the event (e.g., 'PASS', 'SHOT'). We use a controlled vocabulary for core types, mapping provider-specific types to canonical ones where possible, or retaining the raw type if no mapping exists.
* `team` (str): The team associated with the event. Controlled vocabulary: `'HOME'`, `'AWAY'`, or `'UNKNOWN'`.
* `from_player_id` (str): The canonical ID of the primary player involved in the event.

### Optional Fields
These fields may be null (or `None`) depending on the event type and provider support.

* `end_frame` (int): The frame number when the event concluded (e.g., when a pass is received).
* `end_timestamp` (float): The time in seconds when the event concluded.
* `event_subtype` (str): Additional context or classification for the event (e.g., 'CORNER KICK', 'HEAD-ON TARGET-GOAL'). Maintained as a raw string to preserve detailed provider nuances.
* `to_player_id` (str): The canonical ID of the secondary/receiving player (e.g., the recipient of a pass).
* `start_x` (float): The normalized X coordinate [0.0, 1.0] where the event began.
* `start_y` (float): The normalized Y coordinate [0.0, 1.0] where the event began.
* `end_x` (float): The normalized X coordinate [0.0, 1.0] where the event ended.
* `end_y` (float): The normalized Y coordinate [0.0, 1.0] where the event ended.

## 4. Design Decisions & Conventions

### 4.1 Null/Missing Semantics
* **Event Participants:** If an event only involves one player (e.g., a shot or ball out), `to_player_id` is set to null.
* **Coordinates:** Events without a clear spatial location (e.g., some tactical changes or cards) will have null for `start_x`/`start_y`. Events without a clear spatial conclusion (e.g., an interception or foul) will have null for `end_x`/`end_y`.
* **Timing:** If an event is instantaneous, `end_frame` and `end_timestamp` may either be equal to the start values or null, depending on the parser implementation.

### 4.2 Event Type Conventions
* `event_type` should be normalized to a standard set (e.g., PASS, SHOT, RECOVERY, BALL OUT) to simplify analytics.
* `event_subtype` remains a raw string, as standardizing the long tail of subtypes (e.g., 51 subtypes in Metrica Sample Game 1) is overly complex for v0.1.0.

### 4.3 Coordinate Conventions
* Spatial data is normalized to a [0.0, 1.0] range, consistent with the canonical tracking schema.
* Origin (0,0) is defined as the top-left corner of the pitch.

### 4.4 Player Identity Conventions
* To ensure compatibility with tracking data, player identities in `from_player_id` and `to_player_id` must follow the format `<TEAM>_<JERSEY_NUMBER>` (e.g., `HOME_10`, `AWAY_19`).

## 5. Metrica-Specific Mapping

Based on the analysis of `Sample_Game_1_RawEventsData.csv`:
* **Row Index:** Used to generate `event_id` (1-indexed).
* **Team:** Mapped from 'Home'/'Away' to canonical `'HOME'`/`'AWAY'`.
* **Type:** Mapped directly to `event_type`.
* **Subtype:** Mapped directly to `event_subtype` (empty strings become null).
* **From/To:** Metrica's 'Player19' combined with the 'Team' column maps to `from_player_id`='AWAY_19'. Empty 'To' strings become null.
* **Coordinates:** Metrica uses 'NaN' strings for missing coordinates. These are parsed to null (None in Python). `Start X`, `Start Y`, `End X`, `End Y` are already in [0,1] format.

## 6. Example Canonical Event Table

| match_id | event_id | period | start_frame | event_type | team | from_player_id | to_player_id | start_x | start_y | end_x | end_y |
|----------|----------|--------|-------------|------------|------|----------------|--------------|---------|---------|-------|-------|
| sample_1 | 1        | 1      | 1           | SET PIECE  | AWAY | AWAY_19        | null         | null    | null    | null  | null  |
| sample_1 | 2        | 1      | 1           | PASS       | AWAY | AWAY_19        | AWAY_21      | 0.45    | 0.39    | 0.55  | 0.43  |
| sample_1 | 3        | 1      | 3           | PASS       | AWAY | AWAY_21        | AWAY_15      | 0.55    | 0.43    | 0.58  | 0.21  |
