# Pitch Insight

Build the full research platform 'BALL-FREE GAME STATE RECONSTRUCTION' (Subtitle: 'PLAYER-ONLY FOOTBALL INTELLIGENCE') as a serious sports-science and football analytics desktop workstation.

Visual Design:

- Strict professional sports-science research aesthetic. NO neon, glowing borders, purple gradients, glassmorphism, floating cards, cartoon pitch graphics, or AI-marketing buzzwords.
- Dark charcoal palette (#111315 background, #1E2225 panels, subtle 1px borders rgba(255,255,255,0.08)).
- Accent: Muted football pitch green (#527A5C) used sparingly for active states, selected items, and key markers.
- Pitch team colors: Muted blue (#4F6F8F) for Home, muted red/orange (#9A5D52) for Away.
- Tabular numerals, clean dense hierarchy (Inter/Geist), high information density.

Layout:

- Left sidebar: compact navigation with subtle branding ('BALL-FREE GAME STATE RECONSTRUCTION', 'PLAYER-ONLY FOOTBALL INTELLIGENCE'), version (v0.8 Research Build), settings, docs.
- Persistent top bar: Dataset (Metrica Sample_Game_1), Experiment (Controlled Analysis), Frame (1250, 00:50.0, 25 FPS), Backend (MOCK DATA / CONNECTED), GSR (INTEGRATION PENDING).

Core Pages & Views:

1. Overview: Research ops dashboard, pipeline stages (Tracking Data -> Player-Only State -> Possession Inference -> Tactical Analysis -> Counterfactual Ranking -> Robustness), key metrics (Possession F1, Tactical MAE, Event F1, Ranking MRR), central football pitch, recent experiments summary.
2. Match Explorer: Dataset/match selector, frame playback controls (0.25x, 0.5x, 1x, 2x, play/pause, prev/next frame), interactive pitch, timeline with event markers and possession transitions, player inspector panel.
3. Game State: State feature matrix table (with type: Model-Derived, Heuristic, Measured Proxy), pressure map, support graph, team geometry, forward options.
4. Possession & Events: Possession timeline, current inferred possessor, predicted vs annotated events comparison tabs, evaluation table (TP, FP, FN, Precision, Recall, F1).
5. Tactical Analysis: Tactical advantage, pressure, support, space, forward progression, team width/length, compactness, defensive spread, local density charts/scatter plots.
6. Counterfactual Decisions: Source player selector, candidate pass lines drawn on pitch, candidate ranking table (Rank, Target, Distance, Pressure, Space, Line Obstruction, Decision Score, Stability), 'Why this option ranked here' analytical breakdown, and prominent disclaimer: 'This is a model-ranked hypothetical option generated from the observed player-only state. It does not indicate that the player selected this option.'
7. Decision Robustness: Perturbation controls (Coordinate Jitter, Player Dropout, Availability Jitter, severity), reference vs perturbed comparison, Top-1/Top-3 retention, Spearman/Kendall rank correlation, decision flip rate.
8. Robustness Benchmark: Condition comparison table, degradation and recovery curves.
9. Noise Lab: Interactive noise parameters (missing detection %, track gaps, jitter sigma, fragmentation) with real-time impact preview.
10. Tracking Quality: Missingness rate, visible player completeness, coordinate validity, jitter, distribution charts.
11. Recovery & Mitigation: Reference -> Degraded -> Interpolated -> Kalman Filter workflow with comparative metrics table (F1, MAE, runtime).
12. Experiments: Experiment management table (EXP_080A_042 etc.) with filterable metadata and slide-over experiment details panel.
13. Research Results: Benchmarks, method comparisons, and honest negative findings section (e.g. baseline comparisons).
14. Broadcast / GSR: Blocked / integration pending status with disabled architecture pipeline modules clearly explaining unverified status.
15. Limitations: Explicit research limitations covering data, models, evaluation, and broadcast tracking.

Pitch Component:

- Professional football pitch with accurate pitch markings, muted dark green turf, clean player tokens with squad numbers, trajectory/velocity vectors, selectable players, pass line overlays, and analytical hover tooltips (coordinates, velocity, nearest opponent/teammate, local density).

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/977ec505-030a-4e43-8626-16c2e613784f).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
