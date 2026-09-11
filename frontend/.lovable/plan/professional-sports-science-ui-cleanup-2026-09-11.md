# Professional Sports-Science UI Cleanup

## Goal

Refine the complete workstation into a sharper, denser research interface while preserving all existing analysis features and interactions.

## Implementation

1. **Lock the visual system to flat research surfaces**
   - Normalize global radii to 2px and remove decorative rounding from controls, tooltips, overlays, and shared UI primitives used by the app.
   - Set the charcoal hierarchy explicitly: workspace background `#111315`, primary panels `#1E2225`, control/secondary surfaces `#24282B`, and 1px white-at-8% dividers.
   - Keep muted pitch green `#527A5C` only for selected navigation, active controls, top-ranked options, and tactical indicators.
   - Remove shadows, glow, blur, gradients, translucent glass effects, and nonessential transitions/animations from shared and route-level presentation.

2. **Create a denser shared workstation language**
   - Tighten the main content inset, panel headers/bodies, metric rows, navigation spacing, form controls, and repeated gaps.
   - Raise page headings to 18–20px and section headings to 13–14px while keeping compact 11–12px labels.
   - Apply the mono/tabular treatment consistently to frames, times, coordinates, seeds, identifiers, and measured values.
   - Replace boxed groups nested inside panels with flat rows, hairline separators, compact definition lists, and table-like stage layouts.

3. **Strengthen the persistent metadata strip**
   - Rework the top bar as a terminal-style run context strip with clear label/value columns for dataset, match, experiment, frame/time/FPS, condition, seed, model, backend, and GSR status.
   - Preserve responsive access by allowing controlled horizontal scrolling rather than collapsing or hiding scientific context.

4. **Make pitch analysis dominant on key screens**
   - **Overview:** move the reconstructed pitch ahead of secondary pipeline/experiment information and give it the largest page area.
   - **Match Explorer:** widen and enlarge the pitch, compress playback/timeline controls, and reduce the player inspector to a narrower data rail.
   - **Game State:** make the pressure-map pitch the primary wide visual, with the feature matrix and geometry presented as dense adjacent/below tables.
   - **Counterfactual Decisions:** enlarge the candidate-pass pitch and keep the analytical breakdown compact; retain the full-width candidate ranking immediately below.
   - Make pitch height responsive without inline visual sizing, preserving accurate aspect ratio and readable overlays.

5. **Elevate tables and scientific charts**
   - Standardize tables with compact sticky-style headers, stronger column separation/alignment, stable numeric widths, subtle row rules, and horizontal overflow on narrow screens.
   - Prioritize Candidate Ranking, Feature Matrix, Experiments, Benchmark, Event Evaluation, and other research tables as full-width analytical surfaces.
   - Improve chart axis/grid/tooltip contrast using semantic tokens, increase practical plot height where needed, and remove soft/decorative styling.

6. **Clean every remaining view**
   - Apply the same flat row/divider treatment to possession, tactical analysis, robustness, noise lab, recovery, tracking quality, results, broadcast, and limitations.
   - Convert card-like pipelines, workflow stages, warnings, and result explanations into compact workstation bands or segmented data rows.
   - Keep warning and blocked states legible but restrained; do not introduce additional colors or decoration.

7. **Validate the complete workstation**
   - Check all routes for consistent tokens, typography, corners, spacing, and absence of gradients/glows/blur/shadows.
   - Verify desktop layouts and narrower viewports for clipping, table access, metadata readability, and pitch prominence.
   - Exercise pitch selection, playback, filters, tabs, sliders, and experiment details; confirm no console errors.

## Technical Notes

- Changes remain presentation-only; existing mock research data, calculations, navigation, and interactions stay intact.
- Shared styling will be centralized in the global token sheet and reusable panel/chart primitives, with focused route layout changes where hierarchy differs.
- Raw SVG colors required for pitch marks and player labels will be moved behind semantic tokens where practical.
