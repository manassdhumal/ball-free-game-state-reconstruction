# SoccerNet-GSR Access Checklist

## Audit Status

As of Step 77, authorized SoccerNet-GSR data access has **not** been verified. No validation sequence, source video, matching labels, or metadata directory is present locally. Do not claim that access has been granted.

The public task page was not reachable from the current audit environment, so current account, NDA, password, and legal terms must be confirmed through the official SoccerNet channel before data is downloaded.

## Official References

- SoccerNet GSR task page: https://www.soccer-net.org/tasks/new-game-state-reconstruction
- Official GSR code repository: https://github.com/SoccerNet/sn-gamestate
- Local checked-out repository: `soccernet-gamestate/`
- Local instructions: `soccernet-gamestate/README.md`
- Local challenge rules: `soccernet-gamestate/ChallengeRules.md`

## Required Request and Access Confirmation

The team must confirm through the official SoccerNet data-access process:

- whether a SoccerNet account is required;
- whether validation videos are publicly downloadable or require an approved request;
- whether an NDA or separate video-use agreement applies;
- whether a dataset password or expiring download authorization is required;
- whether labels can be downloaded separately from videos;
- whether storage on university infrastructure, institutional GPU/HPC, Google Drive, or Colab-mounted Drive is permitted under the applicable dataset terms.

Do not place passwords, tokens, account details, private links, or credentials in this repository. Use the provider's approved download mechanism and the hosting platform's secret facility where necessary.

## One-Sequence Data Package

Before execution, obtain one complete validation sample consisting of:

1. One validation source video or the source frames required by TrackLab.
2. The corresponding `Labels-GameState.json` annotation file.
3. The sequence's metadata and image/frame index mapping.
4. Any required calibration or pitch metadata referenced by the dataset.
5. Dataset version evidence, with labels checked for the repository's documented version requirement (currently version >= 1.3).
6. A checksum and acquisition record stored outside Git.

The baseline documentation expects a dataset layout similar to:

```text
data/SoccerNetGS/
  valid/
    <sequence directory>/
      <video or frames>
      Labels-GameState.json
      <other official metadata>
```

The exact filenames and folder contents must be taken from the downloaded validation metadata. Do not select `SNGS-04` or another identifier until it is observed in the actual validation split.

## Sequence Record to Complete After Access

| Field | Value |
|---|---|
| Sequence ID | TBD until validation metadata is available |
| Video ID | TBD until validation metadata is available |
| Split | `valid` |
| Expected FPS | TBD from official metadata/video metadata |
| Label file | Expected `Labels-GameState.json`; verify actual path |
| Image/frame count | TBD from official metadata |
| Dataset version | TBD; verify labels version >= 1.3 |
| Download size | TBD; measure actual one-sequence package |
| Source/checksum | TBD; record outside Git |

## Expected Size

The repository documentation does not provide a reliable per-sequence download size. Do not estimate a one-sequence size from the 69.3 MB official example prediction archive: that archive contains 49 JSON predictions and is not source video or labels. Measure the actual downloaded sequence, labels, and metadata after access is granted.

## Storage Recommendation

Preferred order:

1. Institutional GPU/HPC storage controlled by the university or project team.
2. University-managed research storage with documented access controls.
3. Google Drive only if the dataset terms and university policy permit it.
4. Colab-mounted Drive for a short test only when persistent storage and access terms are confirmed.

Do not place the dataset, video, labels, checkpoints, or raw GSR outputs in Git. Keep credentials outside notebooks and configuration files.

The team must obtain a written or otherwise authoritative confirmation of storage permissions where the provider's terms are unclear. This document does not itself establish legal permission.

## Verification After Download

Before installing or running TrackLab:

- confirm the split is `valid`;
- confirm the chosen sequence ID exists in the downloaded metadata;
- confirm the source video/frames and matching label file are present;
- inspect `Labels-GameState.json` version;
- record frame count, FPS, dimensions, and file sizes;
- calculate checksums outside Git;
- confirm no file is a submission example or prediction archive mislabeled as source data;
- verify the data directory is readable from the GPU environment;
- preserve the acquisition and provenance record.

## Access Decision

Step 77 cannot be marked complete until one sequence and its matching labels are legitimately accessible, the storage permission is understood, and the metadata record above is filled from actual files. The single next action is to request/obtain authorized access to one validation sequence and labels through the official SoccerNet process.
