# EEGDash Data-Viz Design

Date: 2026-05-04

This document defines the first EEGDash visual identity element for figures,
dataset cards, task cards, benchmark outputs, README images, and Hugging Face
cards.

The goal is to mirror MOABB's evidence-first plotting discipline without copying
MOABB's brand. MOABB uses compact scientific plots, source-aware titles,
chance-level context, muted grids, and a consistent accent rail. EEGDash should
use the same level of consistency, but with its own data-discovery and
reproducibility identity.

## Core Motif: The Data Rail

The EEGDash identity element is the **Data Rail**:

- a thin EEGDash-blue horizontal line near the top of plots and cards;
- a short EEGDash-orange pulse segment at the left edge;
- optional small ticks when the object represents a split, benchmark, or
  reproducibility artifact.

This mirrors MOABB's publication-style accent line, but the meaning is different:

- blue rail: dataset/task evidence path;
- orange pulse: EEG signal and active decoding task;
- small ticks: manifest, split, provenance, or export checkpoints.

Use it everywhere EEGDash wants to say: "this object is traceable, reproducible,
and ready for ML."

## Palette

Use the existing EEGDash logo colors as anchors:

| Role | Token | Hex | Usage |
| --- | --- | --- | --- |
| Primary | `EEGDASH_BLUE` | `#006CA3` | Data rail, selected task, links, axes accent |
| Primary dark | `EEGDASH_BLUE_DARK` | `#004A76` | Titles, strong borders, dark labels |
| Accent | `EEGDASH_ORANGE` | `#F7941D` | Pulse segment, warnings that need attention, active baseline |
| Sky | `EEGDASH_SKY` | `#4F8CFF` | Secondary model/dataset series |
| Mint | `EEGDASH_MINT` | `#22D3EE` | Validated/export-ready status |
| Ink | `EEGDASH_INK` | `#102A43` | Main text |
| Muted | `EEGDASH_MUTED` | `#64748B` | Subtitles, source text |
| Grid | `EEGDASH_GRID` | `#7A8CA0` | Grid lines and neutral dividers |
| Surface | `EEGDASH_SURFACE` | `#F7FBFE` | Card backgrounds |

The palette must remain colorblind-aware. Do not rely on color alone for split
warnings, leakage warnings, or benchmark status. Pair color with labels,
markers, line styles, or icons.

## Figure Rules

Every EEGDash exported figure should include:

- short title;
- subtitle with dataset/task/split context;
- visible axis units;
- sample count or subject count;
- split strategy when relevant;
- chance level or baseline line when relevant;
- source/provenance note;
- generation date or manifest ID for benchmark artifacts.

Avoid decorative charts. A figure should encode at least one of:

- dataset readiness;
- task definition;
- split safety;
- leakage or overlap;
- chance/baseline comparison;
- uncertainty or subject variability;
- channel/montage coverage;
- export completeness.

## Required Visuals

Initial EEGDash visuals should be:

1. Dataset atlas card.
2. Task manifest card.
3. Split audit plot.
4. Leakage check report.
5. Baseline score plot with chance line.
6. Paired model-vs-baseline plot.
7. Learning curve.
8. Channel coverage plot.
9. Pretraining/evaluation overlap plot.
10. OpenEEG-Bench export completeness card.

## Plot Layout

Use this hierarchy:

- Data Rail at the top.
- Title and subtitle in the top-left.
- Plot area with restrained grid lines.
- Legend outside the dense data area when possible.
- Source/provenance text at the lower-left.

Recommended defaults:

```text
figure background: white
plot background: white or #F7FBFE
grid alpha: 0.24 to 0.32
spines: remove top/right, mute left/bottom
title: strong, dark blue-gray
subtitle/source: muted slate
accent line: #006CA3
pulse: #F7941D
```

## Card Layout

Task and dataset cards should use the same identity:

- top Data Rail;
- small uppercase kicker, for example `Task Snapshot`;
- clear title;
- compact metrics row;
- badges for license, split, ML readiness, export status, and leakage checks;
- no nested cards.

The card should answer a concrete user question before they run code:

```text
Can I use this dataset/task for ML, under what split, with which labels,
licenses, and caveats?
```

## Relationship To MOABB

Reuse MOABB plots when the data is a MOABB evaluation result. Do not duplicate
MOABB's plotting API for `score_plot`, `paired_plot`, `summary_plot`, or
`meta_analysis_plot`.

EEGDash-specific visuals should focus on:

- dataset search and atlas views;
- task readiness;
- split manifests;
- leakage checks;
- pretraining/evaluation overlap;
- channel coverage;
- OpenEEG-Bench export completeness.

## Implementation

The reusable identity tokens and helper functions live in:

```text
docs/plot_dataset/identity.py
```

The helpers provide:

- EEGDash color constants;
- a reusable categorical palette;
- a Matplotlib styling function;
- a Plotly layout function;
- a small HTML card helper.

Future production code can move this module into `eegdash.viz` once the API is
stable.
