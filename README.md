# Vision Code – README

## Overview
This module provides the core computer vision pipeline for domino perception and manipulation. It handles real-time detection, classification, spatial modeling, and physical interaction with dominoes in the environment.

---

## Features

### 1. Detection, Recognition & Classification
- Script: LiveDomino.py
- Detects upright and fallen dominoes, recognizes their face values, and classifies them for downstream tasks.

### 2. World Modeling & Mapping
- Maintains a global representation of domino positions and states.
- Files involved:
  - modelling.py       – Core modeling logic
  - aim-fsm/worldmap.py – World state management
  - domino.py          – Domino object abstraction
  - qml/WorldMapView.qml – QML-based visualization interface

### 3. Kicking & Picking
- Script: kickpickrefine.py
- Executes physical manipulation of dominoes from the side, including kicking (toppling) and picking (grabbing) actions.

---

## Pretrained Weights

| Weight File       | Model Architecture | Purpose |
|-------------------|---------------------|---------|
| bestieee.pt       | YOLO26              | Segmentation of upright dominoes |
| different.pt      | EfficientNetB0      | Half-face value detection for upright dominoes |
| fallen.pt         | YOLO26              | Segmentation of fallen dominoes |
| fallenhalf.pt     | EfficientNetB0      | Half-face value detection for fallen dominoes |

---

## Dependencies
- Python 3.x
- PyTorch / YOLO26
- EfficientNetB0 (via timm or similar)
- QML (for world map visualization)
- VEX AIM tools environment

---

## Usage
1. Run LiveDomino.py to start the detection pipeline.
2. The world map is updated in real time via modelling.py and worldmap.py.
3. Use kickpickrefine.py to issue manipulation commands based on current world state.

---

## Notes
- All weights are pre-trained and should be placed in the appropriate directory before execution.
- The system is designed for integration with the VEX AIM robotics platform.
