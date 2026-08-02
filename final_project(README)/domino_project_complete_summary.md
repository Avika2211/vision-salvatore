# Domino Perception and World-Map Integration Project Summary

(This documentation is concluded by codex)

## 1. Project Overview

This project extends the robot’s existing `aim_fsm` world-map system so that physical dominoes on a table can be detected from the camera image, localized in world coordinates, tracked persistently, rendered in the 3D world-map viewer, used as planning obstacles, and labeled with pip counts when visual evidence is reliable.

The final system combines three major components:

1. **Domino world-map integration**
   - Detects flat dominoes from camera frames.
   - Projects their image-space geometry into world coordinates.
   - Stores them as persistent `DominoObj` objects inside `robot.world_map.objects`.
   - Allows dominoes to follow the standard world-map lifecycle: pending confirmation, visible, missing, and reclaimed.
   - Treats dominoes as rectangular obstacles for path planning.

2. **Divider-line detection and orientation recovery**
   - Detects the center divider line on the domino face.
   - Uses the divider as a short-axis cue.
   - Computes the domino long axis as perpendicular to the divider.
   - Handles long-side-facing and short-side-facing views, including strong foreshortening.

3. **Pip counting and face labeling**
   - Rectifies each domino into a canonical crop.
   - Splits the crop into two halves using the detected divider.
   - Counts colored pips on each half with a classical, non-CNN pipeline.
   - Uses color, blob geometry, relative layout priors, repair logic, and confidence gates.
   - Stores and renders final face labels such as `4-5`, or returns `?` when evidence is unreliable.

The main design principle throughout the project is conservative robustness: the system should prefer returning `?` over confidently inserting a wrong face label or unstable orientation.

---

## 2. Overall Runtime Pipeline

The final pipeline operates as follows:

1. The robot receives a camera frame.
2. YOLO segmentation detects candidate domino masks.
3. Each mask is mapped back from the model input size to the original camera frame.
4. A contour and coarse rotated rectangle are extracted using OpenCV.
5. Duplicate overlapping detections are suppressed.
6. A divider-line detector runs inside the domino patch.
7. If the divider is found, the domino long axis is set perpendicular to the divider.
8. If no divider is found, the system falls back to the coarse rectangle-based long-axis estimate for geometry, but skips pip labeling.
9. The image-space center and long-axis endpoints are projected to the ground plane.
10. A `DominoObj` candidate is created with world position, orientation, dimensions, confidence, and debug metadata.
11. The world map associates the candidate with existing objects or promotes it after repeated observations.
12. If a valid divider exists, the pip-counting system rectifies the domino, splits it into halves, counts pips, and assigns a label.
13. The final mapped domino is shown in the camera overlay, world-map viewer, and path-planning obstacle set.

---

## 3. Main Code Structure

### 3.1 Domino-Specific Lab Files

#### `lab8/domino_world_detector.py`

This is the core domino detector wrapper. It:

- loads the existing YOLO segmentation model;
- reuses Roboflow-style resize and mask-remapping helpers;
- extracts contours and rotated rectangles;
- estimates centers, quadrilaterals, axes, divider lines, and confidence;
- exposes `DominoObservation` objects instead of only drawing overlays;
- contains the divider detection logic;
- connects to the label-provider abstraction for pip recognition.

It also defines:

- `DominoLabelProvider`, used for real pip-count labeling;
- `NullDominoLabelProvider`, used when face recognition is disabled.

#### `lab8/domino_half_face_pipeline.py`

This file contains the final pip-counting pipeline for rectified half-domino crops. It performs:

- active-region extraction;
- local background estimation;
- saturation/chroma response computation;
- blob filtering;
- repair-candidate tracking;
- color-matched repair;
- relative layout scoring;
- count selection;
- confidence gating.

#### `lab8/DominoWorldMap.py`

This is the runnable demonstration FSM. It:

- enables domino detection through `StateMachineProgram`;
- launches the camera viewer, world-map viewer, and path viewer;
- overlays raw detector observations and promoted world-map objects;
- shows ids, divider lines, axes, and optional face labels;
- provides a debugging command to print mapped dominoes.

#### `lab8/test_domino_world_detector.py`

This is a small geometry sanity-test script. It checks that:

- synthetic domino masks produce observations;
- centers are extracted correctly;
- axis angles are reasonable;
- opposite long-axis directions are treated as equivalent;
- missing dividers force unknown labels.

---

### 3.2 Shared `aim_fsm` Framework Changes

#### `vex-aim-tools/aim_fsm/program.py`

`StateMachineProgram` now accepts domino-related options:

- `domino=False`
- `domino_labeling=False`
- `domino_conf_threshold=0.3`
- `domino_weights_path=None`

If `domino=True`, the robot creates a `robot.domino_detector`, making domino detection a shared runtime capability rather than a one-off lab script.

#### `vex-aim-tools/aim_fsm/worldmap.py`

This is the main integration point. It adds `DominoObj`, which stores:

- persistent object id, such as `Domino.a`;
- world pose: `x`, `y`, and axis-only `theta`;
- physical dimensions: length, width, and height;
- obstacle semantics;
- optional face metadata;
- latest image-space debug metadata, including center, quadrilateral, axis, divider, mask area, and confidence.

It also adds domino candidate creation and association. New detections are projected into world coordinates, compared against existing dominoes, and inserted into the normal world-map lifecycle.

Domino association uses both:

- center-position difference;
- long-axis angle difference modulo 180 degrees.

This reduces identity swaps when nearby dominoes have different orientations.

#### `vex-aim-tools/aim_fsm/rrt.py`

Dominoes are added as path-planning obstacles through `generate_domino_obstacle()`. Each domino is modeled as an oriented rectangle using its world pose, physical length, and width.

#### `vex-aim-tools/aim_fsm/path_planner.py`

Dominoes can be converted into goal shapes, making them compatible with the existing path-planning setup.

#### `vex-aim-tools/aim_fsm/pilot.py`

`PilotToObject.CenterTheObject` can now center on dominoes using `obj.image_center`, even though dominoes do not come from the built-in SDK detector.

#### `vex-aim-tools/viewer/worldmap_model.py`

The viewer model now recognizes `DominoObj` as type `"domino"` and exports its dimensions, position, orientation, z-height, face label, and half-count metadata.

#### `vex-aim-tools/qml/WorldMapView.qml`

The QML viewer now renders dominoes as flat rectangular solids. It also renders the center divider and colored top-surface pips when counts are known.

---

## 4. Domino World-Map Integration

### 4.1 Original Goal

The world-map integration was designed to make dominoes first-class world objects. The intended v1 behavior was:

- detect flat dominoes from the camera image;
- estimate each domino’s world position;
- estimate long-axis orientation modulo 180 degrees;
- store dominoes persistently in `robot.world_map.objects`;
- follow the normal lifecycle of other objects;
- act as path-planning obstacles.

Face recognition was originally left as a future extension, but the later pip-counting system now fills that hook.

### 4.2 Candidate Creation

For each valid `DominoObservation`, the world map:

1. projects the image centroid to the ground plane;
2. projects both image-space long-axis endpoints to the ground plane;
3. converts robot-relative coordinates into world coordinates;
4. computes world `theta` from the projected axis endpoints;
5. creates a `DominoObj` candidate;
6. stores image debug metadata;
7. appends the candidate into the world-map candidate list.

### 4.3 Persistence and Lifecycle

Dominoes reuse the existing `WorldObject` lifecycle. This means they can be:

- pending before confirmation;
- promoted after stable repeated observations;
- visible when currently detected;
- missing when expected but not seen;
- reclaimed when a missing object reappears.

The insertion logic requires repeated observations, which helps prevent one-frame false positives from becoming persistent objects.

### 4.4 Planning Integration

The path planner treats mapped dominoes as oriented rectangular obstacles. This is appropriate because dominoes are flat objects with known approximate physical dimensions.

The main calibration issue is whether the dimensions and projection accuracy are conservative enough for real robot navigation.

---

## 5. Divider-Line Detection and Orientation Recovery

### 5.1 Problem

The original orientation estimate came from `cv2.minAreaRect(contour)`. This worked when a domino was viewed broadside, because the long side was clearly longer in the image.

However, when the short edge faced the robot, perspective made the silhouette close to square. In those cases, `minAreaRect` could swap the apparent long and short sides, causing the estimated orientation to rotate by about 90 degrees.

The solution was to detect the physical divider line on the domino face. Since the divider is aligned with the domino short axis, the long axis can be recovered as the perpendicular direction.

### 5.2 Failed and Abandoned Approaches

#### Baseline: `minAreaRect` Long Axis

This was simple and often worked in easy views, but failed under strong foreshortening because a nearly square contour does not have a stable long side.

#### Raw Image Hough Detection

The first divider detector searched for dark line segments in the original image using grayscale conversion, blur, black-hat morphology, thresholding, and `cv2.HoughLinesP`.

It failed because the divider was not always a clean binary line. Pips, shadows, highlights, and pip edges often produced stronger or more connected dark segments.

#### Geometrically Constrained Raw Search

The next version restricted the search to eroded interiors, central bands, and orientations roughly aligned with the expected short axis.

This reduced some false positives but was too rigid. Under perspective, the divider could move away from the image-space midpoint, especially for short-side-facing or distant dominoes.

#### Adaptive Position Constraints

The search region was widened for foreshortened boxes, and offset along the long direction was penalized less than offset across the short direction.

This was more physically reasonable, but the detector was still fundamentally searching for dark features in raw image space, so it remained sensitive to blur, shadows, and pip structure.

### 5.3 Final Divider Solution

The final solution is based on rectified-patch stripe detection.

#### Step 1: Coarse Shape from Segmentation

The segmentation mask is mapped back to the original frame. The largest contour is extracted and `cv2.minAreaRect` gives a coarse quadrilateral. This rectangle is not trusted as the final orientation; it is only used to define a stable warp.

#### Step 2: Rectify to Canonical Coordinates

The domino patch is warped into a canonical rectangle:

- long dimension: `120 px`;
- short dimension: `60 px`.

The segmentation mask is warped into the same canonical coordinate frame. This ensures that the divider search only reasons over pixels inside the actual domino region.

#### Step 3: Search Candidate Divider Angles

The detector searches over candidate divider orientations:

- angle range: `±35°`;
- step size: `5°`.

The search allows for imperfect rectification and perspective-induced skew. It does not assume that the divider is perfectly horizontal or vertical in the rectified patch.

#### Step 4: Build a 1D Profile

For each candidate stripe angle, the detector:

- defines a coordinate system perpendicular to the candidate stripe;
- projects valid mask pixels onto that normal direction;
- accumulates grayscale intensity into a 1D profile;
- smooths the profile with Gaussian blur.

This turns the 2D divider problem into a 1D dark-band detection problem.

#### Step 5: Detect a Dark Band With Lighter Sides

The final detector does not simply choose the darkest line. Instead, it scores whether a narrow center band is darker than both neighboring side bands.

This rejects many false positives because a true divider should look like a dark stripe surrounded by lighter domino surface.

Current parameters include:

- `DIVIDER_CENTER_HALF_WIDTH = 2`
- `DIVIDER_SIDE_HALF_WIDTH = 4`
- `DIVIDER_SIDE_GAP = 2`
- `DIVIDER_RESPONSE_THRESHOLD = 10.0`

#### Step 6: Map Divider Back to Image Space

Once the best stripe is selected in canonical coordinates, its endpoints are mapped back to the original image using the inverse perspective transform.

#### Step 7: Compute Long Axis

The divider is treated as the short-axis cue. The domino long axis is set perpendicular to the divider. This produces:

- `divider_endpoints` for debugging;
- `axis_endpoints` for projection;
- `axis_theta` for orientation.

If the divider detector fails, the system falls back to the old rectangle-based axis estimate.

### 5.4 Why This Works

The final divider method is stronger because:

- it does not rely on a nearly square contour having a stable orientation;
- it does not require a connected binary black line;
- it searches for a stripe-like intensity structure rather than arbitrary dark pixels;
- it searches across multiple candidate angles;
- it uses the warped segmentation mask, so background pixels are excluded.

---

## 6. Pip Counting and Face Labeling

### 6.1 Goal

After segmentation, pose estimation, and divider detection were working, the remaining goal was to recognize the number of colored pips on each half of each domino.

The target behavior was:

- crop each detected half separately;
- count pips using a classical, non-CNN baseline;
- if no divider is detected, skip counting and label the face as `?`;
- if counting is trustworthy, return labels such as `4-5`;
- if counting is ambiguous, return `?` instead of a forced label;
- show counts in the robot camera overlay, debug window, and 3D world map.

### 6.2 Failed and Abandoned Pip-Counting Attempts

#### Attempt 1: Whole-Half Color Blob Counting

The first approach cropped each half and counted colored blobs using Lab color distance, HSV saturation, thresholding, connected components, and contour filtering.

It failed because bevels, vertical edges, bottom edges, and shadows often became candidate blobs. This caused inflated labels such as `6-6`.

#### Attempt 2: Hard Top-Face ROI

The next approach used a conservative hand-defined top-face ROI, cutting off bottom strips and excluding boundary bands.

This removed some edge artifacts, but it was too brittle. Perspective and foreshortening changed where the true top face appeared, causing real pips to be rejected.

#### Attempt 3: Adaptive Top-Surface Support Mask

This version tried to estimate the visible top surface from non-white pixels and only count blobs inside that support.

It failed because the support mask was sensitive to shadows, body gradients, and bevel transitions. It sometimes cut away valid pip regions, especially the far row of `6`.

#### Attempt 4: Hardcoded Template Masks

Count-specific pip templates were drawn and compared against detected blobs.

This was abandoned because fixed image-space templates did not survive perspective changes. The template was often misaligned, upside down, or too tight.

The useful lesson was that geometric layout matters, but it must be used as a soft relative prior rather than a hard mask.

### 6.3 Final Pip-Counting Direction

The successful approach reuses CNN-style normalized crops but keeps the classifier classical.

The final crop path is:

1. rectify each domino to a canonical `120x60` crop;
2. use the detected divider line in rectified coordinates;
3. split the crop into two halves using the actual divider geometry;
4. mask out pixels outside each half;
5. crop each half to its non-white bounds;
6. letterbox each half to a square `64x64` image.

This made the pip-counting problem much cleaner because each half is normalized before classical analysis begins.

### 6.4 Active Half Mask

For each `64x64` half crop, the system estimates an active region:

- pixels with any RGB channel below `250` are marked active;
- only the largest connected component is kept;
- the active mask is eroded with a `3x3` elliptical kernel;
- if erosion removes too much, the system falls back to the original largest component.

This avoids relying on a fixed ROI and instead follows the actual visible half crop.

### 6.5 Local Background Estimate

The system estimates the local domino-body background inside the half crop:

- convert to HSV and Lab;
- select likely body pixels from the active region;
- choose pixels with saturation below the 65th percentile and value above the 35th percentile;
- compute the median Lab color of these body pixels.

This allows pip color response to be measured relative to the local domino surface rather than the entire image.

### 6.6 Color Response and Candidate Mask

Candidate pip pixels are based on saturation and Lab color contrast:

- saturation response increases when saturation exceeds about `24`;
- chroma response increases when Lab distance exceeds about `12`;
- the final response is a weighted combination of saturation and chroma;
- response is clipped to the active mask and lightly blurred.

The binary candidate mask requires:

- saturation greater than `28`;
- Lab distance greater than `16`;
- membership inside the active mask.

Morphological opening and closing with a `3x3` ellipse are applied to clean the mask.

### 6.7 Blob Filtering

Connected components are scored as candidate pips. Rejection criteria include:

- zero width or height;
- too small relative to active area;
- too large relative to active area;
- elongated shape;
- low circularity;
- low extent;
- empty component;
- low mean saturation.

Accepted blobs store:

- contour;
- component mask;
- image-space center;
- normalized center inside the active bounding box;
- area ratio;
- quality score;
- mean saturation;
- mean Lab chroma;
- circular mean hue.

The blob quality score combines:

- shape;
- area;
- saturation;
- chroma.

Using active-area-relative size was important because the crop can scale under perspective and letterboxing.

### 6.8 Repair Candidates

Some rejected blobs are kept as repair candidates if they are not hard failures. Hard failures such as degenerate, empty, large, elongated, or sparse components are not repairable.

Soft rejected blobs can be reconsidered when their size, saturation, and chroma are plausible. This helps recover true pips that were rejected because of mild thresholding or partial connection issues.

### 6.9 Color-Matched Repair

The system can promote a rejected blob if it matches the color of accepted high-confidence blobs on the same half.

The logic is:

- use accepted blobs with strong quality, saturation, and chroma as color references;
- require at least two reference blobs;
- compute their circular mean hue;
- promote rejected candidates whose hue, chroma, saturation, and area ratio are compatible.

This is especially useful when one real pip is weak but has the same color family as other visible pips.

### 6.10 Physical Color Prior

The physical domino set uses count-specific colors:

- `1`: purple / burgundy;
- `2`: green;
- `3`: purple;
- `4`: cyan;
- `5`: green;
- `6`: amber.

The system compares the candidate blobs’ circular mean hue to the expected hue family for each count. This is used only as a secondary tie-breaker, not as the main classifier.

This design avoids allowing lighting changes or camera auto-white-balance to dominate the decision.

### 6.11 Relative Layout Priors

Hardcoded templates were replaced with soft relative-layout scores.

The system checks whether the detected blob positions are geometrically plausible for each count:

- `0`: valid only when there are no blobs and global color response is low;
- `1`: one blob near the center;
- `2`: separated diagonal pair;
- `3`: diagonal pair plus center blob;
- `4`: weak layout prior, because four detected blobs are usually reliable enough;
- `5`: four corners plus a center, roughly forming two diagonals;
- `6`: two columns and three rows, with soft tolerance for perspective and foreshortening.

The `6` case receives special care because the far row is often compressed, weak, or partially missed.

### 6.12 Candidate Count Selection

The final count is not always the raw number of accepted blobs. The system considers candidate blob sets generated by:

- keeping the base accepted set;
- dropping one weak blob;
- adding one or two repair blobs.

Each candidate count is scored using:

- relative layout score;
- average blob quality;
- color-family score;
- edit penalty.

The weights differ by count. For example, `4` relies more on blob quality, while `6` can receive more help from color and layout when amber support is strong.

A repaired candidate must beat the base candidate by enough margin. This prevents the repair logic from hallucinating pips too easily.

### 6.13 Special Handling for Sixes

The `6` label was unstable because far-row pips sometimes flickered in and out.

Two stabilizers were added:

1. **Strong amber support**
   - If a candidate `6` has strong amber color evidence, the system reduces the edit penalty and loosens some layout thresholds.

2. **Temporal stabilization**
   - The label provider remembers up to eight recent stable labels.
   - If a previous label was `6` and a new frame flickers to `4` or `5`, the system keeps `6` unless the new evidence is strongly contradictory.

This directly addresses observed `6` flickering.

### 6.14 Failure Gates

Even after scoring, a half can be marked invalid. Failure reasons include:

- `zero-conflict`: no blobs, but strong color response;
- `weak-blobs`: raw blob confidence too low;
- `too-many`: more than six raw blobs without a strong drop-to-six result;
- `count-jump`: selected count differs too much from base count;
- `ambiguous`: selected count differs by one but lacks enough margin;
- `low-confidence`: nonzero count has insufficient confidence.

If either half is invalid, the whole face becomes `?`, and `half_counts` becomes `(None, None)`.

### 6.15 Divider Gate

The divider is a hard gate for pip counting.

If no divider is detected:

- pip counting is skipped;
- face label becomes `?`;
- half counts become `(None, None)`;
- the debug panel explicitly reports that counting was skipped.

This avoids interpreting partial crops, blank tops, or bad segmentations as valid zeros.

---

## 7. Rendering and Visualization

### 7.1 Camera Overlay

The camera overlay shows:

- raw current detector observations;
- promoted world-map dominoes;
- object ids;
- quadrilaterals;
- long-axis lines;
- detected divider lines;
- optional face labels.

Visual conventions:

- green quadrilateral: mapped domino object;
- white line: inferred long axis;
- cyan / blue line: detected divider;
- orange quadrilateral: raw detector observation.

These distinctions make it easier to debug whether errors come from raw detection, divider recovery, axis computation, or world-map projection.

### 7.2 `domino-label-debug` Window

A live debug window was added for pip recognition. For each detected domino, it shows a contact sheet containing:

- rectified full domino with divider line;
- left-half overlay;
- right-half overlay;
- response map;
- crop and active mask;
- relation mask;
- candidate mask;
- rejected blobs;
- accepted blobs.

Each half overlay annotates:

- count;
- confidence;
- layout score;
- color score;
- color family;
- base blob count;
- whether the result came from base, drop, or add repair;
- failure gate if invalid.

Rejected blobs are marked with reasons such as small, large, elongated, noncompact, or low saturation.

This debug view was essential because most improvements came from visually identifying exactly which blobs were accepted, rejected, or repaired.

### 7.3 3D World-Map Viewer

The Qt Quick 3D viewer renders dominoes as flat cuboids. The model exposes:

- object type;
- dimensions;
- z-height;
- face label;
- half-count metadata.

The QML rendering draws:

- a domino cuboid;
- a black center divider;
- colored flush pip discs on the top surface;
- `?` for unknown halves.

Pip colors are:

- `1`: dark burgundy;
- `2`: green;
- `3`: purple;
- `4`: cyan / teal;
- `5`: green;
- `6`: amber.

A previous rendering version used a dark well cylinder under each colored pip, but the dark layer dominated visually and made pips appear black. The final version renders a single flush colored disc.

---

## 8. Duplicate Detection Suppression

YOLO sometimes produced two overlapping masks for one physical domino. To prevent duplicate world objects, detector-side suppression was added:

1. build all valid observations;
2. sort observations by confidence;
3. keep the highest-confidence observation first;
4. suppress later observations if their mask overlaps an already kept mask by more than 50% of the smaller mask area.

This happens before pip labeling and world-map insertion, so one physical domino is less likely to become two persistent objects.

---

## 9. Testing and Validation

### 9.1 Static Checks

Compilation checks were repeatedly run with commands such as:

```bash
python3 -m py_compile lab8/domino_world_detector.py
python3 -m py_compile vex-aim-tools/viewer/worldmap_model.py vex-aim-tools/viewer/worldmap_viewer.py
```

### 9.2 Synthetic and Unit-Style Checks

The detector test checks that:

- synthetic masks produce observations;
- axis angle is reasonable;
- opposite axis directions are treated as equivalent;
- foreshortened masks are not rejected;
- missing divider forces `?`;
- missing divider suppresses half counts.

Example command:

```bash
PYTHONPATH=/Users/zhuhengjin/Code/15494-CogRob /Users/zhuhengjin/Code/15494-CogRob/venv/bin/python lab8/test_domino_world_detector.py
```

### 9.3 Real Snapshot Checks

Real images from `final_project/snapshots` and live camera screenshots were used to inspect:

- vertical edges falsely counted as pips;
- bottom/front bevels counted as multiple pips;
- missed far-row pips on `6`;
- true pips rejected by fixed ROIs;
- color-based repair of weak pips;
- unstable `6` flickering;
- no-divider behavior.

### 9.4 End-to-End Runtime Checks

The demo can be launched with:

```python
runfsm("DominoWorldMap")
```

Expected visual checks include:

- Robot View shows detector outlines, divider, half centers, and counts;
- `domino-label-debug` shows per-half internals;
- World Map shows mapped domino cuboids with colored top-surface pips;
- unknown faces show `?`;
- path planning avoids promoted domino obstacles.

---

## 10. Important Lessons Learned

### 10.1 For Orientation

- `minAreaRect` alone is not enough for foreshortened dominoes.
- A divider line is a better orientation cue than the apparent silhouette.
- Raw-image Hough detection is too fragile for divider recovery.
- Rectification makes the divider-search problem much easier.
- A true divider should be detected as a dark band with lighter sides, not merely as a dark line.
- Searching multiple candidate angles in canonical coordinates makes the method robust to imperfect rectification.

### 10.2 For Pip Counting

- Counting pips directly in the original perspective image is too noisy.
- Fixed ROIs fail under perspective and foreshortening.
- Adaptive support masks can still cut away valid pip regions.
- Hard templates are too brittle.
- Soft relative layout priors are useful when combined with blob quality.
- Color is helpful as a repair cue and tie-breaker, but should not be the main classifier.
- The `6` case needs special handling because the far row often flickers.
- Returning `?` is better than forcing a confident but wrong count.
- Debug visualization is essential for improving classical vision pipelines.

### 10.3 For World-Map Integration

- Dominoes fit naturally into the existing `WorldObject` lifecycle.
- Association should use both position and orientation.
- Flat-object projection requires real-world calibration because segmentation centroids may not behave like upright object detections.
- Obstacle dimensions should be checked against the real physical domino set.

---

## 11. Remaining Assumptions and Risks

### 11.1 Divider Detection Assumptions

The final divider detector assumes:

- the domino is lying flat;
- the segmentation mask is reasonably accurate;
- the divider is visible enough to produce a dark-band signal.

It may struggle with:

- heavy blur;
- strong glare;
- poor segmentation;
- pips that dominate the intensity profile.

### 11.2 World Projection Risks

Ground-plane projection still needs real robot validation. Potential issues include:

- centroid projection may not perfectly match the physical domino center;
- flat objects are lower-profile than objects the original projection code may have been tuned for;
- left/right image distortion may affect segmentation-derived points differently;
- projected axis endpoints may be unstable near image borders or at far distances.

### 11.3 Association Risks

Current thresholds are plausible but hand-tuned. Risks include:

- nearby dominoes swapping identities;
- strict orientation thresholds preventing valid updates;
- loose thresholds incorrectly reclaiming the wrong object.

### 11.4 Lifecycle Risks

Because insertion requires six successive update cycles, dominoes may appear slowly if segmentation is intermittent or the robot update loop is slow.

Generic missing-object heuristics may also mark flat objects missing too aggressively under occlusion or partial visibility.

### 11.5 Planning Risks

Dominoes are modeled as oriented rectangular obstacles with nominal dimensions. If dimensions are too small, paths may cut too close. If too large, the planner may become unnecessarily conservative.

---

## 12. TODO: Runtime Code Placement Cleanup

One architectural cleanup remains.

Currently, shared runtime files depend on lab-side code. Specifically:

- `vex-aim-tools/aim_fsm/program.py`
- `vex-aim-tools/aim_fsm/worldmap.py`

ultimately rely on:

- `lab8/domino_world_detector.py`
- `lab8/DominoSegment.py`

This works, but it is not the ideal dependency direction for a shared background perception feature.

### Recommended Refactor

Move the runtime domino detector into `vex-aim-tools/aim_fsm`, including:

- Roboflow-style fit resize;
- mask remapping to original image coordinates;
- domino geometry extraction helpers;
- divider detection;
- runtime label-provider interfaces.

Keep `lab8` for:

- training code;
- lab demos;
- experimental scripts.

Optionally move `lab8/DominoWorldMap.py` into `final_project/` if it is meant to be the final project demo rather than a lab artifact.

The intended long-term structure should be:

- `vex-aim-tools`: shared runtime perception and world-map code;
- `lab8`: course-lab segmentation experiments and model-development code;
- `final_project`: project-specific demos, snapshots, and documentation.

This cleanup is not a functional blocker, but it matters if dominoes are intended to remain a first-class `aim_fsm` feature.

---

## 14. Final Bottom Line

The project now implements a complete domino perception-to-world-map pipeline:

- YOLO segmentation detects domino masks.
- Mask geometry produces coarse object observations.
- Duplicate overlapping detections are suppressed.
- Rectified-patch divider detection recovers robust orientation.
- Ground-plane projection inserts dominoes into the world map.
- Dominoes persist as `DominoObj` objects and follow the normal object lifecycle.
- Dominoes render in the 3D world-map viewer.
- Dominoes act as path-planning obstacles.
- Rectified half-face analysis counts colored pips when reliable.
- Unknown or ambiguous labels are safely represented as `?`.

The most important technical shift was moving from brittle raw-image heuristics to normalized, rectified representations. For both divider detection and pip counting, rectification made classical vision feasible by reducing perspective variation before applying geometry, color, and layout reasoning.

The most important engineering shift was integrating dominoes into the existing shared world-map pipeline rather than maintaining a separate lab-only object list. This gives dominoes persistence, association, visualization, planning semantics, and compatibility with existing robot behaviors.

The system is structurally complete. The main remaining work is real-robot calibration, threshold tuning, stress testing under lighting and occlusion, and cleaning up the runtime dependency from shared framework code to lab-side modules.

