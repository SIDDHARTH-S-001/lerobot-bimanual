```markdown
# LeRobot Dataset Structure Analysis

**Dataset:** `pranavnaik98/bimanual_so101_stacking_100`

This document provides a **fact-based, evidence-backed analysis** of the LeRobot dataset structure. All observations are derived directly from inspection of the dataset’s parquet files and associated video assets.

---

## 📊 Dataset Overview

- **Total Episodes:** 8  
- **Total Frames:** 4,800 (600 frames per episode)  
- **Task:** *Bimanual box stacking*  
- **FPS:** 30  
  - Inferred from timestamps with ~0.033333s intervals  
- **Cameras:** 2  
  - `observation.images.top`  
  - `observation.images.front`  

---

## 📁 File Structure

```

data/pranavnaik98/bimanual_so101_stacking_100/
├── data/
│   └── chunk-000/
│       └── file-000.parquet       # Frame-level data (4,800 rows)
├── meta/
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet   # Episode-level metadata (8 rows)
└── videos/
└── observation.images.{top,front}/
└── chunk-000/
└── file-{000,001}.mp4 # Video files

````

---

## 🎬 1. Frame-Level Data  
**Path:** `data/chunk-000/file-000.parquet`  

- **Shape:** `(4800, 7)`  
- **Description:** Contains observations and actions for every frame across all episodes.

### Columns

#### `action`
- **dtype:** `object` (stored as `np.ndarray`)
- **Shape per row:** `(12,)`
- **Type:** `float32`
- **Meaning:** Commanded robot action at this timestep  

**Evidence:**
```python
[-4.972875, -79.25554, 89.13336, 63.088673, 5.4456654,
 0.23291926, 0.08628128, -100.0, 98.5872, 73.28909,
 9.98779, 1.4820592]
````

**Interpretation:**

* 12 dimensions = 6 DOF × 2 arms
* Likely format:
  `[arm1_joint1, ..., arm1_joint6, arm2_joint1, ..., arm2_joint6]`

---

#### `observation.state`

* **dtype:** `object` (`np.ndarray`)
* **Shape per row:** `(12,)`
* **Type:** `float32`
* **Meaning:** Observed robot joint state at this timestep

**Evidence:**

```python
[0.3270646, -91.09076, 99.64061, 75.605606, -3.003663,
 1.1881188, 0.0923361, -99.071335, 97.96352, 76.317924,
 -1.6849817, 1.8452381]
```

**Interpretation:**
Same 12-dimensional structure as `action` (joint angles, likely in degrees).

---

#### `timestamp`

* **dtype:** `float32`
* **Meaning:** Time elapsed since episode start (seconds)

**Evidence:**
`[0.0, 0.033333, 0.066667, 0.1, 0.133333]`

**Interpretation:**
~0.033s increments → **30 FPS**

---

#### `frame_index`

* **dtype:** `int64`
* **Meaning:** Frame number within the episode (0-indexed)
* **Range:** 0–599

---

#### `episode_index`

* **dtype:** `int64`
* **Meaning:** Episode ID for this frame
* **Range:** 0–7

---

#### `index`

* **dtype:** `int64`
* **Meaning:** Global frame index across dataset
* **Range:** 0–4799

---

#### `task_index`

* **dtype:** `int64`
* **Meaning:** Task identifier
* **Values:** All `0`

**Interpretation:**
Single-task dataset (*Bimanual box stacking*).

---

## 📋 2. Episode Metadata

**Path:** `meta/episodes/chunk-000/file-000.parquet`

* **Shape:** `(8, 107)`
* **Description:** One row per episode with metadata, video pointers, and statistics.

### Base Metadata Columns

#### `episode_index`

* Values: `[0, 1, 2, 3, 4, 5, 6, 7]`
* Meaning: Unique episode identifier

#### `tasks`

* Type: `list[str]`
* Values: `['Bimanual box stacking']`
* Meaning: Human-readable task description

#### `length`

* Values: `600`
* Meaning: Frames per episode

#### `dataset_from_index`

* Values: `[0, 600, 1200, 1800, 2400, 3000, 3600, 4200]`
* Meaning: Starting global frame index

#### `dataset_to_index`

* Values: `[600, 1200, 1800, 2400, 3000, 3600, 4200, 4800]`
* Meaning: Ending global frame index (exclusive)

---

### Data File Pointers

* `data/chunk_index`: `0`
* `data/file_index`: `0`

All episodes reside in the same data file.

---

### Video Metadata (per camera)

For each camera (`observation.images.top`, `observation.images.front`):

* `videos/{camera}/chunk_index`
* `videos/{camera}/file_index`
* `videos/{camera}/from_timestamp`
* `videos/{camera}/to_timestamp`

**Evidence (top camera, from_timestamp):**

```
[0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 0.0]
```

**Interpretation:**

* Each episode = **20 seconds** (600 frames ÷ 30 FPS)
* Episodes 0–6 share one video file
* Episode 7 starts a new video file (`file_index = 1`)

---

### Statistics Columns (90 total)

For each feature (`action`, `observation.state`, images, timestamps, etc.):

* `stats/{feature}/min`
* `stats/{feature}/max`
* `stats/{feature}/mean`
* `stats/{feature}/std`
* `stats/{feature}/count`
* `stats/{feature}/q01`
* `stats/{feature}/q10`
* `stats/{feature}/q50`
* `stats/{feature}/q90`
* `stats/{feature}/q99`

**Example (action, episode 0):**

```python
min:  [-17.2694397, -100.0, -45.75662231, ...]
max:  [25.13562393, 9.66123009, 93.26448059, ...]
count: [600]
```

**Purpose:**

* Episode-specific normalization
* Outlier detection
* Robust statistics via quantiles

---

## 🔍 Key Insights

1. **Bimanual Robot Configuration**

   * 12D action/state → 2 arms × 6 joints
   * Both arms controlled simultaneously

2. **Video Organization**

   * Multiple episodes per video file
   * Timestamp ranges specify seek locations
   * Episode 7 begins a new video file (top camera)

3. **Statistics Utility**

   * Enables normalization and anomaly detection
   * Quantiles (`q01`, `q99`) useful for robust filtering

4. **Data Efficiency**

   * State/action in parquet, videos stored separately
   * Columnar storage + chunking supports scalability

5. **Frame Indexing Semantics**

   * `index`: global frame ID (0–4799)
   * `frame_index`: within-episode (0–599)
   * `episode_index`: episode selector (0–7)

---

## 📈 Data Flow Example

**Goal:** Load episode 3, frame 100

1. Read episode metadata (row 3)
2. Global frame index:
   `dataset_from_index[3] + 100 = 1800 + 100 = 1900`
3. Load row 1900 from:
   `data/chunk-000/file-000.parquet`
4. Extract:

   * `action[1900]` → 12D array
   * `observation.state[1900]` → 12D array
5. Load video frame:

   * Camera: `observation.images.top`
   * File: `videos/observation.images.top/chunk-000/file-000.mp4`
   * Timestamp:
     `from_timestamp[3] + (100 / 30) = 60.0 + 3.33 = 63.33s`

---

## ✅ Validation

All findings were directly extracted from the dataset:

* Column names: verified via `df.columns`
* Data types: verified via `df.dtypes`
* Shapes: verified via `array.shape`
* Values: verified via `df.head()` and `df.iloc[...]`