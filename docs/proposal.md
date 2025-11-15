# EE 6770: Applications of Neural Networks (Fall 2025)

## Final Project Proposal

### Human–Robot Distance and DOA Classification from mmWave MIMO Radar Using CNNs and Federated Learning

- Team Member 1: Hossein Molhem
- Team Member 2: (TBD)
- Electrical and Computer Engineering, Kennesaw State University
- Faculty: Dr. Jeffrey L. Yiin

---

## 1. Problem Statement and Project Objectives

Collaborative robots (cobots) increasingly share workspace with human operators on factory floors. To guarantee safety and efficiency, the robot controller must continuously know where the human is — both distance from the robot and direction of arrival (DOA). Conventional solutions often rely on cameras or wearable sensors, which can be intrusive, sensitive to lighting, or raise privacy concerns.

mmWave MIMO radar offers a contactless and privacy‑preserving alternative. The IEEE DataPort dataset "Federated Learning: mmWave MIMO Radar Dataset for Testing" provides range–azimuth maps collected by TI IWR1843BOOST radars around a robotic manipulator, together with labels indicating which of ten human positions (including "no human") is present at each time. The original dataset was created to benchmark federated learning (FL) algorithms for industrial human–robot monitoring and is accompanied by an official federated learning software package on GitHub: <https://github.com/labRadioVision/federated>

In this project, we use this dataset to train neural networks that classify the human operator’s region of interest (ROI) from radar range–azimuth maps. The task is a 10‑class classification problem where each class corresponds to distance and azimuth intervals relative to the robot, including the empty workspace (class 0).

We do not propose a new federated learning algorithm. Instead, we implement and compare centralized and federated training of a convolutional neural network (CNN) for 10‑class human–robot position classification, and extend the analysis with: (i) a detailed, per‑class and safety‑aware evaluation that emphasizes critical misclassifications (e.g., predicting an empty workspace when the operator is actually near the robot), and (ii) a study of robustness across different collection days by training on one day and testing on others. This combination of reproduction, comparison, and extended evaluation is appropriate for a course project and provides practical insight into how federated learning behaves on a real radar‑based human–robot collaboration scenario.

This topic fits the course because it combines neural networks with radar signal processing, and naturally supports both centralized and federated training.

### Project Objectives

- Implement a compact CNN that takes a $256 \times 63$ range–azimuth map as input and outputs a 10‑class prediction for the human operator’s ROI (including empty workspace).
- Train and evaluate this CNN in a centralized setting, where all device data for a given day are merged into a single training set.
- Train and evaluate the same CNN in a federated setting, where each device is a separate client with local data, and a global model is learned using a FedAvg‑style aggregation of local updates.
- Compare centralized and federated training in terms of overall accuracy, per‑class performance, and confusion matrices, with special attention to safety‑critical errors (e.g., near‑robot classes predicted as empty).
- Investigate robustness across days by training on one day and testing on the others, and discuss how environmental or distribution shifts affect performance.

### Existing Solutions and Baselines

- The `labRadioVision/federated` GitHub repository provides a general Python/TensorFlow package for federated learning experiments based on this dataset.
- Published work by Savazzi, Nicoli, Bennis, and others analyzes FL in connected, cooperative, and automated industrial systems (communication overhead, non‑IID data, robustness).

These works emphasize federated optimization and system aspects. They provide limited exploration of different CNN architectures on this specific radar dataset and do not deeply analyze safety‑related misclassifications (e.g., confusing near‑robot with empty workspace).

### Why This Topic

- Combines neural networks and radar signal processing.
- Data are preprocessed into range–azimuth maps, enabling focus on CNN design, training, and evaluation.
- Naturally supports both centralized and federated training, allowing analysis under realistic partitions (devices, days).

### Planned Contributions and Novelty

- Design and compare CNN architectures:
  - A compact baseline CNN for 10‑class classification on $256 \times 63$ maps.
  - A slightly deeper/wider CNN (more conv layers, batch norm) to study accuracy–complexity trade‑offs for edge deployment.
- Study centralized vs. federated training:
  - Centralized: all training data merged.
  - Federated: multiple clients (devices/days), FedAvg‑style aggregation; compare final test accuracy and convergence.
- Analyze spatial performance: confusion matrices and per‑class accuracy to find hard distance/angle regions and FL differences (close vs. far, left vs. right).
- Safety‑aware analysis (time permitting): group classes into near/mid/far/empty and highlight critical errors (e.g., predicting empty when operator is near).

Our objective is a clean, reproducible pipeline, competitive classification, and a clear centralized vs. federated comparison.

---

## 2. Approach and Methodology

### Dataset

**Source and Link.**

- Title: Federated Learning: mmWave MIMO Radar Dataset for Testing (Savazzi et al., IEEE DataPort)
- Access: MATLAB `.mat` files on IEEE DataPort; official FL software: <https://github.com/labRadioVision/federated>

**Content and Size.** Organized by day and device:

- Three training days: day 0, day 1, day 2
- For each day i, training data contain 9 device files: `mmwave_data_train_k.mat`, k=1..9

Inside each training file:

- `mmwave_data_train_k`: approx. shape $N_k \times 256 \times 63$ (range–azimuth maps)
  - 256 range bins (~0.5 m to 11 m)
  - 63 azimuth bins (~−75° to +75°)
- `label_train_k`: integer labels in {0..9}

Test data per day:

- `mmwave_data_test`: approx. $N_{test} \times 256 \times 63`
- `label_test`: integer labels in {0..9}

**Classes and Labels.**

- Class 0: empty workspace (operator outside monitored area)
- Classes 1–9: operator in specific distance/azimuth regions (examples: class 1 ~0.5–0.7 m, +40°–+60°; class 2 ~0.3–0.5 m, −10°–+10°; …; class 9 ~1.2–1.6 m, −20° to −10°)

**Examples.** (for final report)

- One example map for class 0 (empty)
- One for a close class (e.g., class 2) and one for a far class (e.g., class 7)

### Input Representation

Each snapshot is a $256 \times 63$ 2D array:

- Convert to magnitude or log‑magnitude (consistent with dataset convention)
- Normalize per snapshot or via global min–max
- Treat as a single‑channel image of shape `(256, 63, 1)`

### Model Architecture

**Baseline CNN.**

- Input: `(256, 63, 1)`
- Conv Block 1: Conv2D(1→16, 3×3, same) + ReLU + MaxPool 2×2
- Conv Block 2: Conv2D(16→32, 3×3, same) + ReLU + MaxPool 2×2
- Flatten
- Dense 128 + ReLU + Dropout
- Output: Dense 10 + Softmax

**Improved CNN.**

- Add a third conv block with batch normalization
- Adjust filters and kernel sizes (e.g., larger along range) for radar structure

### Centralized vs. Federated Training

**Centralized.** Merge all training samples for a day and train a single CNN with Adam, using a train/validation split for early stopping and tuning. Evaluate on the day’s official test set.

**Federated.** Keep data partitioned by device. In each round, server broadcasts global model; clients train locally for a few epochs and return weights; server aggregates via weighted averaging (FedAvg). Evaluate the final global model on the same test set(s).

### Techniques and Tools

- Frameworks: Python with TensorFlow/Keras (or PyTorch if needed)
- Data loading: SciPy to read `.mat` files, NumPy for preprocessing
- Training: train/val split, early stopping, LR scheduling (ReduceLROnPlateau), dropout, weight decay
- Federated learning: Simplified FedAvg setup inspired by `labRadioVision/federated`

---

## 3. Evaluation Plan

### Metrics

- Overall test accuracy (top‑1)
- Macro‑averaged precision, recall, F1 across 10 classes
- Per‑class accuracy
- Confusion matrices

Optional: safety‑aware grouping (near/mid/far/empty) to highlight critical errors.

### Baseline Comparison

- Trivial baselines: random guess; majority class
- Simple ML baseline: logistic regression or shallow MLP on flattened maps
- CNN baselines: baseline vs. improved CNN (centralized)
- Federated vs. centralized: compare for at least one CNN

### Validation Strategy

- Use official `mmwave_data_test`/`label_test` as held‑out test sets
- Inside training data, reserve 10–20% for validation
- Explore cross‑day generalization: train on day 0, test on days 1 and 2

We will present:

- Tables of accuracy, precision, recall, F1
- Confusion matrices for best model(s)
- Training/validation curves (or FL round metrics)

---

## 4. References

1. S. Savazzi, "Federated Learning: mmWave MIMO Radar Dataset for Testing," IEEE DataPort, 2021.
2. S. Savazzi, M. Nicoli, M. Bennis, S. Kianoush, L. Barbieri, "Opportunities of Federated Learning in Connected, Cooperative, and Automated Industrial Systems," IEEE Communications Magazine, vol. 59, no. 2, pp. 16–21, 2021.
3. labRadioVision/federated GitHub repository: "Federated Learning package," <https://github.com/labRadioVision/federated>
