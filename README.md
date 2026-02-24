# SUMO RL Ego

A modular reinforcement learning framework for training and evaluating ego-vehicle policies in SUMO using Gymnasium-style environments and Stable-Baselines3.

This repository provides a configurable and research-oriented foundation for RL experimentation in traffic simulation, combining a modular environment, config-driven pipelines, and built-in benchmarks.

Core features:
- A reusable Gymnasium wrapper around SUMO
- Modular components for observations, rewards, metrics, and ego logic (plugin-based)
- Training and evaluation pipelines with YAML-driven experiments
- Built-in benchmark scenarios for reproducible comparisons
- Unified interface for RL and handcrafted policies

---

🎯 Motivation

The SUMO ecosystem for reinforcement learning is still fragmented and largely outdated. Many existing repositories rely on deprecated Gym integrations, are no longer maintained, or are tightly coupled research codebases that are hard to reuse or extend.

In particular, there is no widely adopted, modern framework that provides:
- a clean Gymnasium-compatible interface for SUMO
- fast iteration for ego-vehicle policy development
- modular, reusable experimentation pipelines

Additionally, the field lacks standardized benchmarks for ego-vehicle autonomy in SUMO. Most works rely on custom, non-reproducible scenarios, making comparisons across methods difficult.

This repository aims to provide a modern, lightweight, and research-friendly foundation for reproducible ego-vehicle RL experimentation in SUMO.

---

## ✨ Contributions

### 1. Modular Gymnasium Wrapper for SUMO

A clean and extensible Gymnasium-compatible wrapper around SUMO designed for **plug-and-play experimentation**.

The environment follows a modular architecture where key components are implemented as plugins:

* observation builder
* reward function
* ego-vehicle logic
* metrics

Users can extend or replace components without modifying the core environment, enabling fast iteration and reusable research modules.

---

### 2. Lightweight RL Experimentation Framework

A structured pipeline to support the full experimentation lifecycle:

* training
* evaluation
* GUI debugging / rollout

Environment and policy creation are handled through reusable builders, reducing boilerplate and enforcing consistent experiment structure.

---

### 3. YAML-Driven Configuration System

All experiments are fully defined through YAML configuration files.

A single config specifies:

* scenarios
* environment plugins
* rl policy type
* training hyperparameters


---

### 4. Built-in SUMO Benchmark Scenarios

The repository includes curated SUMO scenarios that act as **standardized benchmarks**, such as:

* highway driving
* intersections
* roundabouts

These scenarios allow:

* fair comparison between policies
* evaluation of generalization
* reproducible baselines across experiments

They can be used as a benchmark suite for RL research in traffic simulation.

---

### 5. Unified Interface for RL and Handcrafted Policies

The framework supports both:

* reinforcement learning agents
* manually designed policies

All policies implement a common interface, enabling:

* direct RL vs rule-based comparisons
* strong baselines
* hybrid approaches (heuristics + learning)

---

## 🗂 Repository Structure

```text
src/
├─ sumo_rl_ego/
│  ├─ env/            # Core Gymnasium SUMO wrapper
│  ├─ ego/            # Ego-vehicle interfaces and implementations
│  ├─ observation/    # Observation spaces and feature builders
│  ├─ reward/         # Reward functions and shaping logic
│  └─ metrics/        # Metrics tracking and evaluation utilities
│
├─ infra/
│  ├─ builders/       # Environment and policy builders
│  ├─ trainer/        # Training loops and orchestration
│  ├─ loaders/        # Config and class loading utilities
│  ├─ utils/          # Shared helpers and abstractions
│  └─ policy/         # Unified interface for RL and handcrafted policies
│
experiments/
├─ configs/           # YAML experiment definitions
├─ train.py           # Training entry point
├─ eval.py            # Evaluation entry point
└─ debug_gui.py       # GUI rollout and inspection
│
scenarios/
└─ ...                # Pre-built SUMO benchmark scenarios
│
plugins/
├─ observation/       # Custom observation plugins
├─ reward/            # Custom reward plugins
├─ ego/               # Custom ego logic implementations
├─ policy/            # Custom handcrafted policies
└─ metrics/           # Custom evaluation metrics
│
tests/
└─ ...                # Unit and integration tests       


```

The structure separates:

* **core environment** (stable)
* **plugins** (experimental components)
* **experiments** (entry points + configs)

This encourages clean research workflows and easier reproducibility.

---

## 🚀 Typical Research Workflow

1. Select a benchmark SUMO scenario
2. Define experiment via YAML config
3. Choose RL algorithm or handcrafted policy
4. Train and evaluate across scenarios
5. Compare results using shared metrics

---

## 📚 Academic Use

This framework is suitable for:

* RL research in autonomous driving
* benchmarking policy generalization
* ablation studies on rewards/observations
* RL vs classical control comparisons
* reproducible experimental sections in papers

---

## 🛠 Use Cases

* Reinforcement learning experimentation on traffic behaviors
* Rapid prototyping of environment designs
* Benchmarking RL algorithms across scenarios
* Comparing learned and rule-based policies

---

## 📌 Notes

The project intentionally favors clarity and modularity over heavy abstractions. It is designed as a strong baseline for research projects and as a foundation for more advanced SUMO-based RL environments.
