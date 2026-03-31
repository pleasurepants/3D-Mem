# ReExplore: Learning from Past Explorations for Embodied Question Answering

This repository contains the code for **ReExplore**, a framework that enables embodied agents to learn from past exploration experiences to improve future embodied question answering (EQA) performance.

---

## Repository Structure

```
.
├── src/                        # Core source code
│   ├── const.py                # API configuration constants
│   ├── utils.py                # General utilities (image processing, geometry)
│   ├── geom.py                 # Camera intrinsics, scene bounds, geometry/IoU
│   ├── habitat.py              # Habitat-sim scene setup, quaternions, navigability
│   ├── tsdf_base.py            # Volumetric TSDF fusion base class
│   ├── tsdf_planner.py         # TSDF planner with frontier-based exploration
│   ├── tsdf_planner_hdbscan.py # Two-level hierarchical frontier grouping (HDBSCAN)
│   ├── hierarchy_clustering.py # Hierarchical clustering helpers
│   ├── scene_aeqa.py           # A-EQA Scene: environment, detection, object map, snapshots
│   ├── scene_goatbench.py      # GOAT-Bench Scene
│   ├── context_generator.py    # Experience retrieval and replay context generation
│   ├── query_vlm_aeqa*.py      # VLM query interfaces for A-EQA (multiple backends)
│   ├── eval_utils_gpt_aeqa*.py # Prompt assembly, retrieval, experience replay
│   ├── logger_aeqa.py          # Logging and metrics for A-EQA
│   ├── logger_goatbench.py     # Logging and metrics for GOAT-Bench
│   └── conceptgraph/           # ConceptGraph SLAM and utilities (adapted)
├── eval/                       # Evaluation entry points
│   ├── run_aeqa_evaluation.py          # A-EQA evaluation (default)
│   ├── run_aeqa_evaluation_qwen.py     # A-EQA with Qwen-VL backend
│   ├── run_aeqa_evaluation_gpt.py      # A-EQA with GPT backend
│   ├── run_aeqa_evaluation_internvl.py # A-EQA with InternVL backend
│   ├── run_aeqa_evaluation_glm.py      # A-EQA with GLM backend
│   └── run_goatbench_evaluation.py     # GOAT-Bench evaluation
├── experience_set/             # Experience set construction
│   ├── build_retrieve_store.py # Build frontier image + question retrieval store
│   └── build_question_store.py # Build question-only vector store (SBERT)
├── abstraction_scripts/        # Abstraction generation utilities
│   ├── extract_two_layer_reasons.py    # Extract two-layer reasoning from logs
│   ├── gen_experience_scripts.py       # Generate experience replay experiment scripts
│   └── gen_exp_at_scripts.py           # Generate experiment scripts with stage control
├── data/                       # Dataset configurations and questions
│   ├── aeqa_questions-*.json   # A-EQA question sets (41, 168, 184 subsets)
│   └── goat_bench/             # GOAT-Bench val_unseen split
├── cfg/                        # Configuration files
│   ├── eval_aeqa.yaml          # A-EQA evaluation config
│   ├── eval_goatbench.yaml     # GOAT-Bench evaluation config
│   └── concept_graph_default.yaml # ConceptGraph default config
└── README.md
```

## Installation

Set up the conda environment (Linux, Python 3.9):

```bash
conda create -n reexplore python=3.9 -y && conda activate reexplore

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge -c aihabitat habitat-sim=0.2.5 headless faiss-cpu=1.7.4 -y
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2 -y

pip install omegaconf==2.3.0 open-clip-torch==2.26.1 ultralytics==8.2.31 supervision==0.21.0 \
    opencv-python-headless==4.10.* scikit-learn==1.4 scikit-image==0.22 open3d==0.18.0 \
    hipart==1.0.4 openai==1.35.3 httpx==0.27.2 sentence-transformers hdbscan
```

## Dataset Preparation

1. Download the train and val split of [HM3D](https://aihabitat.org/datasets/hm3d-semantics/).
2. Specify the path in `cfg/eval_aeqa.yaml` and `cfg/eval_goatbench.yaml` under `scene_data_path`.

## API Setup

Set up the endpoint and API key for the OpenAI-compatible API in `src/const.py`.

## Run Evaluation

### A-EQA Evaluation

```bash
python eval/run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml
```

To split tasks across multiple runs:
```bash
python eval/run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml --start_ratio 0.0 --end_ratio 0.5
```

### A-EQA with Experience Replay

```bash
python eval/run_aeqa_evaluation_qwen.py -cf cfg/eval_aeqa.yaml \
    --caption true --critique true --abstraction true \
    --exp_tuple /path/to/exp_tuple.json \
    --retrieve_root /path/to/retrieve_store
```

### GOAT-Bench Evaluation

```bash
python eval/run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml
```

## Building Experience Sets

To build the retrieval store from exploration trajectories:

```bash
python experience_set/build_retrieve_store.py \
    --src_root /path/to/exploration_output \
    --dst_root /path/to/retrieve_store \
    --questions_path data/aeqa_questions-168.json
```

To build the question-only vector store:

```bash
python experience_set/build_question_store.py \
    --questions_path data/aeqa_questions-168.json \
    --dst_root /path/to/retrieve_store
```

## Acknowledgement

The codebase is built upon [OpenEQA](https://github.com/facebookresearch/open-eqa), [Explore-EQA](https://github.com/Stanford-ILIAD/explore-eqa), and [ConceptGraph](https://github.com/concept-graphs/concept-graphs).
