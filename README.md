<br/>
<p align="center">
  <h1 align="center">3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning</h1>
  <p align="center">
    CVPR 2025
  </p>
  <p align="center">
    <a href="https://yyuncong.github.io/">Yuncong Yang</a>,
    <a href="https://hanyangclarence.github.io/">Han Yang</a>,
    <a href="https://www.linkedin.com/in/jiachen-zhou5/">Jiachen Zhou</a>,
    <a href="https://peihaochen.github.io/">Peihao Chen</a>,
    <a href="https://icefoxzhx.github.io/">Hongxin Zhang</a>,
    <a href="https://yilundu.github.io/">Yilun Du</a>,
    <a href="https://people.csail.mit.edu/ganchuang">Chuang Gan</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2411.17735">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://embodied-agi.cs.umass.edu/3dmem/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
</p>

---

This is the official repository of **3D-Mem**: 3D Scene Memory for Embodied Exploration and Reasoning.

![](assets/teaser.png)

---

## News

- [2025/03] Inference code for A-EQA and GOAT-Bench is released.
- [2025/02] 3D-Mem is accepted to CVPR 2025!
- [2024/12] [Paper](https://www.arxiv.org/abs/2411.17735) is on arXiv.

## Installation
Set up the conda environment (Linux, Python 3.9):
```bash
conda create -n 3dmem python=3.9 -y && conda activate 3dmem

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge -c aihabitat habitat-sim=0.2.5 headless faiss-cpu=1.7.4 -y
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2 -y

pip install omegaconf==2.3.0 open-clip-torch==2.26.1 ultralytics==8.2.31 supervision==0.21.0 opencv-python-headless==4.10.* \
 scikit-learn==1.4 scikit-image==0.22 open3d==0.18.0 hipart==1.0.4 openai==1.35.3 httpx==0.27.2                                                      

```


## Run Evaluation

### 1 - Preparations

#### Dataset
Please download the train and val split of [HM3D](https://aihabitat.org/datasets/hm3d-semantics/), and specify
the path in `cfg/eval_aeqa.yaml` and `cfg/eval_goatbench.yaml`. For example, if your download path is `/your_path/hm3d/` that 
contains `/your_path/hm3d/train/` and `/your_path/hm3d/val/`, you can set the `scene_data_path` in the config files as `/your_path/hm3d/`.

The test questions of A-EQA and GOAT-Bench are provided in the `data/` folder. For A-EQA, since some questions are
not suitable for embodied QA in our settings, we provide two subsets of different size: `aeqa_questions-41.json` and `aeqa_questions-184.json`.
For GOAT-Bench, we include the complete `val_unseen` split in this repository.

#### OpenAI API Setup
Please set up the endpoint and API key for the OpenAI API in `src/const.py`.

### 2 - Run Evaluation on A-EQA

First run the following script to generate the predictions for the A-EQA dataset:

```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml
```
To split tasks, you can add `--start_ratio` and `--end_ratio` to specify the range of tasks to evaluate. For example,
to evaluate the first half of the dataset, you can run:
```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml --start_ratio 0.0 --end_ratio 0.5
```
After the scripts finish, the results from all splits will be automatically aggregated and saved.

To evaluate the predictions with the pipeline from OpenEQA, you can refer to [link](https://github.com/yyuncong/3D-Mem-AEQA-Eval)

### 3 - Run Evaluation on GOAT-Bench
You can directly run the following script:
```bash
python run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml
```
The results will be saved and printed after the script finishes. You can also split the task similarly by adding `--start_ratio` and `--end_ratio`.
Note that GOAT-Bench provides 10 explore episodes for each scene, and by default we only test the first episode due to the time and resource constraints.
You can also specify the episode to evaluate for each scene by setting `--split`.

### 4 - Save Visualization
The default evaluation config will save visualization results including topdown maps, egocentric views, memory snapshots, and frontier snapshots at each step. Although saving visualization is very helpful, it may slows down the evaluation process. Please make save_visualization false if you would like to run large-scale evaluation.


## Acknowledgement

The codebase is built upon [OpenEQA](https://github.com/facebookresearch/open-eqa), [Explore-EQA](https://github.com/Stanford-ILIAD/explore-eqa), and [ConceptGraph](https://github.com/concept-graphs/concept-graphs).
We thank the authors for their great work.

## Citing 3D-Mem

```tex
@misc{yang20243dmem3dscenememory,
      title={3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning}, 
      author={Yuncong Yang and Han Yang and Jiachen Zhou and Peihao Chen and Hongxin Zhang and Yilun Du and Chuang Gan},
      year={2024},
      eprint={2411.17735},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17735}, 
}
```
