# GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation

This repository contains the implementation of **GraphLoRA**, a graph-enhanced
LoRA architecture that injects user-item collaborative graph signals into the
intermediate hidden state of LoRA adapters in large language models for
recommendation tasks.

---

## Overview

![](C:\Users\饭团子\AppData\Roaming\marktext\images\2026-06-23-23-50-59-image.png)

Existing LLM4Rec methods typically inject collaborative signals only at the
input token layer (e.g., replacing `<UserID>` / `<ItemID>` placeholders with
collaborative embeddings). However, this single-layer injection misses the
rich graph topology between users and items.

**GraphLoRA proposes to inject GNN-derived node representations into the LoRA
intermediate state `A·x`** of selected LLM attention projections, fusing
**textual semantics** (from the frozen LLM) and **collaborative graph
structure** (from a GNN over the user-item bipartite graph).

The core modification is a new `MyLinear` layer:

```
output = W·x + B · ( α · LoRA(A·x) + β · GNN(node_id) )
                    └────────────┘   └───────────────┘
                    textual path     graph path  ★ NEW
```

where `α = lora_scale`, `β = gnn_scale`, and `GNN ∈ {LightGCN, NGCF, GCN}` is selected via the `lora_gnn` config field.

---

## Getting Started

### 1. Installation

Clone the repository and create the python environment:

```bash
git clone <this-repo-url> GraphLoRA-main
cd GraphLoRA-main
conda env create -f environment.yml
conda activate minigpt4
```

### 2. Prepare the pretrained Vicuna weights

Please refer to the instruction [here](PrepareVicuna.md) to prepare the Vicuna
weights.

Then, set the path to the Vicuna weight in the `llama_model` field of the
training config file, e.g.,
[here](tran_configs/collm_pretrain_mf_ood.yaml#L15).

### 3. Prepare the Datasets

You can process the data yourself using the code provided in the `./dataset`
directory:

```
dataset/
├── ml-1m/
│   ├── processing_ood.ipynb     # MovieLens-1M preprocessing
│   └── read_info.ipynb
├── amazon_book/
│   ├── processing_ood.ipynb     # Amazon-Book preprocessing
│   └── read_info.ipynb
└── processing_ood.py             # Reference standalone script
```

After processing, the following three `.pkl` files should be placed in your
dataset root directory:

```
<DATA_DIR>/
├── train_ood2.pkl    # columns: uid, iid, label, ...
├── valid_ood2.pkl    # same schema
└── test_ood2.pkl     # same schema
```

### 4. Build Adjacency Matrix

GraphLoRA needs a normalized user-item bipartite adjacency matrix
`D^(-1/2) A D^(-1/2)` for its GNN component. Use the provided `build_graph.py`
to generate it from the `.pkl` files:

```bash
# Default (ml-1m / movie)
python build_graph.py

# Custom path / dataset
python build_graph.py --data_dir /path/to/amazon_book/ --dataset book

# Force regenerate
python build_graph.py --force
```

This produces `{dataset}_graph_{train,valid,test}.npz` files in `<DATA_DIR>/`,
which will be loaded by the model at training/evaluation time.

---

## Training

### Stage 1: Pre-train the Collaborative Filtering Model

Before injecting collaborative signals into the LLM, you need to pre-train a
collaborative filtering model (e.g., matrix factorization):

```bash
# Movie (ml-1m)
python baseline_train_mf_ood.py

# Amazon book
python baseline_train_mf_ood_amazon.py
```

The resulting `.pth` checkpoint will be referenced by the LLM stage via the
`rec_config.pretrained_path` field.

### Stage 2: GraphLoRA Tuning

To fine-tune the LLM with the pre-trained CF model, set the hyper-parameters
in the training config file
([tran_configs/collm_pretrain_mf_ood.yaml](tran_configs/collm_pretrain_mf_ood.yaml))
as follows (example for the **movie** dataset):

```yaml
model:
  arch:           mini_gpt4rec_v2
  rec_model:      MF
  dataset:        movie
  prompt_path:    "prompts/collm_movie.txt"

  freeze_rec:     False                     # train the CF model
  freeze_lora:    False                     # train LoRA
  freeze_proj:    False                     # train the projector

  llama_model:    "/path/to/vicuna-7b-v1.1"

  lora_gnn:       "ngcf"                    # GNN backbone: "lightgcn" | "ngcf" | "gcn"
  gcn_layer:      15                        # Llama layer to inject GNN into (dataset-dependent hyperparameter)
  lora_scale:     1.0
  gnn_scale:      0.1

  lora_config:
    use_lora:       True
    r:              8
    alpha:          16
    target_modules: ["q_proj", "v_proj"]
    dropout:        0.05

  rec_config:
    embedding_size:  256
    pretrained_path: "/path/to/your/mf_pretrain.pth"

  ckpt:           None

datasets:
  movie_ood:
    path:           /path/to/data/ml-1m/
    data_type:      default
    build_info:
      storage:      /path/to/data/ml-1m/

run:
  evaluate:       False
  output_dir:     /path/to/save/
```

Then run the following command:

```bash
python train_collm_mf_din.py \
    --cfg-path=tran_configs/collm_pretrain_mf_ood.yaml
```

> **Note on `gcn_layer`.** The best layer at which to inject the GNN signal is
> dataset-dependent. In our experiments, layer `15` works best on MovieLens-1M,
> while layer `31` (the final layer) works best on Amazon-Book. We recommend
> performing a small sweep over `gcn_layer ∈ {7, 15, 23, 31}` for any new
> dataset.

---

## Evaluation

Set the hyper-parameters in the training config file as follows:

```yaml
model:
  ckpt:       /path/to/your/checkpoint_best.pth   # trained model path

run:
  evaluate:   True                                # only evaluate
```

Then run the same command as the training stage:

```bash
python train_collm_mf_din.py \
    --cfg-path=tran_configs/collm_pretrain_mf_ood.yaml
```

---

## Model Architecture

The core innovation lives in `minigpt4/models/minigpt4rec_v2.py` (the
`MyLinear` and `MiniGPT4Rec_v2` classes).

```
┌─────────────────────────────────────────────────────────────────────┐
│  Input prompt with <UserID> / <TargetItemID> placeholders            │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │  Tokenize + embed (frozen) │
            └────────────┬───────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │  Replace <UserID> / <ItemID>     │
        │  token embeddings with           │
        │  llama_proj(MF embedding)        │
        └────────────┬─────────────────────┘
                     │
                     ▼
       ┌──────────────────────────────────────────┐
       │  Frozen LLaMA layers  (0..gcn_layer-1)   │
       └────────────┬─────────────────────────────┘
                    │
                    ▼  (at layer `gcn_layer`, q_proj / v_proj)
       ┌─────────────────────────────────────────────────────────┐
       │              MyLinear.forward()                          │
       │  ───────────────────────────────────────────             │
       │    base   = W · x                                        │
       │    A·x   = LoRA-A · x                                    │
       │    GNN   = GNN(sub_adj_matrix, MF_embeddings)            │
       │                                                          │
       │    A·x[user/item positions]                              │
       │        = α · A·x[...]  +  β · GNN[node_id]   ★ FUSION   │
       │                                                          │
       │    out  = base + LoRA-B · A·x                            │
       └────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
       ┌──────────────────────────────────────┐
       │  Final layer + LM head                │
       └────────────┬─────────────────────────┘
                    │
                    ▼
        Output logits for "Yes" / "No"
                    │
                    ▼
            CrossEntropy loss
```

---

## Citation

Citation information will be added here once the paper is officially published.

---

## Acknowledgement

+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and
  [CoLLM](https://github.com/zyang1580/CoLLM). Our repository is built upon
  MiniGPT-4 and CoLLM. We thank the authors for their wonderful work.
+ [Vicuna](https://github.com/lm-sys/FastChat). The fantastic language ability
  of Vicuna with only 7B parameters is just amazing. And it is open-source!
+ [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch) and
  [NGCF](https://github.com/xiangwang1223/neural_graph_collaborative_filtering).
  The GNN modules in `rec_base_models.py` follow their official implementations.
