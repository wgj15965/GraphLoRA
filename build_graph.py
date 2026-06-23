#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_graph.py — GraphLoRA 邻接矩阵自动生成工具
================================================

从 {train,valid,test}_ood2.pkl 自动构造用户-商品二部图的归一化邻接矩阵
(D^(-1/2) A D^(-1/2))，输出模型推理 / 训练所需的 npz 文件。

复用现有 GnnDataset (minigpt4/datasets/datasets/rec_gnndataset.py) 内的
getSparseGraph_mode_a2() 算法，与 LightGCN/NGCF 论文实现一致。

------------------------------------------------------------------------
✦ 背景
   minigpt4/models/minigpt4rec_v2.py::_load_adjacency_matrix() 要求加载
   3 个文件（按 train/valid/test split）：
     ├─ {dataset}_graph_train.npz
     ├─ {dataset}_graph_valid.npz
     └─ {dataset}_graph_test.npz
   但仓库未附带这些文件，也无生成脚本。本工具填补该空缺。

✦ 设计说明
   1. 实际运行时只有 mode='train' 的 npz 会被加载（dataset __getitem__
      不输出 is_valid/is_test 标志，验证已确认）。
   2. 为保证模型代码 robust，本脚本仍生成 3 份文件（同内容复制）。
   3. GnnDataset 末尾会 .cuda()，若环境无 GPU 会抛 RuntimeError，
      但 sp.save_npz() 在 .cuda() 之前已完成，所以即便报错 npz 已保存。

------------------------------------------------------------------------
✦ 用法

    # 默认配置（ml-1m 数据，dataset=movie）
    python build_graph.py

    # 自定义路径 / 数据集类型
    python build_graph.py --data_dir /data2/guoji_wang/datasets/ml-1m/ \
                          --dataset  movie

    # 处理 Amazon book
    python build_graph.py --data_dir /data2/guoji_wang/datasets/amazon_book/ \
                          --dataset  book

    # 强制重新生成（覆盖旧 npz）
    python build_graph.py --force

------------------------------------------------------------------------
作者: guoji_wang
日期: 2026-05-26
"""

import os
import sys
import shutil
import argparse
import scipy.sparse as sp

# 确保能从项目根目录 import minigpt4.*
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraphLoRA 邻接矩阵生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="详见脚本顶部 docstring。"
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='/data2/guoji_wang/datasets/ml-1m/',
        help='含 train_ood2.pkl/valid_ood2.pkl/test_ood2.pkl 的目录（默认: ml-1m）'
    )
    parser.add_argument(
        '--dataset', type=str, default='movie',
        choices=['movie', 'book'],
        help='数据集类型（决定输出文件名前缀，默认: movie）'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='强制重新生成（删除所有已有 npz）'
    )
    parser.add_argument(
        '--keep_intermediate', action='store_true',
        help='保留中间文件 s_pre_adj_mat_graph.npz（默认会删除）'
    )
    return parser.parse_args()


def check_inputs(data_dir):
    """检查 3 个 pkl 输入是否存在"""
    missing = []
    for split in ('train', 'valid', 'test'):
        path = os.path.join(data_dir, f'{split}_ood2.pkl')
        if not os.path.exists(path):
            missing.append(path)
        else:
            size_kb = os.path.getsize(path) / 1024
            print(f"  ✓ {split}_ood2.pkl  ({size_kb:>8.1f} KB)")
    if missing:
        print("\n❌ 缺失以下输入文件：")
        for p in missing:
            print(f"   - {p}")
        print("\n👉 请先跑 dataset/ml-1m/processing_ood.ipynb 生成这些 pkl 文件。")
        sys.exit(1)


def remove_if_exists(path, label):
    if os.path.exists(path):
        os.remove(path)
        print(f"  🗑  已删除旧 {label}: {os.path.basename(path)}")


def build_via_gnndataset(data_dir, dataset_name):
    """实例化 GnnDataset 触发邻接矩阵构造"""
    print(f"\n[2/4] 实例化 GnnDataset 构造邻接矩阵...")
    print(f"      算法: D^(-1/2) · A · D^(-1/2)  (标准 LightGCN 归一化)")
    print(f"      节点: m_users + n_items  二部图")
    print(f"      边:   仅采用 train 集 label>0 的正样本")
    print()

    import omegaconf
    config = omegaconf.OmegaConf.create({
        'A_split':  False,
        'A_n_fold': 1,
        'dataset':  dataset_name,
    })

    from minigpt4.datasets.datasets.rec_gnndataset import GnnDataset

    try:
        _ = GnnDataset(config, path=data_dir)
    except RuntimeError as e:
        # GnnDataset.__init__ 末尾会调 .cuda()，无 GPU 会抛 RuntimeError，
        # 但此时 sp.save_npz 已经执行完，文件已落盘
        msg = str(e).lower()
        if 'cuda' in msg or 'gpu' in msg or 'no kernel image' in msg:
            print(f"\n  ⚠️  捕获 CUDA 错误（{type(e).__name__}），但 npz 已保存，忽略。")
        else:
            raise


def verify_output(intermediate_path):
    """读回 npz 验证并打印图统计"""
    print(f"\n[3/4] 验证生成的邻接矩阵...")
    if not os.path.exists(intermediate_path):
        print(f"❌ 未找到中间文件: {intermediate_path}")
        sys.exit(1)

    norm_adj = sp.load_npz(intermediate_path)
    size_kb = os.path.getsize(intermediate_path) / 1024
    sparsity = norm_adj.nnz / (norm_adj.shape[0] * norm_adj.shape[1])

    print(f"  ✓ 形状     : {norm_adj.shape}")
    print(f"  ✓ 非零元素 : {norm_adj.nnz:,}")
    print(f"  ✓ 稀疏度   : {sparsity:.6f}  ({100*sparsity:.4f}%)")
    print(f"  ✓ 文件大小 : {size_kb:.1f} KB")
    return norm_adj


def fan_out_to_splits(intermediate_path, data_dir, dataset_name):
    """把中间文件复制为模型期望的 {dataset}_graph_{split}.npz"""
    print(f"\n[4/4] 重命名为模型加载所需格式...")
    targets = []
    for split in ('train', 'valid', 'test'):
        target = os.path.join(data_dir, f'{dataset_name}_graph_{split}.npz')
        shutil.copy(intermediate_path, target)
        size_kb = os.path.getsize(target) / 1024
        print(f"  ✓ {dataset_name}_graph_{split}.npz  ({size_kb:.1f} KB)")
        targets.append(target)
    return targets


def main():
    args = parse_args()

    # 标准化路径
    if not args.data_dir.endswith('/'):
        args.data_dir += '/'

    print("=" * 70)
    print("  GraphLoRA 邻接矩阵生成 (build_graph.py)")
    print("=" * 70)
    print(f"  data_dir : {args.data_dir}")
    print(f"  dataset  : {args.dataset}")
    print(f"  force    : {args.force}")
    print()

    # ---------- [1/4] 输入检查 + 旧文件清理 ----------
    print("[1/4] 检查输入文件...")
    check_inputs(args.data_dir)

    intermediate = os.path.join(args.data_dir, 's_pre_adj_mat_graph.npz')
    target_files = [
        os.path.join(args.data_dir, f'{args.dataset}_graph_{s}.npz')
        for s in ('train', 'valid', 'test')
    ]

    if args.force:
        print()
        remove_if_exists(intermediate, "中间文件")
        for tf in target_files:
            remove_if_exists(tf, "目标文件")

    # 如果所有目标文件已存在且未 --force，直接退出
    if (not args.force) and all(os.path.exists(tf) for tf in target_files):
        print("\n⚠️  所有目标文件已存在，跳过生成。如需重新生成请加 --force。")
        for tf in target_files:
            print(f"     - {tf}")
        sys.exit(0)

    # ---------- [2/4] 构造邻接矩阵 ----------
    build_via_gnndataset(args.data_dir, args.dataset)

    # ---------- [3/4] 验证 ----------
    verify_output(intermediate)

    # ---------- [4/4] 重命名 ----------
    fan_out_to_splits(intermediate, args.data_dir, args.dataset)

    # ---------- 清理 ----------
    if not args.keep_intermediate and os.path.exists(intermediate):
        os.remove(intermediate)
        print(f"\n🗑  已清理中间文件: s_pre_adj_mat_graph.npz")

    # ---------- 完成 ----------
    print()
    print("=" * 70)
    print("✅ 所有邻接矩阵已生成！下一步可启动训练：")
    print()
    print("   python train_collm_mf_din.py \\")
    print("       --cfg-path=tran_configs/collm_pretrain_mf_ood.yaml")
    print()
    print("⚠️  启动前确认 yaml 中以下路径已改为本机：")
    print(f"   ├─ llama_model              (Vicuna-7B 权重路径)")
    print(f"   ├─ datasets.*.path          → {args.data_dir}")
    print(f"   ├─ rec_config.pretrained_path (MF 预训练，可选)")
    print(f"   ├─ ckpt                     (CIE 权重，evaluate=True 时需要)")
    print(f"   └─ run.output_dir           (训练产出目录)")
    print("=" * 70)


if __name__ == "__main__":
    main()
