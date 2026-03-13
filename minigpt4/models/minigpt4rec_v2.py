import logging
import random
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
import os
import scipy.sparse as sp
from minigpt4.common.registry import registry
from peft.tuners.lora import Linear as OriginalLinear
from minigpt4.models.rec_model import Rec2Base, disabled_train
from peft.utils import transpose
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, GenerationConfig
import re
import numpy as np
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, \
    set_peft_model_state_dict


def get_ids_order(prompt):
    id_flags = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
    id_order_ = []
    for flag_ in id_flags:
        pos_ = prompt.find(flag_)
        if pos_ >= 0:
            id_order_.append(pos_)
    id_order_ = np.argsort(np.array(id_order_))
    return id_order_


# GCN Layer Definition
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(GCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, adj_matrix, node_features, node_ids=None):
        adj_matrix = adj_matrix.to(dtype=torch.float32, device=self.device)
        node_features = node_features.to(dtype=torch.float32, device=self.device)
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense().to(dtype=torch.float32)
        row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum, dtype=torch.float32))
        d_mat_inv_sqrt = torch.diag(torch.pow(row_sum, -0.5)).to(dtype=torch.float32, device=self.device)
        norm_adj = torch.sparse.mm(d_mat_inv_sqrt, torch.sparse.mm(adj_matrix, d_mat_inv_sqrt)).to(dtype=torch.float32,
                                                                                                   device=self.device)
        support = self.linear(node_features)
        support = self.dropout(support)
        output = torch.sparse.mm(norm_adj, support)
        return output


class TwoLayerGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(TwoLayerGCN, self).__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, out_dim, dropout)

    def forward(self, adj_matrix, node_features, node_ids=None):
        h = self.gcn1(adj_matrix, node_features, node_ids)
        h = F.relu(h)
        h = self.gcn2(adj_matrix, h, node_ids)
        return h


class LightGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=3, dropout=0.1):
        super(LightGCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 仅使用一层线性变换将初始嵌入映射到输出维度
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, adj_matrix, node_features, node_ids=None):
        adj_matrix = adj_matrix.to(dtype=torch.float32, device=self.device)
        node_features = node_features.to(dtype=torch.float32, device=self.device)

        # 计算归一化邻接矩阵的度数
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum, dtype=torch.float32))
        d_mat_inv_sqrt = torch.diag(torch.pow(row_sum, -0.5)).to(dtype=torch.float32, device=self.device)
        norm_adj = torch.sparse.mm(d_mat_inv_sqrt, torch.sparse.mm(adj_matrix, d_mat_inv_sqrt))

        # 多层传播
        embeddings = [node_features]
        h = node_features
        for _ in range(self.num_layers):
            h = torch.sparse.mm(norm_adj, h)
            h = self.dropout(h)
            embeddings.append(h)

        # 平均所有层的嵌入
        final_emb = torch.mean(torch.stack(embeddings, dim=0), dim=0)

        # 映射到目标维度
        output = self.linear(final_emb)
        output = F.normalize(output, dim=-1) * 0.1  # 归一化并缩放
        output = output.to(dtype=torch.float16)
        return output


class NGCFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(NGCFLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        # 两个线性变换：W1 用于自身和邻居，W2 用于交互项
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn.init.xavier_uniform_(self.W1.weight, gain=nn.init.calculate_gain('leaky_relu', param=0.2))
        nn.init.xavier_uniform_(self.W2.weight, gain=nn.init.calculate_gain('leaky_relu', param=0.2))
        if self.W1.bias is not None:
            nn.init.zeros_(self.W1.bias)
        if self.W2.bias is not None:
            nn.init.zeros_(self.W2.bias)

    def forward(self, adj_matrix, node_features):
        adj_matrix = adj_matrix.to(dtype=torch.float32, device=self.device)
        node_features = node_features.to(dtype=torch.float32, device=self.device)

        # 计算归一化系数
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        col_sum = torch.sparse.sum(adj_matrix, dim=0).to_dense()
        row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum, dtype=torch.float32))
        col_sum = torch.where(col_sum > 0, col_sum, torch.ones_like(col_sum, dtype=torch.float32))
        d_mat_inv_sqrt_row = torch.diag(torch.pow(row_sum, -0.5))
        d_mat_inv_sqrt_col = torch.diag(torch.pow(col_sum, -0.5))
        norm_adj = torch.sparse.mm(d_mat_inv_sqrt_row, torch.sparse.mm(adj_matrix, d_mat_inv_sqrt_col))

        # 自身变换
        h_self = self.W1(node_features)

        # 邻居聚合
        h_neigh = torch.sparse.mm(norm_adj, node_features)
        h_neigh = self.W1(h_neigh)

        # 交互项（逐元素乘法）
        h_inter = torch.sparse.mm(norm_adj, node_features * node_features)
        h_inter = self.W2(h_inter)

        # 合并并激活
        output = F.leaky_relu(h_self + h_neigh + h_inter, negative_slope=0.2)
        output = self.dropout(output)
        return output


class NGCF(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.1):
        super(NGCF, self).__init__()
        self.layers = nn.ModuleList([NGCFLayer(in_dim if i == 0 else hidden_dim,
                                               hidden_dim if i < num_layers - 1 else out_dim,
                                               dropout) for i in range(num_layers)])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, adj_matrix, node_features, node_ids=None):
        h = node_features
        for layer in self.layers:
            h = layer(adj_matrix, h)
        output = F.normalize(h, dim=-1) * 0.1
        output = output.to(dtype=torch.float16)
        return output


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, sample_size=25):
        super(GraphSAGE, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.sample_size = sample_size
        self.dropout = nn.Dropout(dropout)

        self.linear_self1 = nn.Linear(in_dim, hidden_dim)
        self.linear_neigh1 = nn.Linear(in_dim, hidden_dim)
        self.linear_self2 = nn.Linear(hidden_dim, out_dim)
        self.linear_neigh2 = nn.Linear(hidden_dim, out_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn.init.xavier_uniform_(self.linear_self1.weight)
        nn.init.xavier_uniform_(self.linear_neigh1.weight)
        nn.init.xavier_uniform_(self.linear_self2.weight)
        nn.init.xavier_uniform_(self.linear_neigh2.weight)

    def forward(self, adj_matrix, node_features, node_ids=None):
        adj_matrix = adj_matrix.to(dtype=torch.float32, device=self.device)
        node_features = node_features.to(dtype=torch.float32, device=self.device)

        # 第一层：聚合邻居特征
        if self.training:
            sampled_indices = []
            for i in range(adj_matrix.shape[0]):
                neighbors = torch.nonzero(adj_matrix[i], as_tuple=False).squeeze(-1)
                if len(neighbors) > self.sample_size:
                    neighbors = neighbors[torch.randperm(len(neighbors))[:self.sample_size]]
                sampled_indices.append(neighbors)
        else:
            sampled_indices = [torch.nonzero(adj_matrix[i], as_tuple=False).squeeze(-1) for i in
                               range(adj_matrix.shape[0])]

        h_neigh = torch.zeros_like(node_features)
        for i in range(node_features.shape[0]):
            if len(sampled_indices[i]) > 0:
                h_neigh[i] = node_features[sampled_indices[i]].mean(dim=0)

        h_self = self.linear_self1(node_features)
        h_neigh = self.linear_neigh1(h_neigh)
        h = F.relu(h_self + h_neigh)
        h = self.dropout(h)

        # 第二层变换
        h_neigh = torch.zeros_like(h)
        for i in range(h.shape[0]):
            if len(sampled_indices[i]) > 0:
                h_neigh[i] = h[sampled_indices[i]].mean(dim=0)
        h_self = self.linear_self2(h)
        h_neigh = self.linear_neigh2(h_neigh)
        output = h_self + h_neigh
        output = F.normalize(output, dim=-1) * 0.1
        output = output.to(dtype=torch.float16)
        return output


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

        self.linear = nn.Linear(in_dim, out_dim)
        self.attention = nn.Linear(2 * out_dim, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, adj_matrix, node_features):
        node_features = node_features.to(dtype=torch.float32, device=self.device)
        h = self.linear(node_features)
        h = self.dropout(h)

        N = h.shape[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_dim)
        e = F.leaky_relu(self.attention(a_input), negative_slope=self.alpha)
        e = e.squeeze(-1)

        adj_matrix = adj_matrix.to(dtype=torch.float32, device=self.device)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, h)
        return h_prime


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, alpha=0.2):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim, dropout, alpha)
        self.gat2 = GATLayer(hidden_dim, out_dim, dropout, alpha)

    def forward(self, adj_matrix, node_features, node_ids=None):
        h = self.gat1(adj_matrix, node_features)
        h = F.elu(h)
        h = self.gat2(adj_matrix, h)
        output = F.normalize(h, dim=-1) * 0.1
        output = output.to(dtype=torch.float16)
        return output


class MyLinear(OriginalLinear):
    def __init__(self, original_linear, layer_idx=None, lora_scale=1.0, gnn_scale=0 ,gcn_layer=0):
        super().__init__(
            adapter_name=original_linear.active_adapter,
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            fan_in_fan_out=original_linear.fan_in_fan_out,
            is_target_conv_1d_layer=original_linear.is_target_conv_1d_layer
        )
        self.weight = original_linear.weight
        self.lora_A = original_linear.lora_A
        self.lora_B = original_linear.lora_B
        self.lora_dropout = original_linear.lora_dropout
        self.scaling = original_linear.scaling
        self.merged = original_linear.merged
        self.lora_scale = lora_scale
        self.gnn_scale = gnn_scale
        self.layer_idx = layer_idx
        self.gcn_layer = gcn_layer
        # GCN data attributes
        self.gcn_output = None
        self.batch_indices = None
        self.token_positions = None
        self.local_node_ids = None

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        # Step 1: Compute the base linear transformation
        weight = transpose(self.weight, self.fan_in_fan_out)
        result = F.linear(x, weight, bias=self.bias)

        # Step 2: Check if LoRA is applicable
        if self.active_adapter not in self.lora_A.keys():
            return result.to(previous_dtype)

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            return result.to(previous_dtype)

        elif self.r[self.active_adapter] > 0 and not self.merged:

            x_lora = x.to(self.lora_A[self.active_adapter].weight.dtype)
            # print(f"[MyLinear] Layer {self.layer_idx} - x_lora shape: {x_lora.shape}")

            x_lora = self.lora_dropout[self.active_adapter](x_lora)
            # print(f"[MyLinear] Layer {self.layer_idx} - x_lora after dropout shape: {x_lora.shape}")

            lora_a_output = self.lora_A[self.active_adapter](x_lora)
            # print(f"[MyLinear] Layer {self.layer_idx} - lora_a_output shape: {lora_a_output.shape}")

            if self.layer_idx == self.gcn_layer and self.gcn_output is not None:

                valid_mask = self.local_node_ids != -1
                # print(
                #     f"[MyLinear] Layer {self.layer_idx} - valid_mask shape: {valid_mask.shape}, count: {valid_mask.sum()}")

                if valid_mask.any():
                    # 3.4.3: Extract valid indices and positions
                    batch_indices = self.batch_indices[valid_mask]
                    token_positions = self.token_positions[valid_mask]
                    local_node_ids = self.local_node_ids[valid_mask]
                    # print(f"[MyLinear] Layer {self.layer_idx} - batch_indices shape: {batch_indices.shape}")
                    # print(f"[MyLinear] Layer {self.layer_idx} - token_positions shape: {token_positions.shape}")
                    # print(f"[MyLinear] Layer {self.layer_idx} - local_node_ids shape: {local_node_ids.shape}")
                    # print(f"[MyLinear] Layer {self.layer_idx} - gcn_output shape: {self.gcn_output.shape}")

                    # lora_a_mean_before = lora_a_output[batch_indices, token_positions].abs().mean()
                    # print(
                    #     f"[MyLinear] Layer {self.layer_idx} - lora_a_output mean before GCN: {lora_a_mean_before}")

                    lora_a_output[batch_indices, token_positions] = (
                            lora_a_output[batch_indices, token_positions] * self.lora_scale +
                            self.gcn_output[local_node_ids] * self.gnn_scale
                    )

                    # lora_a_mean_after = lora_a_output[batch_indices, token_positions].abs().mean()
                    # print(f"[MyLinear] Layer {self.layer_idx} - lora_a_output mean after GCN: {lora_a_mean_after}")
                    # print(
                    #     f"[MyLinear] Layer {self.layer_idx} - lora_a_output shape after GCN: {lora_a_output.shape}")

            lora_output = self.lora_B[self.active_adapter](lora_a_output)
            # print(f"[MyLinear] Layer {self.layer_idx} - lora_b_output shape: {lora_b_output.shape}")

            result += lora_output
            # print(f"[MyLinear] Layer {self.layer_idx} - result shape after adding lora_output: {result.shape}")

        result = result.to(previous_dtype)
        # print(f"[MyLinear] Layer {self.layer_idx} - final result shape: {result.shape}")

        return result


@registry.register_model("mini_gpt4rec_v2")
class MiniGPT4Rec_v2(Rec2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4rec.yaml",
    }

    def __init__(
            self,
            rec_model="MF",
            rec_config=None,
            pretrained_rec=None,
            freeze_rec=True,
            rec_precision='fp16',
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,
            device_8bit=0,
            proj_token_num=1,
            proj_drop=0,
            lora_config=None,
            proj_mid=5,
            freeze_lora=False,
            freeze_proj=False,
            lora_gnn="lightgcn",
            dataset="book",
            lora_scale=1.0,
            gnn_scale=0,
            gcn_layer =0
    ):
        super().__init__()
        self.low_resource = low_resource
        self.proj_token_num = proj_token_num
        self.dataset = dataset
        print("running MiniGPT4Rec_v2 ...... ")
        print('Loading Rec_model')
        self.rec_model_type = rec_model
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, rec_precision)
        if self.rec_encoder is not None and pretrained_rec != "not_have":
            self.rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
            print("successfully load the pretrained model......")
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train
            logging.info("freeze rec encoder")
            print("freeze rec encoder")
        print('Loading Rec_model Done')
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')
        self.use_lora = False
        if lora_config is not None and lora_config.use_lora:
            print("Setting LoRA")
            self.use_lora = True
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],  #,"k_proj", "o_proj"
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model_lora = get_peft_model(self.llama_model, peft_config)

            if lora_gnn == "lightgcn":
                self.gcn = LightGCN(in_dim=256, out_dim=lora_config.r, num_layers=3, dropout=0.1 )
            elif lora_gnn == "ngcf":
                self.gcn = NGCF(in_dim=256, hidden_dim=256, out_dim=lora_config.r, num_layers=3, dropout=0.1)
            elif lora_gnn == "graphsage":
                self.gcn = GraphSAGE(in_dim=256, hidden_dim=256, out_dim=lora_config.r, dropout=0.1, sample_size=25)
            elif lora_gnn == "gat":
                self.gcn = GAT(in_dim=256, hidden_dim=256, out_dim=lora_config.r, dropout=0.1, alpha=0.2)
            elif lora_gnn == "gcn":
                self.gcn = TwoLayerGCN(in_dim=256, hidden_dim=256, out_dim=lora_config.r, dropout=0.1)
            else:
                raise ValueError(
                    f"Unsupported lora_gnn type: {lora_gnn}. Choose from ['lightgcn', 'ngcf', 'graphsage', 'gat' , 'gcn']")
            self.num_users = rec_config.user_num
            self.num_items = rec_config.item_num
            # Store MyLinear instances
            target_layer_str = f'model.layers.{gcn_layer}'
            print(f"Preparing to inject GCN into: {target_layer_str} with gnn_scale: {gnn_scale}")

            # Store MyLinear instances
            self.my_linears = []
            for name, module in self.llama_model_lora.named_modules():
                # 2. 使用 target_layer_str 替代硬编码的 'model.layers.31'
                if target_layer_str in name and any(t in name for t in ['q_proj', 'v_proj']):   #, 'k_proj', 'o_proj'
                    if isinstance(module, OriginalLinear):
                        # 3. 实例化 MyLinear 时传入所有动态参数
                        # 注意：这里必须传入 gnn_scale 和 gcn_layer，否则 MyLinear 里这些值默认为 0
                        new_module = MyLinear(
                            original_linear=module,
                            layer_idx=gcn_layer,
                            lora_scale=lora_scale,
                            gnn_scale=gnn_scale,
                            gcn_layer=gcn_layer
                        )

                        # 替换模型中的层
                        parent_name = '.'.join(name.split('.')[:-1])
                        attr_name = name.split('.')[-1]
                        parent_module = self.llama_model_lora
                        for part in parent_name.split('.'):
                            parent_module = getattr(parent_module, part)
                        setattr(parent_module, attr_name, new_module)

                        self.my_linears.append(new_module)

            print(
                f"Setting LoRA, GCN, and MyLinear for layer {gcn_layer} Done. Replaced {len(self.my_linears)} modules.")
        else:
            self.gcn = None
        if freeze_lora and self.use_lora:
            print("freeze LoRA...")
            for name, param in self.llama_model_lora.named_parameters():
                param.requires_grad = False
        if self.rec_encoder is not None and 'prompt' not in rec_model:
            self.llama_proj = nn.Sequential(
                nn.Linear(self.rec_encoder.config.embedding_size,
                          self.rec_encoder.config.embedding_size * int(proj_mid)),
                nn.ReLU(),
                nn.Linear(self.rec_encoder.config.embedding_size * int(proj_mid),
                          self.llama_model.config.hidden_size * self.proj_token_num),
            )
        elif self.rec_encoder is not None and rec_model == "personlized_prompt":
            self.llama_proj = nn.Linear(rec_config.item_num + rec_config.user_num,
                                        self.llama_model.config.hidden_size * self.proj_token_num, bias=False)
        elif self.rec_encoder is not None and rec_model == "soft_prompt":
            self.llama_proj = nn.Linear(2, self.llama_model.config.hidden_size * self.proj_token_num, bias=False)
        else:
            self.llama_proj = None
        if freeze_proj and self.llama_proj is not None:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            self.llama_proj = self.llama_proj.eval()
            self.llama_proj.train = disabled_train
            logging.info("!!!! freeze llama_proj...")
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt = False
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt List: \n{}'.format(self.prompt_list))
            self.has_pri_decode = False
            self.prompt_list_p = None
        else:
            self.prompt_list = []
            self.prompt_list_p = None
        self._load_adjacency_matrix(mode='train')

    def _load_adjacency_matrix(self, mode='train'):
        try:
            if self.dataset == "book":
                # print("book adj")
                adj_matrix_paths = {
                    'train': "/mnt/disk2/wgj/wgj_collm/collm-datasets/amazon_book/book/book_graph_train.npz",
                    'valid': "/mnt/disk2/wgj/wgj_collm/collm-datasets/amazon_book/book/book_graph_valid.npz",
                    'test': "/mnt/disk2/wgj/wgj_collm/collm-datasets/amazon_book/book/book_graph_test.npz"
                }

            elif self.dataset == "movie":
                # print("movie adj")
                adj_matrix_paths = {
                    'train': "/mnt/disk2/wgj/wgj_collm/collm-datasets/ml-1m/ml-1m/movie_graph_train.npz",
                    'valid': "/mnt/disk2/wgj/wgj_collm/collm-datasets/ml-1m/ml-1m/movie_graph_valid.npz",
                    'test': "/mnt/disk2/wgj/wgj_collm/collm-datasets/ml-1m/ml-1m/movie_graph_test.npz"
                }

            adj_matrix_path = adj_matrix_paths.get(mode, adj_matrix_paths['train'])
            # print(f"Loading adjacency matrix for {mode} from: {adj_matrix_path}")

            # 加载 .npz 文件
            norm_adj = sp.load_npz(adj_matrix_path)
            # print(f"Loaded adjacency matrix shape: {norm_adj.shape}, non-zero elements: {norm_adj.nnz}")

            # 转换为 COO 格式并存储
            self.adj_matrix_coo = norm_adj.tocoo()
            self.adj_matrix_shape = norm_adj.shape
            self.adj_matrix_csr = norm_adj.tocsr()
            # print(f"Stored COO matrix, shape: {self.adj_matrix_shape}")
        except Exception as e:
            print(f"Failed to load adjacency matrix: {e}")
            raise

    def get_subgraph(self, node_ids):
        DEFAULT_HOP = 1
        MAX_NODES = None
        hop = int(DEFAULT_HOP)
        if hop < 0:
            hop = 0
        if not hasattr(self, "adj_matrix_csr") or self.adj_matrix_csr is None:
            self.adj_matrix_csr = self.adj_matrix_coo.tocsr()
        csr = self.adj_matrix_csr


        node_ids_np = node_ids.unique().detach().cpu().numpy().astype(np.int64)
        node_ids_np = node_ids_np[(node_ids_np >= 0) & (node_ids_np < csr.shape[0])]
        if node_ids_np.size == 0:
            empty = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=self.device),
                torch.zeros((0,), dtype=torch.float32, device=self.device),
                (0, 0)
            ).coalesce()
            return empty, torch.empty((0,), dtype=torch.long, device=self.device)

        current = set(node_ids_np.tolist())
        frontier = set(current)

        # --- BFS 扩展 hop 层 ---
        for _ in range(hop):
            if not frontier:
                break

            new_neighbors = set()
            for u in frontier:
                start, end = csr.indptr[u], csr.indptr[u + 1]
                if end > start:
                    new_neighbors.update(csr.indices[start:end].tolist())

            new_neighbors.difference_update(current)
            if not new_neighbors:
                break


            if MAX_NODES is not None and (len(current) + len(new_neighbors) > MAX_NODES):
                remaining = MAX_NODES - len(current)
                if remaining <= 0:
                    break
                new_list = list(new_neighbors)
                chosen = np.random.choice(new_list, size=remaining, replace=False).tolist()
                new_neighbors = set(chosen)

            current.update(new_neighbors)
            frontier = new_neighbors

        sub_nodes = np.array(sorted(current), dtype=np.int64)
        sub_sp = csr[sub_nodes][:, sub_nodes].tocoo()

        indices = torch.tensor(
            np.vstack([sub_sp.row, sub_sp.col]),
            dtype=torch.long,
            device=self.device
        )
        values = torch.tensor(sub_sp.data, dtype=torch.float32, device=self.device)
        shape = (len(sub_nodes), len(sub_nodes))

        sub_adj_matrix = torch.sparse_coo_tensor(indices, values, size=shape).coalesce()
        sub_node_ids = torch.tensor(sub_nodes, dtype=torch.long, device=self.device)
        return sub_adj_matrix, sub_node_ids

    def to_be_trained(self):
        if self.use_lora:
            return True
        id_terms = ["<UserID>", "<ItemIDList>", "<TargetItemID>", "<DCNFeature>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True
        return False

    def set_mode(self, mode):
        self.run_mode_ = mode

    def rec_to_cpu(self):
        self.rec_encoder.to("cpu")
        self.rec_encoder.float()

    def set_answer_type(self, mode):
        if mode == 'v1':
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        elif mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            pos_ans_id = self.llama_tokenizer(self.pos_ans[0], add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(self.neg_ans[0], add_special_tokens=False).input_ids[0]
            print("answer token ids: pos:", pos_ans_id, "neg ids:", neg_ans_id)
        else:
            raise NotImplementedError("not implement this types of answers")

    def print_prompt(self):
        print('Prompt Pos Example \n{} {} or {}'.format(random.choice(self.prompt_list), self.pos_ans[0],
                                                        self.neg_ans[0]))

    def encode_recdata_v1(self, sample):
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        with self.maybe_autocast():
            all_user_embeds, all_items_embeds = self.rec_encoder.computer()
            user_embeds = self.rec_encoder.user_encoder(sample['UserID'], all_users=all_user_embeds).unsqueeze(-2)
            targetItem_embed = self.rec_encoder.item_encoder(sample['PairItemIDs'], all_items=all_items_embeds)
            user_embeds_llama = self.llama_proj(user_embeds)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed)
        sample_embeds_llama = {
            'User_emb': user_embeds_llama,
            'PairItem_emb': targetItem_embeds_llama,
        }
        sample_atts_llama = None
        return sample_embeds_llama, sample_atts_llama

    def encode_recdata_v2(self, sample, ids_order=None):
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        with self.maybe_autocast():
            batch_size = sample['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            if self.rec_model_type == "sasrec":
                user_embeds = self.rec_encoder.seq_encoder(sample['sas_seq']).unsqueeze(-2)
            elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
                user_embeds = self.rec_encoder.all_encode(sample['UserID'], sample['TargetItemID'],
                                                          sample['sas_seq'][:, -10:]).unsqueeze(-2)
            else:
                user_embeds = self.rec_encoder.user_encoder(sample['UserID'], all_users=all_user_embeds).unsqueeze(-2)
            targetItem_embed = self.rec_encoder.item_encoder(sample['TargetItemID'],
                                                             all_items=all_item_embeds).unsqueeze(-2)
            user_embeds_llama = self.llama_proj(user_embeds).reshape(batch_size, -1, self.proj_token_num, hidden_size)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed).reshape(batch_size, -1, self.proj_token_num,
                                                                                hidden_size)
            num_users = self.rec_encoder.config.user_num
            node_ids = []
            if 'InteractedItemIDs_pad' in sample.keys() and len(ids_order) == 3:
                interactedItem_embeds = self.rec_encoder.item_encoder(sample['InteractedItemIDs_pad'],
                                                                      all_items=all_item_embeds)
                interactedItem_embeds_llama = self.llama_proj(interactedItem_embeds).reshape(batch_size, -1,
                                                                                             self.proj_token_num,
                                                                                             hidden_size)
                merged_embeds = [user_embeds_llama, interactedItem_embeds_llama, targetItem_embeds_llama]
                merged_embeds = [merged_embeds[k] for k in ids_order]
                merged_embeds = torch.cat(merged_embeds, dim=1)
                idx_flag = torch.ones_like(sample['InteractedItemIDs_pad'])
                idx_flag = torch.where(sample['InteractedItemIDs_pad'] == self.rec_encoder.padding_index, 0, idx_flag)
                idx_flag = [torch.ones([idx_flag.shape[0], 1]).to(idx_flag.device), idx_flag,
                            torch.ones([idx_flag.shape[0], 1]).to(idx_flag.device)]
                idx_flag = [idx_flag[k] for k in ids_order]
                idx_flag = torch.cat(idx_flag, dim=1).to(device)
                idx_nopad = torch.nonzero(idx_flag)
                for b in range(batch_size):
                    batch_mask = idx_nopad[:, 0] == b
                    batch_idx_nopad = idx_nopad[batch_mask]
                    for _, pos in batch_idx_nopad:
                        if pos == 0:  # UserID
                            node_ids.append(sample['UserID'][b].item())
                        elif pos < idx_flag.shape[1] - 1:  # InteractedItemIDs
                            item_pos = pos - 1
                            item_id = sample['InteractedItemIDs_pad'][b, item_pos].item()
                            if item_id != self.rec_encoder.padding_index:
                                node_ids.append(item_id + num_users)
                        else:  # TargetItemID
                            node_ids.append(sample['TargetItemID'][b].item() + num_users)
                node_ids = torch.tensor(node_ids, device=device, dtype=torch.long)
                mf_user_embeds = self.rec_encoder.user_embedding.weight
                mf_item_embeds = self.rec_encoder.item_embedding.weight
                mf_embeddings = torch.cat([mf_user_embeds, mf_item_embeds], dim=0)
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'InteractedItems_embs': interactedItem_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'merged_embs': merged_embeds[idx_nopad[:, 0], idx_nopad[:, 1]].reshape(-1, hidden_size),
                    'node_ids': node_ids,
                    'mf_embeddings': mf_embeddings,
                }
            else:
                node_ids = torch.cat([
                    sample['UserID'],
                    sample['TargetItemID'] + num_users
                ], dim=0)
                mf_user_embeds = self.rec_encoder.user_embedding.weight
                mf_item_embeds = self.rec_encoder.item_embedding.weight
                mf_embeddings = torch.cat([mf_user_embeds, mf_item_embeds], dim=0)
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'InteractedItems_embs': None,
                    'merged_embs': None,
                    'node_ids': node_ids,
                    'mf_embeddings': mf_embeddings,
                }
        sample_atts_llama = None
        return sample_embeds_llama, sample_atts_llama

    def recprompt_wrap_v1(self, samples, ori_samples, atts_sample, prompt):
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<ItemID>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            unk_ = self.llama_tokenizer.unk_token
            prompt = bos + prompt
            prompt = prompt.replace("<UserID>", unk_)
            prompt = prompt.replace("<ItemID>", unk_)
            prompt_list = []
            for k in range(batch_size):
                prompt_ = prompt + ""
                prompt_list.append(prompt_)
            self.llama_tokenizer.padding_side = "left"
            prompts_tokens = self.llama_tokenizer(
                prompt_list,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(samples['User_emb'].device)
            unk_token_id = self.llama_tokenizer.unk_token_id
            replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)
            prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            if "<UserID>" in prompt_ori and "<ItemID>" in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = torch.cat(
                    [samples['User_emb'], samples['PairItem_emb']], dim=-2
                ).reshape(-1, samples['User_emb'].shape[-1])
            else:
                raise RuntimeError("the pretraining just support one type prompt")
            return prompt_embeds, prompts_tokens.attention_mask

    def recprompt_wrap_v2(self, samples, ori_samples, atts_sample, prompt):
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<ItemIDList>", "<ItemTitleList>", "<TargetItemID>", "<TargetItemTitle>",
                            "<DCNFeature>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            unk_ = self.llama_tokenizer.unk_token
            unk_ = ".".join([unk_] * self.proj_token_num)
            prompt = bos + prompt
            prompt = prompt.replace("<UserID>", unk_)
            prompt = prompt.replace("<TargetItemID>", unk_)
            prompt = prompt.replace("<DCNFeature>", unk_)
            prompt_list = []
            for k in range(batch_size):
                prompt_ = prompt + ""
                if 'InteractedNum' in ori_samples.keys():
                    prompt_ = prompt_.replace('<ItemIDList>', ', '.join([unk_] * ori_samples['InteractedNum'][k]))
                    prompt_ = prompt_.replace("<ItemTitleList>", ori_samples['InteractedItemTitles'][k])
                prompt_ = prompt_.replace("<TargetItemTitle>", ori_samples['TargetItemTitle'][k])
                prompt_list.append(prompt_)
            if not self.has_print_prompt:
                print("prompt example:", random.choice(prompt_list))
                self.has_print_prompt = True
            self.llama_tokenizer.padding_side = "left"
            prompts_tokens = self.llama_tokenizer(
                prompt_list,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(ori_samples['UserID'].device)
            unk_token_id = self.llama_tokenizer.unk_token_id
            if not self.has_pri_decode:
                print("#######prompt decoded example:",
                      ' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
                self.has_pri_decode = True
            replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)
            node_ids = samples['node_ids']

            num_found = replaced_idx.shape[0]
            num_target = node_ids.shape[0]

            if num_found != num_target:
                print(
                    f"[Warning-Fix] Shape mismatch detected! Found {num_found} tokens but expected {num_target} nodes.")
                print(f"Sample UserID: {samples.get('UserID', 'Unknown')}")
                min_len = min(num_found, num_target)
                replaced_idx = replaced_idx[:min_len]
                node_ids = node_ids[:min_len]
            replaced_idx_with_node = torch.cat([
                replaced_idx,
                node_ids.unsqueeze(-1)
            ], dim=-1)


            prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            if "<UserID>" in prompt_ori and "<ItemIDList>" in prompt_ori and "<TargetItemID>" in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = samples['merged_embs']
            elif "<UserID>" in prompt_ori and "<TargetItemID>" in prompt_ori and "<ItemIDList>" not in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = torch.cat(
                    [samples['User_emb'], samples['TargetItem_emb']], dim=-2
                ).reshape(-1, samples['User_emb'].shape[-1])
            elif "<DCNFeature>" in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = samples['User_emb'].reshape(-1, samples[
                    'User_emb'].shape[-1])
            else:
                pass
            return prompt_embeds, prompts_tokens.attention_mask, replaced_idx

    def prompt_with_p(self, p):
        if self.prompt_list_p is None:
            prompt_list_p = []
            for k in range(len(p)):
                prompt_list_p.extend([self.prompt_list[k]] * p[k])
            self.prompt_list_p = prompt_list_p
            return self.prompt_list_p
        else:
            return self.prompt_list_p

    def forward(self, samples):
        # 动态加载邻接矩阵基于样本的模式
        mode = 'train'
        if 'is_valid' in samples and samples['is_valid']:
            mode = 'valid'
        elif 'is_test' in samples and samples['is_test']:
            mode = 'test'
        self._load_adjacency_matrix(mode=mode)
        # print(f"[forward] Loaded adjacency matrix for mode: {mode}")

        if self.run_mode_ == 'v1':
            return self.forward_v1(samples)
        elif self.run_mode_ == 'v2':
            return self.forward_v2(samples)
        else:
            raise NotImplementedError("None-template version has not been implemented...")

    def prompt_based_encode_v2(self, prompt, samples):
        """
        Encode prompt and samples into embeddings and attention masks for LLaMA,
        and prepare node_ids and replaced_idx for GCN.

        Args:
            prompt (str): The prompt template with placeholders.
            samples (dict): Contains 'UserID', 'TargetItemID', 'label', and optionally 'InteractedItemIDs_pad', 'InteractedNum', 'InteractedItemTitles', 'TargetItemTitle'.

        Returns:
            sample_embeds (torch.Tensor): Input embeddings for LLaMA.
            atts_samples (torch.Tensor): Attention mask for LLaMA.
            samples_encode (dict): Contains 'node_ids' and 'replaced_idx' for GCN.
        """
        id_orders = get_ids_order(prompt)
        samples_encode, atts_samples = self.encode_recdata_v2(samples, ids_order=id_orders)
        sample_embeds, atts_samples, replaced_idx = self.recprompt_wrap_v2(samples_encode, samples, atts_samples,
                                                                           prompt)

        # Construct samples_encode with node_ids and replaced_idx
        samples_encode_out = {
            'node_ids': samples_encode['node_ids'],
            'replaced_idx': replaced_idx,
            'mf_embeddings': samples_encode['mf_embeddings'] if 'mf_embeddings' in samples_encode else None
        }

        # print("prompt_based_encode_v2: samples_encode keys:", samples_encode_out.keys())
        # print("node_ids shape:", samples_encode_out['node_ids'].shape)
        # print("replaced_idx shape:", samples_encode_out['replaced_idx'].shape)
        return sample_embeds, atts_samples, samples_encode_out

    def forward_v1(self, samples):
        samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)
        self.llama_tokenizer.padding_side = "right"
        device = samples['UserID'].device
        ans_ = {1: self.pos_ans, 0: self.neg_ans}
        text = [random.choice(ans_[int(t)]) + self.end_sym for t in samples["label"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(
            -100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)
        with self.maybe_autocast():
            if self.use_lora:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        loss = outputs.loss
        return {"loss": loss}

    def forward_v2(self, samples):
        user_selective_prompts = False
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_with_p([5, 5, 5, 1]))
            # print("Selected prompt:", prompt)
            sample_embeds, atts_samples, samples_encode = self.prompt_based_encode_v2(prompt, samples)
            # print("sample_embeds shape:", sample_embeds.shape)
            # print("atts_samples shape:", atts_samples.shape)
            # print("samples_encode keys:", samples_encode.keys())

        self.llama_tokenizer.padding_side = "right"
        device = samples['UserID'].device
        ans_ = {1: self.pos_ans[0], 0: self.neg_ans[0]}
        text = [ans_[int(t)] for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(
            -100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        if not hasattr(self, 'adj_matrix_coo'):
            raise ValueError("Adjacency matrix not loaded in model")
        if 'UserID' not in samples or 'TargetItemID' not in samples:
            raise ValueError("samples must contain 'UserID' and 'TargetItemID' for GCN")
        if 'node_ids' not in samples_encode or 'replaced_idx' not in samples_encode:
            raise ValueError("samples_encode must contain 'node_ids' and 'replaced_idx' for GCN enhancement")

        node_ids = samples_encode['node_ids'].to(self.device)
        sub_adj_matrix, sub_node_ids = self.get_subgraph(node_ids)
        node_map = {old_id.item(): new_id for new_id, old_id in enumerate(sub_node_ids)}

        with self.maybe_autocast():
            mf_user_embeds = self.rec_encoder.user_embedding.weight
            mf_item_embeds = self.rec_encoder.item_embedding.weight
            mf_embeddings = torch.cat([mf_user_embeds, mf_item_embeds], dim=0)
            self.gcn.eval()
            with torch.cuda.amp.autocast(enabled=False):
                sub_adj_matrix = sub_adj_matrix.to(dtype=torch.float32, device=self.device)
                node_features = mf_embeddings[sub_node_ids].to(dtype=torch.float32, device=self.device)
                gcn_output = self.gcn(sub_adj_matrix, node_features, sub_node_ids)
                gcn_output = torch.nn.functional.normalize(gcn_output, dim=-1) * 0.1
                gcn_output = gcn_output.to(dtype=torch.float16)

            batch_mask = samples_encode['replaced_idx'][:, 0] < samples['UserID'].shape[0]
            batch_indices = samples_encode['replaced_idx'][batch_mask, 0]
            token_positions = samples_encode['replaced_idx'][batch_mask, 1]
            batch_node_ids = samples_encode['node_ids'][batch_mask]
            local_node_ids = torch.tensor([node_map.get(nid.item(), -1) for nid in batch_node_ids], device=self.device)

            if self.use_lora:
                for my_linear in self.my_linears:
                    my_linear.gcn_output = gcn_output
                    my_linear.batch_indices = batch_indices
                    my_linear.token_positions = token_positions
                    my_linear.local_node_ids = local_node_ids

            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )

            if self.use_lora:
                for my_linear in self.my_linears:
                    my_linear.gcn_output = None
                    my_linear.batch_indices = None
                    my_linear.token_positions = None
                    my_linear.local_node_ids = None

        pos_ans_id = self.llama_tokenizer(ans_[int(1)], add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(ans_[int(0)], add_special_tokens=False).input_ids[0]
        logits = outputs.logits[:, -t_posi, :][:, pos_ans_id]
        loss = nn.functional.binary_cross_entropy_with_logits(logits, samples['label'].float())

        return {"loss": loss}

    def generate_for_samples_v1(self, samples):
        samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)
        self.llama_tokenizer.padding_side = "right"
        device = samples_encode['User_emb'].device
        ans_ = {1: self.pos_ans, 0: self.neg_ans}
        text = [random.choice(ans_[int(t)]) + self.end_sym for t in samples["label"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(
            -100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)
        with self.maybe_autocast():
            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        loss = outputs.loss
        return {"loss": loss}

    def generate_for_samples_v2(self, samples, return_all=False):
        # Load adjacency matrix based on sample mode
        mode = 'train'
        if 'is_valid' in samples and samples['is_valid']:
            mode = 'valid'
        elif 'is_test' in samples and samples['is_test']:
            mode = 'test'
        self._load_adjacency_matrix(mode=mode)

        user_selective_prompts = False
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_with_p([5, 5, 5, 1]))
            id_orders = get_ids_order(prompt)
            samples_encode, atts_samples = self.encode_recdata_v2(samples, id_orders)
            sample_embeds, atts_samples, replaced_idx = self.recprompt_wrap_v2(samples_encode, samples, atts_samples,
                                                                               prompt)

        self.llama_tokenizer.padding_side = "right"
        device = samples['UserID'].device
        pos_ans = self.pos_ans[0]
        neg_ans = self.neg_ans[0]
        ans_ = {1: pos_ans, 0: neg_ans}
        text = [ans_[int(t)] for t in samples["label"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)
        t_posi = to_regress_tokens.input_ids.shape[-1] + 1
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(
            -100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)
        if not hasattr(self, 'adj_matrix_coo'):
            raise ValueError("Adjacency matrix not loaded in model")
        if 'UserID' not in samples or 'TargetItemID' not in samples:
            raise ValueError("samples must contain 'UserID' and 'TargetItemID' for GCN")
        node_ids = samples_encode['node_ids'].to(self.device)
        sub_adj_matrix, sub_node_ids = self.get_subgraph(node_ids)
        node_map = {old_id.item(): new_id for new_id, old_id in enumerate(sub_node_ids)}
        with self.maybe_autocast():
            mf_user_embeds = self.rec_encoder.user_embedding.weight
            mf_item_embeds = self.rec_encoder.item_embedding.weight
            mf_embeddings = torch.cat([mf_user_embeds, mf_item_embeds], dim=0)
            if self.use_lora:
                self.gcn.eval()
                self.llama_model_lora.eval()
                with torch.cuda.amp.autocast(enabled=False):
                    sub_adj_matrix = sub_adj_matrix.to(dtype=torch.float32, device=self.device)
                    node_features = mf_embeddings[sub_node_ids].to(dtype=torch.float32, device=self.device)
                    gcn_output = self.gcn(sub_adj_matrix, node_features, sub_node_ids)
                    gcn_output = gcn_output.to(dtype=torch.float16)
                batch_mask = replaced_idx[:, 0] < samples['UserID'].shape[0]
                batch_indices = replaced_idx[batch_mask, 0]
                token_positions = replaced_idx[batch_mask, 1]
                batch_node_ids = samples_encode['node_ids'][batch_mask]
                local_node_ids = torch.tensor([node_map.get(nid.item(), -1) for nid in batch_node_ids],
                                              device=self.device)
                # Set GCN data
                for my_linear in self.my_linears:
                    my_linear.gcn_output = gcn_output
                    my_linear.batch_indices = batch_indices
                    my_linear.token_positions = token_positions
                    my_linear.local_node_ids = local_node_ids
            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            # Clear GCN data
            if self.use_lora:
                for my_linear in self.my_linears:
                    my_linear.gcn_output = None
                    my_linear.batch_indices = None
                    my_linear.token_positions = None
                    my_linear.local_node_ids = None
            pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]
            logits = outputs.logits[:, -t_posi, :][:, pos_ans_id]
            loss = nn.functional.binary_cross_entropy_with_logits(logits, samples['label'].float())
            if return_all:
                return {"outputs": outputs, "logits": logits, "loss": loss}
            return {"loss": loss, "logits": logits}

    def generate_for_samples(self, samples):
        if self.run_mode_ == 'v1':
            return self.generate_for_samples_v1(samples)
        elif self.run_mode_ == 'v2':
            return self.generate_for_samples_v2(samples)
        else:
            raise NotImplementedError("Not implement the default version")

    def generate_sequence(self, samples):
        # 动态加载邻接矩阵基于样本的模式
        mode = 'train'
        if 'is_valid' in samples and samples['is_valid']:
            mode = 'valid'
        elif 'is_test' in samples and samples['is_test']:
            mode = 'test'
        self._load_adjacency_matrix(mode=mode)
        # print(f"[generate_sequence] Loaded adjacency matrix for mode: {mode}")

        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            id_orders = get_ids_order(prompt)
            samples_encode, atts_samples = self.encode_recdata_v2(samples, id_orders)
            sample_embeds, atts_samples, replaced_idx = self.recprompt_wrap_v2(samples_encode, samples, atts_samples,
                                                                               prompt)
        inputs_embeds = sample_embeds

        if not hasattr(self, 'adj_matrix_coo'):
            raise ValueError("Adjacency matrix not loaded in model")

        if 'UserID' not in samples or 'TargetItemID' not in samples:
            raise ValueError("samples must contain 'UserID' and 'TargetItemID' for GCN")
        node_ids = torch.cat([
            samples['UserID'],
            samples['TargetItemID'] + self.num_users
        ], dim=0).to(self.device)

        # 提取子图
        sub_adj_matrix, sub_node_ids = self.get_subgraph(node_ids)
        # print(f"[generate_sequence] sub_adj_matrix shape: {sub_adj_matrix.shape}")
        # print(f"[generate_sequence] sub_node_ids shape: {sub_node_ids.shape}")

        with torch.no_grad():
            try:
                if not self.use_lora:
                    outputs = self.llama_model.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=10,
                        num_beams=1,
                        do_sample=True,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.0,
                        length_penalty=1,
                        temperature=1.0,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                else:
                    batch_size = samples['UserID'].shape[0]
                    node_features = samples_encode['mf_embeddings']
                    with torch.cuda.amp.autocast(enabled=False):
                        sub_adj_matrix = sub_adj_matrix.to(dtype=torch.float32, device=self.device)
                        node_features = node_features[sub_node_ids].to(dtype=torch.float32, device=self.device)
                        gcn_output = self.gcn(sub_adj_matrix, node_features, sub_node_ids)
                        # print(f"[generate_sequence] gcn_output shape: {gcn_output.shape}")
                    original_inputs_embeds = inputs_embeds.clone()
                    outputs = self.llama_model_lora.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=10,
                        num_beams=1,
                        do_sample=True,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.0,
                        length_penalty=1,
                        temperature=1.0,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    node_map = {old_id.item(): new_id for new_id, old_id in enumerate(sub_node_ids)}
                    # print(f"[generate_sequence] node_map: {list(node_map.items())[:5]}")
                    for name, module in self.llama_model_lora.named_modules():
                        if hasattr(module, 'lora_A') and any(t in name for t in ["q_proj", "v_proj"]):  #, "k_proj", "o_proj"
                            lora_a_output = module.lora_A[module.active_adapter](
                                module.lora_dropout[module.active_adapter](original_inputs_embeds)
                            )
                            for b in range(batch_size):
                                batch_mask = replaced_idx[:, 0] == b
                                if not batch_mask.any():
                                    continue
                                batch_replaced_idx = replaced_idx[batch_mask, 1]
                                batch_node_ids = samples_encode['node_ids'][batch_mask]
                                for token_pos, node_id in zip(batch_replaced_idx, batch_node_ids):
                                    local_node_id = node_map.get(node_id.item(), -1)
                                    if local_node_id == -1:
                                        print(
                                            f"[generate_sequence] Warning: node_id {node_id} not found in sub_node_ids")
                                        continue
                                    lora_a_output[b, token_pos] += gcn_output[local_node_id]
            except Exception as e:
                print(f"errors.....: {e}")
        print(inputs_embeds.shape, outputs.sequences.shape)
        print(self.llama_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True), samples['label'])
        print()
        return {"loss": 0, 'logits': outputs.logit}

    def encode_allinputs(self, samples, mode='v1'):
        if mode == 'v2':
            samples_encode, atts_samples = self.encode_recdata_v2(samples)
        else:
            samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            if mode == 'v2':
                sample_embeds, atts_samples, _ = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, prompt)
            else:
                sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)
        inputs_embeds = sample_embeds
        return inputs_embeds

    @classmethod
    def from_config(cls, cfg):
        rec_model = cfg.get('rec_model', "MF")
        rec_config = cfg.get("rec_config")
        embedding_size = cfg.get("rec_emb_size")
        freeze_rec = cfg.get("freeze_rec", True)
        rec_precision = cfg.get("rec_precision", 'fp16')
        lora_config = cfg.get("lora_config")
        llama_model = cfg.get("llama_model")
        proj_token_num = cfg.get("proj_token_num")
        proj_mid = cfg.get("proj_mid_times")
        freeze_proj = cfg.get("freeze_proj")
        freeze_lora = cfg.get("freeze_lora")
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        lora_gnn = cfg.get("lora_gnn", "")
        dataset = cfg.get("dataset", "")
        lora_scale = cfg.get("lora_scale", 1.0)
        gnn_scale = cfg.get("gnn_scale", 0)
        gcn_layer = cfg.get("gcn_layer", 0)
        model = cls(
            rec_model=rec_model,
            rec_config=rec_config,
            pretrained_rec=rec_config['pretrained_path'],
            freeze_rec=freeze_rec,
            rec_precision=rec_precision,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            proj_token_num=cfg.get("proj_token_num"),
            proj_drop=cfg.get("proj_drop"),
            lora_config=lora_config,
            proj_mid=proj_mid,
            freeze_lora=freeze_lora,
            freeze_proj=freeze_proj,
            lora_gnn=lora_gnn,
            lora_scale=lora_scale,
            gnn_scale=gnn_scale,
            gcn_layer=gcn_layer,
            dataset=dataset,
        )
        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load MiniGPT4Rec Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print("loading message, msg.... ", msg)
            if os.path.exists(rec_config['pretrained_path']) and freeze_rec:
                model.rec_encoder.load_state_dict(torch.load(rec_config['pretrained_path'], map_location="cpu"))
        ans_type = cfg.get('ans_type')
        model.set_answer_type(mode=ans_type)
        model.print_prompt()
        return model