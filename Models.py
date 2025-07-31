import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.parser import parse_args
from utility.hgnnp import HGNNP
from utility.load_data import *

args = parse_args()

class MHGR(nn.Module):
    def __init__(self, n_users, n_items, num_non_zero_elements, embedding_dim, weight_size, dropout_list, image_feats, text_feats, ui_graph, iu_graph):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size

        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)

        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()

        # self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        # self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()  
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(args.embed_size)
        self.tau = 0.5

        self.Hv2eimages_weight = nn.Parameter(torch.ones(num_non_zero_elements))
        self.Hv2etexts_weight = nn.Parameter(torch.ones(num_non_zero_elements))
        self.He2vimages_weight = nn.Parameter(torch.ones(num_non_zero_elements))
        self.He2vtexts_weight = nn.Parameter(torch.ones(num_non_zero_elements))

        self.HG= Gener_hyper(#, self.con_HG , self.text_HG, self.image_HG
            args.data_path + '{}/data_pre_processing/'.format(args.dataset), self.n_items, ui_graph, iu_graph)
        self.HGNN1 = HGNNP(self.image_feats.shape[1], args.embed_size, use_bn=True).cuda()  # .to(device)
        self.HGNN2 = HGNNP(self.text_feats.shape[1], args.embed_size, use_bn=True).cuda()


        self.embedding_dict = {'user':{}, 'item':{}}

        user_dim = 128
        item_dim = 128

        self.net1 = nn.Sequential(
            nn.Linear(user_dim, int(user_dim * 2)),  # 全连接层，输入维度为dim
            nn.LeakyReLU(True),  # LeakyReLU激活函数
            nn.BatchNorm1d(int(user_dim * 2)),  # 逐批标准化层
            nn.Dropout(0.2),  # Dropout层，按照args.G_drop1的参数进行随机失活

            nn.Linear(user_dim * 2, int(user_dim)),  # 全连接层，输入维度为dim
            nn.LeakyReLU(True),  # LeakyReLU激活函数
            # nn.BatchNorm1d(int(user_dim/2)),  # 逐批标准化层
            nn.Dropout(0.2),  # Dropout层，按照args.G_drop1的参数进行随机失活

            # nn.Linear(int(user_dim), int(user_dim/2)),  # 全连接层
            # nn.LeakyReLU(True),  # LeakyReLU激活函数
            # #nn.BatchNorm1d(int(user_dim/4)),  # 逐批标准化层
            # nn.Dropout(0.1),  # Dropout层，按照args.G_drop2的参数进行随机失活

            nn.Linear(int(user_dim), 64),  # 全连接层
            nn.Sigmoid()  # Sigmoid激活函数
        )


        self.net2 = nn.Sequential(
            nn.Linear(item_dim, int(item_dim * 2)),  # 全连接层，输入维度为dim，输出维度为
            nn.LeakyReLU(True),  # LeakyReLU激活函数
            nn.BatchNorm1d(int(item_dim * 2)),  # 逐批标准化层
            nn.Dropout(0.2),  # Dropout层，按照args.G_drop1的参数进行随机失活

            nn.Linear(int(item_dim * 2), int(item_dim)),  # 全连接层，输入维度为
            nn.LeakyReLU(True),  # LeakyReLU激活函数
            # nn.BatchNorm1d(int(item_dim/4)),  # 逐批标准化层
            nn.Dropout(0.2),  # Dropout层，按照args.G_drop2的参数进行随机失活

            # nn.Linear(int(item_dim), int(item_dim/2)),  # 全连接层，输入维度为
            # nn.LeakyReLU(True),  # LeakyReLU激活函数
            # #nn.BatchNorm1d(int(item_dim/4)),  # 逐批标准化层
            # nn.Dropout(0.1),  # Dropout层，按照args.G_drop2的参数进行随机失活

            nn.Linear(int(item_dim), 64),  # 全连接层
            nn.Sigmoid()  # Sigmoid激活函数
        )

    def mm(self, x, y):
        if args.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)

    def weight(self, graph, weights):
        graph_indices = graph._indices()
        graph_size = graph.size()
        # num_non_zero_elements = graph._nnz()
        graph_values = weights.data
        new_sparse_tensor = torch.sparse_coo_tensor(graph_indices, graph_values, graph_size)
        return new_sparse_tensor

    # def HGCN(self, ui_graph, iu_graph, uX, iX, num_layers = 2):
    #     uX_list = [uX]
    #     iX_list = [iX]
    #     for i in range(num_layers):
    #         if i == (num_layers-1):
    #             uX = self.softmax(torch.mm(ui_graph, iX))
    #             iX = self.softmax(torch.mm(iu_graph, uX))
    #         else:
    #             uX = torch.mm(ui_graph, iX)
    #             iX = torch.mm(iu_graph, uX)
    #         uX_list.append(uX)
    #         iX_list.append(iX)
    #     uX = torch.mean(torch.stack(uX_list), dim=0)
    #     iX = torch.mean(torch.stack(iX_list), dim=0)
    #     return uX, iX

    def forward(self, ui_graph, iu_graph):
        image_feats  = self.image_feats
        text_feats = self.text_feats

        image_item_feats, image_user_feats, _, _ = self.HGNN1(image_feats, self.HG, self.Hv2eimages_weight, self.He2vimages_weight,)  # _, _,

        image_ui_graph = self.weight(ui_graph, self.Hv2eimages_weight)
        image_user_id = self.mm(image_ui_graph, self.item_id_embedding.weight)

        image_iu_graph = self.weight(iu_graph, self.He2vimages_weight)
        image_item_id = self.mm(image_iu_graph, self.user_id_embedding.weight)

        #image_user_id, image_item_id = self.HGCN(image_ui_graph, image_iu_graph, image_user_id, image_item_id,  num_layers=1)


        text_item_feats, text_user_feats, _, _ = self.HGNN2(text_feats, self.HG, self.Hv2etexts_weight, self.He2vtexts_weight)  # _, _,
        text_ui_graph = self.weight(ui_graph, self.Hv2etexts_weight)
        text_user_id = self.mm(text_ui_graph, self.item_id_embedding.weight)

        text_iu_graph = self.weight(iu_graph, self.He2vtexts_weight)
        text_item_id = self.mm(text_iu_graph, self.user_id_embedding.weight)

        #text_user_id, text_item_id = self.HGCN(text_ui_graph, text_iu_graph, text_user_id, text_item_id,  num_layers=1)


        user_emb = (image_user_id + text_user_id) / 2
        item_emb = (image_item_id + text_item_id) / 2

        u_g_embeddings = self.user_id_embedding.weight + args.id_cat_rate*F.normalize(user_emb, p=2, dim=1)# + 0.1 * F.normalize(image_user_concept, p=2, dim=1) + 0.1 * F.normalize(text_user_concept, p=2, dim=1)#args.id_cat_rate*F.normalize(user_emb, p=2, dim=1) + self.user_id_embedding.weight
        i_g_embeddings = self.item_id_embedding.weight + args.id_cat_rate*F.normalize(item_emb, p=2, dim=1)# + 0.1 * F.normalize(image_item_concept, p=2, dim=1) + 0.1 * F.normalize(image_item_concept, p=2, dim=1)#args.id_cat_rate*F.normalize(item_emb, p=2, dim=1) + self.item_id_embedding.weight

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(2):#self.layers):
            if i == (self.n_ui_layers-1):
                u_g_embeddings = self.softmax( torch.mm(ui_graph, i_g_embeddings) )
                i_g_embeddings = self.softmax( torch.mm(iu_graph, u_g_embeddings) )

            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)

        u_g_embeddings = u_g_embeddings + args.model_cat_rate*F.normalize(image_user_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_user_feats, p=2, dim=1)  # 特征h
        i_g_embeddings = i_g_embeddings + args.model_cat_rate*F.normalize(image_item_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_item_feats, p=2, dim=1)  # 特征h


        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, u_g_embeddings, i_g_embeddings, image_user_id, text_user_id, image_item_id, text_item_id

    # def process_batches(self, image_feats, text_feats, ui_type):
    #     num_instances = image_feats.shape[0]  # 计算行数
    #     permutation = np.random.permutation(num_instances)  # 随机打乱行顺序
    #     output_dim = None  # 用于存储输出的维度，将在第一次迭代时设置
    #     all_outputs = []  # 用于收集所有批次的输出
    #
    #     # 分批处理
    #     batch_size = 1024
    #     for start_idx in range(0, num_instances, batch_size):
    #         end_idx = min(start_idx + batch_size, num_instances)
    #         batch_indices = permutation[start_idx:end_idx]
    #         batch_image_feats = image_feats[batch_indices]
    #         batch_text_feats = text_feats[batch_indices]
    #         if ui_type == 'user':
    #             batch_output = self.user_attention(batch_image_feats.unsqueeze(0), batch_text_feats.unsqueeze(0))  # 调用处理函数并获取输出
    #         elif ui_type == 'item':
    #             batch_output = self.item_attention(batch_image_feats.unsqueeze(0), batch_text_feats.unsqueeze(0))
    #
    #         # 确保所有批次的输出维度一致
    #         if output_dim is None:
    #             output_dim = batch_output.shape[1]
    #         elif batch_output.shape[1] != output_dim:
    #             raise ValueError("Inconsistent output dimensions from self.user_attention")
    #
    #         all_outputs.append(batch_output)  # 收集当前批次的输出
    #
    #     # 合并所有批次的输出，并根据原始的permutation顺序重新排列
    #     concatenated_outputs = torch.cat(all_outputs, axis=0)
    #     inv_permutation = np.argsort(permutation)  # 获取原始顺序的索引
    #     final_output = concatenated_outputs[inv_permutation]  # 重新排列输出以匹配原始顺序
    #
    #     return final_output  # 返回最终的输出，shape为（num_instances，dim）