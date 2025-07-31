from datetime import datetime
import math
import os
import random
import sys
from tqdm import tqdm
import dgl
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from utility.parser import parse_args
from Models import MHGR
from utility.batch_test import *
from utility.logging import Logger

args = parse_args()

#训练模型
class Trainer(object):
    def __init__(self, data_config):

        self.task_name = "%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset))
        self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset))
        #self.audio_feats = np.load(args.data_path + '{}/audio_feat.npy'.format(args.dataset))
        self.ui_graph = self.ui_graph_raw = pickle.load(open(args.data_path + args.dataset + '/train_mat', 'rb'))

        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]
        self.iu_graph = self.ui_graph.T
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(self.ui_graph, mean_flag=True))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))

        num_non_zero_elements = self.ui_graph_raw.nnz
        self.H_weight = self.iu_graph.data
        self.model = MHGR(self.n_users, self.n_items, num_non_zero_elements, self.emb_dim, self.weight_size, self.mess_dropout,
                           self.image_feats, self.text_feats, self.ui_graph, self.iu_graph)
        self.model = self.model.cuda()

        self.optimizer_D = optim.AdamW(
            [
                {'params': self.model.parameters()},
            ]
            , lr=self.lr)
        self.scheduler_D = self.set_lr_scheduler()

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=fac)
        return scheduler_D

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag * csr_mat * colsum_diag
        else:
            return rowsum_diag * csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  #

    def sampleTrainBatch_dgl(self, batIds, pos_id=None, g=None, g_neg=None, sample_num=None, sample_num_neg=None):

        sub_g = dgl.sampling.sample_neighbors(g.cpu(), {'user': batIds}, sample_num, edge_dir='out', replace=True)
        row, col = sub_g.edges()
        row = row.reshape(len(batIds), sample_num)
        col = col.reshape(len(batIds), sample_num)

        if g_neg == None:
            return row, col
        else:
            sub_g_neg = dgl.sampling.sample_neighbors(g_neg, {'user': batIds}, sample_num_neg, edge_dir='out',
                                                      replace=True)
            row_neg, col_neg = sub_g_neg.edges()
            row_neg = row_neg.reshape(len(batIds), sample_num_neg)
            col_neg = col_neg.reshape(len(batIds), sample_num_neg)
            return row, col, col_neg

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def gradient_penalty(self, D, xr, xf):

        LAMBDA = 0.3

        xf = xf.detach()
        xr = xr.detach()

        alpha = torch.rand(args.batch_size * 2, 1).cuda()
        alpha = alpha.expand_as(xr)

        interpolates = alpha * xr + ((1 - alpha) * xf)
        interpolates.requires_grad_()

        disc_interpolates = D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gp

    def weighted_sum(self, anchor, nei, co):

        ac = torch.multiply(anchor, co).sum(-1).sum(-1)
        nc = torch.multiply(nei, co).sum(-1).sum(-1)

        an = (anchor.permute(1, 0, 2)[0])
        ne = (nei.permute(1, 0, 2)[0])

        an_w = an * (ac.unsqueeze(-1).repeat(1, args.embed_size))
        ne_w = ne * (nc.unsqueeze(-1).repeat(1, args.embed_size))

        res = (args.anchor_rate * an_w + (1 - args.anchor_rate) * ne_w).reshape(-1, args.sample_num_ii,
                                                                                args.embed_size).sum(1)

        return res
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        # z1 = z1/((z1**2).sum(-1) + 1e-8)
        # z2 = z2/((z2**2).sum(-1) + 1e-8)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=1024):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / args.tau)  #

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))
                tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))

                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)

            refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
            between_sim = torch.cat(tmp_between_sim_list, dim=-1)

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() / (
                        refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:,
                                                               i * batch_size:(i + 1) * batch_size].diag()) + 1e-8))
            # a = between_sim[:, i * batch_size:(i + 1) * batch_size]
            # a = a.diag()  # 对角是同一用户不同embed的相似度
            # b = refl_sim.sum(1)
            # c = between_sim.sum(1)
            # d = refl_sim[:, i * batch_size:(i + 1) * batch_size]
            # d = d.diag()
            # e = a / (b + c - d + 1e-8)  # 同一用户比不同的用户，分母不同用户应该去掉相同计算的
            # #为什么不减between_sim的diag呢？不也包含了相同用户不同embed

            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model(self.ui_graph, self.iu_graph)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):
        now_time = datetime.now()
        run_time = datetime.strftime(now_time, '%Y_%m_%d__%H_%M_%S')

        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        line_var_loss, line_g_loss, line_d_loss, line_cl_loss, line_var_recall, line_var_precision, line_var_ndcg = [], [], [], [], [], [], []
        stopping_step = 0

        best_recall = 0
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            contrastive_loss = 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.
            self.gene_u, self.gene_real, self.gene_fake = None, None, {}
            self.topk_p_dict, self.topk_id_dict = {}, {}

            for idx in tqdm(range(n_batch)):
                self.model.train()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                G_ua_embeddings, G_ia_embeddings, G_image_item_embeds, G_text_item_embeds, G_image_user_embeds, G_text_user_embeds \
                    , G_user_emb, _, G_image_user_id, G_text_user_id, _, _ \
                    = self.model(self.ui_graph, self.iu_graph)


                G_u_g_embeddings = G_ua_embeddings[users]
                G_pos_i_g_embeddings  = G_ia_embeddings[pos_items]
                G_neg_i_g_embeddings = G_ia_embeddings[neg_items]
                G_batch_mf_loss, G_batch_emb_loss, G_batch_reg_loss = self.bpr_loss(G_u_g_embeddings,
                                                                                    G_pos_i_g_embeddings,
                                                                                    G_neg_i_g_embeddings)

                batch_loss = G_batch_mf_loss + G_batch_emb_loss + G_batch_reg_loss

                line_var_loss.append(batch_loss.detach().data)

                self.optimizer_D.zero_grad()
                batch_loss.backward(retain_graph=False)
                self.optimizer_D.step()

                loss += float(batch_loss)
                mf_loss += float(G_batch_mf_loss)
                emb_loss += float(G_batch_emb_loss)
                reg_loss += float(G_batch_reg_loss)

            del G_ua_embeddings, G_ia_embeddings, G_u_g_embeddings, G_neg_i_g_embeddings, G_pos_i_g_embeddings

            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            # 分别输出self.model的.Hv2eimages_weight  .Hv2etexts_weight  .He2vimages_weight  .He2vtexts_weight值
            # print(self.model.Hv2eimages_weight)
            # print(self.model.Hv2etexts_weight)
            # print(self.model.He2vimages_weight)
            # print(self.model.He2vtexts_weight)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'].data)
            pre_loger.append(ret['precision'].data)
            ndcg_loger.append(ret['ndcg'].data)
            hit_loger.append(ret['hit_ratio'].data)

            line_var_recall.append(ret['recall'][1])
            line_var_precision.append(ret['precision'][1])
            line_var_ndcg.append(ret['ndcg'][1])

            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], ' \
                       'precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                        ret['recall'][1], ret['recall'][2],
                        ret['recall'][-1],
                        ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][-1],
                        ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][-1])
            self.logger.logging(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (
                eval(args.Ks)[1], test_ret['recall'][1], test_ret['precision'][1], test_ret['ndcg'][1]))
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                break
        self.save_model('./models')
        self.logger.logging(str(test_ret))

        return best_recall, run_time

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def save_model(self, path):
        """
        保存模型参数到指定路径
        """
        torch.save(self.model.state_dict(), path)
        self.logger.logging(f"Model saved to {path}")

    def predict(self, user_id, history, top_k=10, model_path=None):
        """
        基于 user_id 和其历史交互，预测推荐的 top_k 个物品

        :param user_id: int, 用户 ID
        :param history: list[int], 用户历史交互的 item ID 列表
        :param top_k: int, 推荐物品数量
        :param model_path: str, 可选模型路径，如果给出则从文件载入模型参数
        :return: list[int], 推荐的物品 ID 列表
        """
        self.model.eval()
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            self.logger.logging(f"Model loaded from {model_path}")

        with torch.no_grad():
            ua_embeddings, ia_embeddings, *_ = self.model(self.ui_graph, self.iu_graph)

            user_embed = ua_embeddings[user_id]  # shape: [embed_dim]

            # 可选：根据 history 做用户 embedding 的微调（如求平均）
            if history:
                item_embeds = ia_embeddings[history]
                user_embed = torch.mean(item_embeds, dim=0)

            scores = torch.matmul(ia_embeddings, user_embed)  # shape: [n_items]

            # 排除历史交互物品
            scores[history] = float('-inf')

            _, recommended_items = torch.topk(scores, top_k)
            return recommended_items.cpu().tolist()





def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


"""
We would like to acknowledge the open-source contributions of related research works. 
Our code framework has drawn inspiration from several GitHub repositories, 
including the following works: 
"Hypergraph Learning: Methods and Practices," 
"Multi-Modal Self-Supervised Learning for Recommendation," 
and "Latent Structure Mining with Contrastive Modality Fusion for Multimedia Recommendation."
"""
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    trainer = Trainer(data_config=config)
    trainer.train()