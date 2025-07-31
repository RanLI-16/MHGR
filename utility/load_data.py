import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
from utility.parser import parse_args
from utility.hypergraph import Hypergraph
import pickle
args = parse_args()

def Gener_hyper(data_path, num_v, ui_graph, iu_graph):
    # # 用二进制读模式打开文件
    # with open(data_path + 'text_edge_list.pkl', 'rb') as f:
    #     # 使用pickle模块的load方法加载数据
    #     text_edge_list = pickle.load(f)
    #
    # with open(data_path + 'image_edge_list.pkl', 'rb') as f:
    #     # 使用pickle模块的load方法加载数据
    #     image_edge_list = pickle.load(f)

    # with open(data_path + 'user2item_list.pkl', 'rb') as f:
    #     # 使用pickle模块的load方法加载数据
    #     user2item_list = pickle.load(f)
    with open(args.data_path + args.dataset + '/train.json', 'r') as file:
        train = json.load(file)
    train = dict(sorted(train.items(), key=lambda item: int(item[0])))
    user2item_list = [tuple(value) for value in train.values()]

    # image_hg = Hypergraph(num_v, image_edge_list[0])
    # text_hg = Hypergraph(num_v, text_edge_list[0])

    hg = Hypergraph(num_v, user2item_list, merge_op="sum", ui_graph = ui_graph, iu_graph = iu_graph)
    # con_hg = Hypergraph(num_v, user2item_list, merge_op="sum")
    #
    # con_hg.add_hyperedges(text_edge_list[0], e_weight=text_edge_list[1], merge_op="sum", group_name='text')
    # con_hg.add_hyperedges(image_edge_list[0], e_weight=image_edge_list[1], merge_op="sum", group_name='image')

    print(hg)

    return hg#, con_hg, text_hg, image_hg

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path # 数据路径，包括训练集、测试集和验证集
        self.batch_size = batch_size

        train_file = path + '/train.json' # 训练集的json文件路径
        val_file = path + '/val.json' # 验证集的json文件路径
        test_file = path + '/test.json' # 测试集的json文件路径

        # 获取用户和物品的数量
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = [] # 用户id列表

        train = json.load(open(train_file)) # 加载训练集的json数据
        test = json.load(open(test_file)) # 加载测试集的json数据
        val = json.load(open(val_file)) # 加载验证集的json数据
        for uid, items in train.items():
            if len(items) == 0:
                continue
            uid = int(uid)
            self.exist_users.append(uid)  # 添加用户id到存在用户列表中
            self.n_items = max(self.n_items, max(items))  # 更新物品数量
            self.n_users = max(self.n_users, uid)  # 更新用户数量
            self.n_train += len(items)  # 训练集中总的交互数量

        for uid, items in test.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))  # 测试集中有无更大id的item
                self.n_test += len(items)  # 更新测试集中的交互数量
            except:
                continue

        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))  # 验证集中有无更大id的item
                self.n_val += len(items)  # 更新验证集中的交互数量
            except:
                continue

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # 构造稀疏矩阵R表示用户-物品交互情况
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)  # 构造稀疏矩阵表示物品之间的交互次数

        self.train_items, self.test_set, self.val_set = {}, {}, {}  # 分别存放训练集、测试集和验证集的用户-物品交互情况
        for uid, train_items in train.items():
            if len(train_items) == 0:
                continue
            uid = int(uid)
            for idx, i in enumerate(train_items):
                self.R[uid, i] = 1.

            self.train_items[uid] = train_items

        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0:
                continue
            try:
                self.test_set[uid] = test_items
            except:
                continue

        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0:
                continue
            try:
                self.val_set[uid] = val_items
            except:
                continue            

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz') # 尝试从磁盘中加载邻接矩阵
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz') # 尝试从磁盘中加载归一化邻接矩阵
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz') # 尝试从磁盘中加载平均邻接矩阵
            print('already load adj matrix', adj_mat.shape, time() - t1) # 打印邻接矩阵的形状和加载时间

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat() # 如果加载失败，则创建邻接矩阵
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat) # 将邻接矩阵保存到磁盘
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat) # 将归一化邻接矩阵保存到磁盘
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat) # 将平均邻接矩阵保存到磁盘
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        # 创建稀物品-疏矩阵adj_mat表示用户-用户和物品之间的邻接关系
        adj_mat = adj_mat.tolil() # 将adj_mat转换为lil_matrix格式
        R = self.R.tolil() # 将R转换为lil_matrix格式

        adj_mat[:self.n_users, self.n_users:] = R # 将R的元素赋值给adj_mat的前n_users行和第n_users行之后的元素
        adj_mat[self.n_users:, :self.n_users] = R.T # 将R的转置的元素赋值给adj_mat的第n_users行之后的行和前n_users列的元素
        adj_mat = adj_mat.todok() # 将adj_mat转换为dok_matrix格式
        print('already create adjacency matrix', adj_mat.shape, time() - t1) # 打印邻接矩阵的形状和创建时间

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1)) # 计算adj矩阵每行元素的和
            d_inv = np.power(rowsum, -1).flatten() # 计算rowsum的倒数
            d_inv[np.isinf(d_inv)] = 0. # 将inf值置为0
            d_mat_inv = sp.diags(d_inv) # 构造一个稀疏对角矩阵d_mat_inv

            norm_adj = d_mat_inv.dot(adj) # 计算归一化邻接矩阵
            # norm_adj = adj.dot(d_mat_inv) # 也可以使用归一化邻接矩阵
            print('generate single-normalized adjacency matrix.') # 打印归一化邻接矩阵的生成信息
            return norm_adj.tocoo() # 将归一化邻接矩阵转换为coo_matrix格式

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1)) # 计算adj矩阵每行元素的和
            d_inv = np.power(rowsum, -1).flatten() # 计算rowsum的倒数
            d_inv[np.isinf(d_inv)] = 0. # 将inf值置为0
            d_mat_inv = sp.diags(d_inv) # 构造一个稀疏对角矩阵d_mat_inv
            return d_mat_inv # 返回稀疏对角矩阵d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense()) # 将稀疏矩阵adj转换为密集矩阵
            degree = np.sum(dense_A, axis=1, keepdims=False) # 计算密集矩阵adj每行元素的和
            temp = np.dot(np.diag(np.power(degree, -1)), dense_A) # 计算一个邻接矩阵
            print('check normalized adjacency matrix whether equal to this laplacian matrix.') # 打印归一化邻接矩阵是否等于拉普拉斯矩阵
            return temp # 返回归一化邻接矩阵

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0])) # 计算单次归一化邻接矩阵
        mean_adj_mat = normalized_adj_single(adj_mat) # 计算平均归一化邻接矩阵

        print('already normalize adjacency matrix', time() - t2) # 打印归一化邻接矩阵的归一化时间和创建时间
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size) # 从存在用户列表中随机选择batch_size个用户
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)] # 从存在用户列表中随机选择batch_size个用户

        def sample_pos_items_for_u(u, num): # 根据用户u和数量num采样正样本物品
            pos_items = self.train_items[u] # 获取用户u的正样本物品列表
            n_pos_items = len(pos_items) # 正样本物品的数量
            pos_batch = [] # 正样本物品列表
            while True:
                if len(pos_batch) == num: break # 如果正样本物品数量达到num，则停止采样
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0] # 在正样本物品列表中随机选择一个id
                pos_i_id = pos_items[pos_id] # 获取选中的id对应的物品id

                if pos_i_id not in pos_batch: # 如果物品id不在正样本物品列表中，则将其加入
                    pos_batch.append(pos_i_id) # 加入正样本物品列表
            return pos_batch # 返回正样本物品列表

        def sample_neg_items_for_u(u, num):
            neg_items = []  # 初始化负样本物品列表
            while True:  # 进入循环
                if len(neg_items) == num: break  # 如果负样本物品数量等于所需数量，结束循环
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]  # 随机生成一个物品id
                if neg_id not in self.train_items[u] and neg_id not in neg_items:  # 如果该物品id不在用户u的训练物品中且不在负样本物品列表中
                    neg_items.append(neg_id)  # 将该物品id添加到负样本物品列表中
            return neg_items  # 返回负样本物品列表

        def sample_neg_items_for_u_from_pools(u, num):
            # 从neg_pools[u]中获取u没有出现过的物品的集合
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            # 从neg_items中随机选择num个物品作为负样本
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            # 对于每个用户，获取其一个正样本和一个负样本
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)

        # 返回用户列表、正样本列表和负样本列表
        return users, pos_items, neg_items

    def print_statistics(self):
        # 打印统计信息
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
        self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))
