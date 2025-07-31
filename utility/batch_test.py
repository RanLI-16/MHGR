import utility.metrics as metrics  # 导入metrics模块
from utility.parser import parse_args  # 导入parse_args函数
from utility.load_data import Data  # 导入Data类
import multiprocessing  # 导入multiprocessing模块
import heapq  # 导入heapq模块
import torch  # 导入torch模块
import pickle  # 导入pickle模块
import numpy as np  # 导入numpy模块
from time import time  # 导入time模块

cores = multiprocessing.cpu_count() // 5  # 获取CPU核心数并除以5赋值给cores

args = parse_args()  # 解析命令行参数并赋值给args
Ks = eval(args.Ks)  # 将args.Ks转换为列表并赋值给Ks

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)  # 创建Data对象并赋值给data_generator
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items  # 获取数据生成器中的用户数和物品数并赋值给USR_NUM和ITEM_NUM
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test  # 获取数据生成器中的训练集大小和测试集大小并赋值给N_TRAIN和N_TEST
BATCH_SIZE = args.batch_size  # 将args.batch_size赋值给BATCH_SIZE

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]  # 遍历测试集中的物品，将评分存入item_score字典中

    K_max = max(Ks)  # 找出Ks中的最大值并赋值给K_max
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)  # 根据item_score的值进行堆排序，获取前K_max个物品并赋值给K_max_item_score

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)  # 如果物品i在用户行为测试集中，则添加1到r列表中
        else:
            r.append(0)  # 否则添加0到r列表中
    auc = 0.  # 初始化auc变量为0.0
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])  # 将item_score按值进行排序
    item_score.reverse()  # 将排序后的item_score列表反转
    item_sort = [x[0] for x in item_score]  # 提取排序后的item_score列表的物品编号并赋值给item_sort
    posterior = [x[1] for x in item_score]  # 提取排序后的item_score列表的评分并赋值给posterior

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)  # 如果物品i在用户行为测试集中，则添加1到r列表中
        else:
            r.append(0)  # 否则添加0到r列表中
    auc = metrics.auc(ground_truth=r, prediction=posterior)  # 使用metrics模块的auc函数计算AUC并赋值给auc
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]  # 遍历测试集中的物品，将评分存入item_score字典中

    K_max = max(Ks)  # 找出Ks中的最大值并赋值给K_max
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)  # 根据item_score的值进行堆排序，获取前K_max个物品并赋值给K_max_item_score

    r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)  # 调用ranklist_by_sorted函数计算r和auc
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []  # 初始化precision、recall、ndcg和hit_ratio列表为空

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))  # 计算不同K值下的precision并添加到precision列表中
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))  # 计算不同K值下的recall并添加到recall列表中
        ndcg.append(metrics.ndcg_at_k(r, K))  # 计算不同K值下的ndcg并添加到ndcg列表中
        hit_ratio.append(metrics.hit_at_k(r, K))  # 计算不同K值下的hit_ratio并添加到hit_ratio列表中

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}  # 返回一个字典，包含recall、precision、ndcg、hit_ratio和auc的值

def  test_one_user(x):
    # 用户u的评分 for user u
    is_val = x[-1]  # 判断是否为验证集
    rating = x[0]  # 用户u的评分
    # 用户u的ID
    u = x[1]  # 获取用户编号
    # 用户u在训练集中的物品
    try:
        training_items = data_generator.train_items[u]  # 获取用户u在训练集中的物品列表
    except KeyError:
        print(f"User {u} not found in training data.")
        # 返回一个表示错误或者忽略这个用户的值
        return None
    # 用户u在测试集中的物品
    if is_val:
        user_pos_test = data_generator.val_set[u]  # 如果是验证集，则将用户u在验证集中的行为作为用户行为测试集
    else:
        user_pos_test = data_generator.test_set[u]  # 否则将用户u在测试集中的行为作为用户行为测试集

    all_items = set(range(ITEM_NUM))  # 创建一个包含所有物品编号的集合

    test_items = list(all_items - set(training_items))  # 根据训练集中的物品列表，获取测试集中的物品列表

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)  # 如果test_flag为'part'，调用ranklist_by_heapq函数计算r和auc
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)  # 否则调用ranklist_by_sorted函数计算r和auc

    return get_performance(user_pos_test, r, auc, Ks)  # 调用get_performance函数计算precision、recall、ndcg、hit_ratio和auc的值

def test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}  # 初始化结果字典
    pool = multiprocessing.Pool(cores)  # 创建多进程池

    u_batch_size = BATCH_SIZE * 2  # 设置用户batch大小
    i_batch_size = BATCH_SIZE  # 设置物品batch大小

    test_users = users_to_test  # 获取要测试的用户列表
    n_test_users = len(test_users)  # 获取测试用户数量
    n_user_batchs = n_test_users // u_batch_size + 1  # 计算用户batch数量
    count = 0  # 初始化计数器为0

    for u_batch_id in range(n_user_batchs):  # 遍历用户batch
        start = u_batch_id * u_batch_size  # 计算当前用户batch的起始索引
        end = (u_batch_id + 1) * u_batch_size  # 计算当前用户batch的结束索引
        user_batch = test_users[start: end]  # 获取当前用户batch
        if batch_test_flag:
            n_item_batchs = ITEM_NUM // i_batch_size + 1  # 计算物品batch数量
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))  # 创建一个评分batch，初始值为0

            i_count = 0  # 初始化物品计数器为0
            for i_batch_id in range(n_item_batchs):  # 遍历物品batch
                i_start = i_batch_id * i_batch_size  # 计算当前物品batch的起始索引
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)  # 计算当前物品batch的结束索引

                item_batch = range(i_start, i_end)  # 获取当前物品batch
                u_g_embeddings = ua_embeddings[user_batch]  # 获取用户g的嵌入向量
                i_g_embeddings = ia_embeddings[item_batch]  # 获取物品g的嵌入向量
                i_rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))  # 计算评分batch

                rate_batch[:, i_start: i_end] = i_rate_batch  # 将评分batch赋值给rate_batch
                i_count += i_rate_batch.shape[1]  # 更新物品计数器

            assert i_count == ITEM_NUM  # 断言物品计数器是否等于ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)  # 获取所有物品batch
            u_g_embeddings = ua_embeddings[user_batch]  # 获取用户g的嵌入向量
            i_g_embeddings = ia_embeddings[item_batch]  # 获取物品g的嵌入向量
            rate_batch = torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1))

        rate_batch = rate_batch.detach().cpu().numpy()  # 将rate_batch转移到CPU上，并将其转换为numpy数组
        user_batch_rating_uid = zip(rate_batch, user_batch, [is_val] * len(user_batch))  # 将rate_batch、user_batch和is_val列表按照用户进行组合

        batch_result = pool.map(test_one_user, user_batch_rating_uid)  # 使用多线程池执行test_one_user函数，得到批量结果
        count += len(batch_result)  # 统计结果数量

        for re in batch_result:
            if re is None: # 如果结果为空，跳过# 遍历批量结果
                n_test_users-= 1  # 更新测试用户数量
                continue
            result['precision'] += re['precision'] / n_test_users  # 更新precision结果
            result['recall'] += re['recall'] / n_test_users  # 更新recall结果
            result['ndcg'] += re['ndcg'] / n_test_users  # 更新ndcg结果
            result['hit_ratio'] += re['hit_ratio'] / n_test_users  # 更新hit_ratio结果
            result['auc'] += re['auc'] / n_test_users  # 更新auc结果

    #assert count == n_test_users  # 断言结果数量是否与测试用户数量相等
    pool.close()  # 关闭多线程池
    return result  # 返回结果