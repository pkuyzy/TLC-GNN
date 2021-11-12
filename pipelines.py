import os
import torch
import loaddatas as lds
import torch.nn.functional as F
import numpy as np
from baselines import TLCGNN as TLCGNN
from sklearn.metrics import roc_auc_score,average_precision_score
from torch.nn.init import xavier_normal_ as xavier

def train():
    model.train()
    optimizer.zero_grad()
    emb = model.encode(data)
    x, y = model.decode(data, emb)
    loss = F.binary_cross_entropy(x,y)
    loss.backward()
    optimizer.step()
    return x

def test():
    model.eval()
    accs = []
    emb = model.encode(data)
    for type in ["val", "test"]:
        pred,y = model.decode(data,emb,type=type)
        pred,y = pred.cpu(),y.cpu()
        if type == "val":
            accs.append(F.binary_cross_entropy(pred, y))
            pred = pred.data.numpy()
            roc = roc_auc_score(y, pred)
            accs.append(roc)
            acc = average_precision_score(y,pred)
            accs.append(acc)
        else:
            pred = pred.data.numpy()
            roc = roc_auc_score(y, pred)
            accs.append(roc)
            acc = average_precision_score(y, pred)
            accs.append(acc)
    return accs

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        xavier(m.weight)
        if not m.bias is None:
            torch.nn.init.constant_(m.bias, 0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


#d_names = ['PPI']; times=range(20)
d_names = ["Photo", "PubMed", "Computers"]; times=range(50)


wait_total= 200
total_epochs = 2000


pipelines=['TLCGNN']
pipeline_acc={'TLCGNN':[i for i in times]}
pipeline_acc_sum={'TLCGNN':0}
pipeline_roc={'TLCGNN':[i for i in times]}
pipeline_roc_sum={'TLCGNN':0}
pipeline_acc_same={'TLCGNN':[i for i in times]}
pipeline_acc_same_sum={'TLCGNN':0}
pipeline_roc_same={'TLCGNN':[i for i in times]}
pipeline_roc_same_sum={'TLCGNN':0}
pipeline_acc_diff={'TLCGNN':[i for i in times]}
pipeline_acc_diff_sum={'TLCGNN':0}
pipeline_roc_diff={'TLCGNN':[i for i in times]}
pipeline_roc_diff_sum={'TLCGNN':0}

if not os.path.exists("./scores"):
    os.mkdir("./scores")

for d_name in d_names:
    f2 = open('scores/pipe_benchmark_' + d_name + '_LP_scores.txt', 'w+')
    f2.write('{0:7} {1:7}\n'.format(d_name, 'TLCGNN'))
    f2.flush()
    dataset = lds.loaddatas(d_name)
    for data_cnt in times:
        for Conv_method in pipelines:
            if d_name in ['Rand_nnodes_github1000', 'PPI']:
                data = dataset[data_cnt]
            else:
                data = dataset[0]
            if d_name in ['Rand_nnodes_github1000']:
                data.x = data.x[:, :10]
            #data.x = torch.ones(data.x.size())
            index = [i for i in range(len(data.y))]
            if d_name != "PPI":
                model, data = locals()[Conv_method].call(data, dataset.name, data.x.size(1), dataset.num_classes,
                                                     data_cnt)
            else:
                model, data = locals()[Conv_method].call(data, 'PPI', data.x.size(1), dataset.num_classes,
                                                         data_cnt)
            model.apply(weights_init)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0)
            best_val_acc = test_acc_same = test_acc_diff = test_acc = 0.0
            best_val_roc = test_roc_same = test_roc_diff = test_roc = 0.0
            best_val_loss = np.inf
            # train and val/test
            wait_step = 0

            # train and test
            for epoch in range(1, total_epochs + 1):
                pred = train()
                val_loss, val_roc, val_acc, tmp_test_roc, tmp_test_acc = test()
                if val_roc >= best_val_roc:
                    test_acc = tmp_test_acc
                    test_roc = tmp_test_roc
                    best_val_acc = val_acc
                    best_val_roc = val_roc
                    best_val_loss = val_loss
                    wait_step = 0
                else:
                    wait_step += 1
                    if wait_step == wait_total:
                        print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc,
                              ', Max roc: ', best_val_roc)
                        break
            del model
            del data
            # print result

            pipeline_acc[Conv_method][data_cnt] = test_acc
            pipeline_roc[Conv_method][data_cnt] = test_roc

            log = 'Epoch: ' + str(
                total_epochs) + ', dataset name: ' + d_name + ', Method: ' + Conv_method + ' Test pr: {:.4f}, roc: {:.4f} \n'
            print((log.format(pipeline_acc[Conv_method][data_cnt], pipeline_roc[Conv_method][data_cnt])))
            #print(pred)

            f2.write('{}, {:.4f}, {:.4f}\n'.format(data_cnt, pipeline_acc[Conv_method][data_cnt],
                                                     pipeline_roc[Conv_method][data_cnt],))
            f2.flush()
    f2.write('{0:4} {1:4f}\n'.format('std', np.std(pipeline_acc[Conv_method])))
    f2.write('{0:4} {1:4f}\n'.format('mean', np.mean(pipeline_acc[Conv_method])))
    f2.write('{0:4} {1:4f}\n'.format('std', np.std(pipeline_roc[Conv_method])))
    f2.write('{0:4} {1:4f}\n'.format('mean', np.mean(pipeline_roc[Conv_method])))
    f2.close()
