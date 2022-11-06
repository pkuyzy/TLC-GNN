import os
os.environ['CUDA_VISIBLE_DEVICES']="5"
import torch
import learnable_filter.loaddatas_LP as lds
import torch.nn.functional as F
import numpy as np
#from baselines import ConvCurv_LP, gcn_LP
#from learnable_filter import gcn_LP_GIN as gcn_LP
#from learnable_filter import gcn_LP_GIN_requiregrad as gcn_LP
from Knowledge_Distillation import gcn_LP_GIN as gcn_LP
from sklearn.metrics import roc_auc_score,average_precision_score
from torch_geometric.utils import remove_self_loops
from sg2dgm import sg2dgm_LP as sg2dgm
import networkx as nx
from torch.nn.init import xavier_normal_ as xavier
#load the neural networks

def train():
    model.train()
    optimizer.zero_grad()
    emb = model.encode(data)
    #seed = 123 if epoch <= 100 else None
    x,y = model.decode(data,emb)
    #F.nll_loss(x,y).backward()
    F.binary_cross_entropy(x,y).backward()
    #print("x")
    #print(model.PI1.grad)
    #print(model.pers_img.grad)
    optimizer.step()
    return x

def test():
    model.eval()
    accs = []
    emb = model.encode(data)
    for type in ["train","val","test"]:
        pred,y = model.decode(data,emb,type=type)
        pred,y = pred.cpu(),y.cpu()
        pred = pred.data.numpy()
        #pred = np.int64(pred > 0.5)
        #print(type)
        #print(sum(pred==np.array(y))/len(pred))
        roc = roc_auc_score(y, pred)
        accs.append(roc)
        acc = average_precision_score(y,pred)
        accs.append(acc)
    val_x,val_y = model.decode(data,emb,type='val')
    accs.append(F.binary_cross_entropy(val_x, val_y))
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
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True



#load dataset
times=range(10)
wait_total= 200
total_epochs = 2000

'''
pipelines=['ConvCurv_LP']
pipeline_acc={'ConvCurv_LP':[i for i in times]}
pipeline_acc_sum={'ConvCurv_LP':0}
pipeline_roc={'ConvCurv_LP':[i for i in times]}
pipeline_roc_sum={'ConvCurv_LP':0}
'''
pipelines=['gcn_LP']
pipeline_acc={'gcn_LP':[i for i in times]}
pipeline_acc_sum={'gcn_LP':0}
pipeline_roc={'gcn_LP':[i for i in times]}
pipeline_roc_sum={'gcn_LP':0}



#d_names = [ 'Cora', 'Citeseer', 'PubMed', 'Photo', 'Computers']
#d_names = ['Cora', 'Citeseer','PubMed']
#d_names = ['Cora', 'Citeseer','Photo','PubMed', 'Computers']
#d_names = ['PPI']
#d_names = ['disease_lp']
d_names = [#"Cora", "Citeseer",
           #"PubMed",
           #"Photo",
           "Computers"
            ]

for d_name in d_names:
    f2=open('scores/pipe_benchmark_' +d_name+ '_LP_scores_GIN.txt', 'w+')
    f2.write('{0:7} {1:7}\n'.format(d_name,'ConvCurv'))
    f2.flush()
    if d_name=='Cora' or d_name=='Citeseer' or d_name=='PubMed':
        d_loader='Planetoid'
    elif d_name=='Computers' or d_name=='Photo':
        d_loader='Amazon'
    elif d_name == 'CS' or d_name == 'Physics':
        d_loader='Coauthor'
    else:
        d_loader = 'PPI'

    dataset=lds.loaddatas(d_loader,d_name)
    if d_loader == 'PPI':
        dataset.name = "PPI"
    for time in times:
        #setup_seed(1234)
        for Conv_method in pipelines:
            if d_loader not in ['PPI']:
                data=dataset[0]
            else:
                data = dataset[time]
                #data.x = data.x[:, :10]
                #data.x = torch.ones(data.x.size())
            index=[i for i in range(len(data.y))]
            model,data = locals()[Conv_method].call(data,dataset.name,data.x.size(1),dataset.num_classes, data_cnt = time)
            model.apply(weights_init)
            #optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.0005)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0)
            #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0, momentum = 0.8, nesterov=True)
            #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [wait_total], gamma=0.1, last_epoch=-1)
            best_val_acc = test_acc = 0.0
            best_val_roc = test_roc = 0.0
            best_val_loss = np.inf
            #train and val/test
            wait_step = 0

            #train and test
            for epoch in range(1, total_epochs+1):
                pred = train()
                train_roc,train_acc,val_roc,val_acc,tmp_test_roc,tmp_test_acc,val_loss = test()
                #scheduler.step()
                #if epoch % 200 == 0:
                #    print("epoch:{}, lr:{}, train_acc: {}, val_acc: {}, test_acc: {}, val_loss: {}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr'], train_acc, val_acc, tmp_test_acc, val_loss))
                if val_roc>=best_val_roc:
                    test_acc=tmp_test_acc
                    test_roc=tmp_test_roc
                    best_val_acc=val_acc
                    best_val_roc=val_roc
                    best_val_loss=val_loss
                    wait_step=0
                else:
                    wait_step += 1
                    if wait_step == wait_total:
                        print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc , ', Max roc: ', best_val_roc)
                        break
            #for name, parameters in model.named_parameters():
            #    print(name, ':', parameters)
            #print(model.state_dict()['linear_2.weight'])
            #print(model.state_dict()['linear_3.weight'])
            #print(torch.matmul(model.state_dict()['linear_3.weight'], model.state_dict()['linear_2.weight']))

            del model
            del data
            #print result

            pipeline_acc[Conv_method][time]=test_acc
            #pipeline_acc_sum[Conv_method]=pipeline_acc_sum[Conv_method]+test_acc/len(times)
            pipeline_roc[Conv_method][time] = test_roc
            #pipeline_roc_sum[Conv_method] = pipeline_roc_sum[Conv_method] + test_roc / len(times)
            log ='Epoch: ' + str(total_epochs) + ', dataset name: '+ d_name + ', Method: '+ Conv_method + ' Test acc: {:.4f}, roc: {:.4f} \n'
            print((log.format(pipeline_acc[Conv_method][time],pipeline_roc[Conv_method][time])))
            print(pred)

            f2.write('{0:4d} {1:4f} {2:4f}\n'.format(time,pipeline_acc[Conv_method][time], pipeline_roc[Conv_method][time]))
            f2.flush()
    f2.write('{0:4} {1:4f}\n'.format('std',np.std(pipeline_acc[Conv_method])))
    f2.write('{0:4} {1:4f}\n'.format('mean',np.mean(pipeline_acc[Conv_method])))
    f2.write('{0:4} {1:4f}\n'.format('std', np.std(pipeline_roc[Conv_method])))
    f2.write('{0:4} {1:4f}\n'.format('mean', np.mean(pipeline_roc[Conv_method])))
    f2.close()
