import os
os.environ['CUDA_VISIBLE_DEVICES']="9"
import torch
import loaddatas as lds
import torch.nn.functional as F
import random
import numpy as np
#from baselines import ConvCurv,ConvCurv_pers
#from baselines import ConvCurv_perslay as ConvCurv_pers
#from learnable_filter import ConvCurv as ConvCurv_pers
#from learnable_filter import ConvCurv_GIN as ConvCurv_pers
from Knowledge_Distillation import ConvCurv_GIN as ConvCurv_pers
#load the neural networks

def train(train_mask):
    model.train()
    #for parameters in model.modelGIN.lin1.parameters():
    #    print(parameters)
    optimizer.zero_grad()
    F.nll_loss(model(data)[train_mask], data.y[train_mask]).backward()
    optimizer.step()

def test(train_mask,val_mask,test_mask):
    model.eval()
    logits, accs = model(data), []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    accs.append(F.nll_loss(model(data)[val_mask], data.y[val_mask]))
    return accs

#load dataset
times=range(10)
wait_total=100
total_epochs = 200
pipelines=['ConvCurv_pers']
pipeline_acc={'ConvCurv_pers':[i for i in times]}
pipeline_acc_sum={'ConvCurv_pers':0}
d_names=[#'Cora',
         #'Citeseer',
         'PubMed',
         #'Photo'
         #'Physics',
        #'CS',
        #'Computers'
          ]
#d_names = ['Photo','CS','Physics','Computers']
#d_names=['Computers']
for d_name in d_names:
    if d_name == 'Photo' or d_name == 'Computers':
        wait_total = 200
        total_epochs = 500
    else:
        wait_total = 100
        total_epochs = 200
    f2=open('scores/pipe_benchmark_' +d_name+ '_GIN_new.txt', 'w+')
    f2.write('{0:7} {1:7}\n'.format(d_name,'ConvCurv'))
    f2.flush()
    if d_name=='Cora' or d_name=='Citeseer' or d_name=='PubMed':
        d_loader='Planetoid'
    elif d_name=='Computers' or d_name=='Photo':
        d_loader='Amazon'
    elif d_name == 'CS' or d_name == 'Physics':
        d_loader='Coauthor'
    else:
        d_loader = 'Synthesis'

    dataset=lds.loaddatas(d_loader,d_name)
    #already generated, no need to generate
    '''
    if not os.path.exists('./data/curvature/graph_'+d_name+'.edge_list'):
        print("start writing ricci curvature")
        Gd = nx.Graph()
        ricci_edge_index_ = np.array(dataset[0].edge_index)
        ricci_edge_index = [(ricci_edge_index_[0, i], ricci_edge_index_[1, i]) for i in
                            range(np.shape(dataset[0].edge_index)[1])]
        Gd.add_edges_from(ricci_edge_index)
        Gd_OT = OllivierRicci(Gd, alpha=0.5, method="OTD", verbose="INFO")
        print("adding edges finished")
        Gd_OT.compute_ricci_curvature()
        ricci_list = []
        for n1, n2 in Gd_OT.G.edges():
            ricci_list.append([n1, n2, Gd_OT.G[n1][n2]['ricciCurvature']])
            ricci_list.append([n2, n1, Gd_OT.G[n1][n2]['ricciCurvature']])
        ricci_list = sorted(ricci_list)
        ricci_file = open('./data/curvature/graph_'+d_name+'.edge_list','w')
        for ricci_i in range(len(ricci_list)):
            ricci_file.write(str(ricci_list[ricci_i][0]) + " " + str(ricci_list[ricci_i][1]) + " " + str(ricci_list[ricci_i][2]) + "\n")
        ricci_file.close()
        print("writing ricci curvature finished")
    '''
    for time in times:
        for Conv_method in pipelines:
            if d_loader != 'Synthesis':
                data=dataset[0]
            else:
                data = dataset[0]
                data.x = data.x[:, :10]
                data.x = torch.ones(data.x.size())
            index=[i for i in range(len(data.y))]
            if d_loader == 'Coauthor' or d_loader == 'Amazon':
                train_len=20*int(data.y.max()+1)
                train_mask=torch.tensor([i < train_len for i in index])
                val_mask=torch.tensor([i >= train_len and i < 500+train_len for i in index])
                test_mask=torch.tensor([i >= len(data.y)-1000 for i in index])
            elif d_loader == 'Planetoid':
                train_mask=data.train_mask.bool()
                val_mask=data.val_mask.bool()
                test_mask=data.test_mask.bool()
            else:
                random.shuffle(index)
                len_mul = int(1000 / 5)
                train_mask = torch.tensor([i < len_mul * 2 for i in index])
                val_mask = torch.tensor([(i >= len_mul * 2) and (i < len_mul * 4) for i in index])
                test_mask = torch.tensor([i >= (len(data.y) - len_mul) for i in index])
            model,data = locals()[Conv_method].call(data,dataset.name,data.x.size(1),dataset.num_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
            best_val_acc = test_acc = 0.0
            best_val_loss = np.inf
            wait_step = 0
            for epoch in range(1, total_epochs+1):
                train(train_mask)
                train_acc,val_acc,tmp_test_acc,val_loss = test(train_mask,val_mask,test_mask)
                if val_acc>=best_val_acc:
                    test_acc=tmp_test_acc
                    best_val_acc=val_acc
                    best_val_loss=val_loss
                    wait_step=0
                else:
                    wait_step += 1
                    if wait_step == wait_total:
                        print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc)
                        break
            del model
            del data
            pipeline_acc[Conv_method][time]=test_acc
            pipeline_acc_sum[Conv_method]=pipeline_acc_sum[Conv_method]+test_acc/len(times)
            log ='Epoch: ' +  str(total_epochs) + ', dataset name: '+ d_name + ', Method: '+ Conv_method + ' Test: {:.4f} \n'
            print((log.format(pipeline_acc[Conv_method][time])))
        f2.write('{0:4d} {1:4f}\n'.format(time,pipeline_acc[Conv_method][time]))
        f2.flush()
    f2.write('{0:4} {1:4f}\n'.format('std',np.std(pipeline_acc[Conv_method])))
    f2.write('{0:4} {1:4f}\n'.format('mean',np.mean(pipeline_acc[Conv_method])))
    f2.close()

    # delete evaluation of other models
    save_GIN_PI = "/data1/curvGN_LP/data/data/KD/predicted/" + d_name + "_temp.pt"
    if os.path.exists(save_GIN_PI):
        os.remove(save_GIN_PI)