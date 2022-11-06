import os
os.environ['CUDA_VISIBLE_DEVICES']="9"
import numpy as np
import torch
import pickle
from Knowledge_Distillation.Teacher_model import Teacher_Model
#from Knowledge_Distillation.Student_model import Student_Model
import random
from torch_geometric.utils import add_self_loops, remove_self_loops,degree
import torch.nn.functional as F
import datetime
import time


def load_dataset(name = 'Cora', filt = 'degree', hks_time = 10, type = 'LP'):
    if type == 'LP':
        type = "new_new"
    else:
        type = 'NC'
    if filt != 'hks':
        save_name = '/data1/curvGN_LP/data/data/KD/' + name + '_' + filt + '_' + type + '.pkl'
    else:
        save_name = '/data1/curvGN_LP/data/data/KD/' + name + '_' + filt + str(hks_time) + '_' +  type + '.pkl'
    with open(save_name, 'rb') as f:
        dict_save = pickle.load(f)
    return dict_save



def train():
    model.train()
    Total_loss_0 = 0
    Total_loss_PI = 0
    Total_loss_xy0 = 0; Total_loss_xd0 = 0; Total_loss_yd0 = 0
    cnt_sample = 0
    optimizer.zero_grad()
    for sample in train_sample:
        data = dict_save[sample]
        if len(data) <= 2:
            continue
        PD, PI = torch.FloatTensor(np.array(data[0].tolist() + data[1].tolist())).cuda(), torch.FloatTensor(data[2]).cuda()
        filt_value, edge_index = torch.FloatTensor(data[3]).cuda().view(-1, 1), torch.LongTensor(data[4]).cuda()
        edge_index = remove_self_loops(edge_index)[0]
        edge_index = add_self_loops(edge_index, num_nodes=len(filt_value))[0]

        # ot.emd cannot compute loss for large graph
        if PD.size()[0] > 30000:
            continue

        #x0, x, loss_0, loss_xy0, loss_xd0, loss_yd0, _, _, _ = model(filt_value, edge_index, PD, p=p, kernel=kernel)
        x0, x, loss_0, loss_xy0, loss_xd0, loss_yd0, _, _ = model(filt_value, edge_index, PD, p=p, kernel=kernel, grad_PI = False)

        #PI = F.normalize(PI, dim = -1)
        loss_PI = Loss(x, PI)
        loss = loss_0 + loss_PI
        Total_loss_0 += loss_0.cpu().detach()
        Total_loss_PI += loss_PI.cpu().detach()
        Total_loss_xy0 += loss_xy0.cpu().detach()
        Total_loss_xd0 += loss_xd0.cpu().detach()
        Total_loss_yd0 += loss_yd0.cpu().detach()
        cnt_sample += 1
        loss.backward()
        if cnt_sample % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        if cnt_sample % loss_interval == 0:
            print("Data: {}, filt: {}, Sample: {}, Train 0-dim Loss: {}, PI loss: {}".format(d_name, filt, cnt_sample, Total_loss_0 / cnt_sample,
                                                                                            Total_loss_PI / cnt_sample))


    model(filt_value, edge_index, PD, p=p, kernel=kernel, draw_fig = True, fig_name = d_name + "_" + filt)

    return Total_loss_0 / cnt_sample, Total_loss_PI / cnt_sample, \
           Total_loss_xy0 / cnt_sample, Total_loss_xd0 / cnt_sample, Total_loss_yd0 / cnt_sample



def test():
    model.eval()
    Total_loss_0 = 0
    Total_loss_PI = 0
    Total_loss_xy0 = 0; Total_loss_xd0 = 0; Total_loss_yd0 = 0
    cnt_sample = 0
    for sample in test_sample:
        with torch.no_grad():
            data = dict_save[sample]
            if len(data) <= 2:
                continue
            PD, PI = torch.FloatTensor(np.array(data[0].tolist() + data[1].tolist())).cuda(), torch.FloatTensor(data[2]).cuda()
            filt_value, edge_index = torch.FloatTensor(data[3]).cuda().view(-1, 1), torch.LongTensor(data[4]).cuda()
            edge_index = remove_self_loops(edge_index)[0]
            edge_index = add_self_loops(edge_index, num_nodes=len(filt_value))[0]

            # ot.emd cannot compute loss for large graph
            if PD.size()[0] > 30000:
                continue

            x0, x, loss_0, loss_xy0, loss_xd0, loss_yd0, _, _ = model( filt_value, edge_index, PD, p = p, kernel = kernel, pair_diagonal = True, grad_PI = False)

            #PI = F.normalize(PI, dim=-1)
            loss_PI = Loss(x, PI)
            Total_loss_0 += loss_0.cpu().detach()
            Total_loss_PI += loss_PI.cpu().detach()
            Total_loss_xy0 += loss_xy0.cpu().detach()
            Total_loss_xd0 += loss_xd0.cpu().detach()
            Total_loss_yd0 += loss_yd0.cpu().detach()
            cnt_sample += 1
            if cnt_sample % loss_interval == 0:
                print("Data: {}, filt: {}, Sample: {}, Test 0-dim Loss: {}, PI loss: {}".format(d_name, filt, cnt_sample, Total_loss_0 / cnt_sample,
                                                                                            Total_loss_PI / cnt_sample))

    x0, x, loss_0, loss_xy0, loss_xd0, loss_yd0, _, _ =  model( filt_value, edge_index, PD, p = p, kernel = kernel, pair_diagonal = True, draw_fig = True, fig_name = d_name + "_" + filt)

    #print(x0)
    #print(PD0)
    #print(x1)
    #print(PD1)
    #print(x)
    #print(PI)
    return Total_loss_0 / cnt_sample, Total_loss_PI / cnt_sample, \
           Total_loss_xy0 / cnt_sample, Total_loss_xd0 / cnt_sample, Total_loss_yd0 / cnt_sample

def evaluate_time():
    # t1 is the time for generating PD, while t2 is to transform PD to PI
    model.eval()
    total_t = 0
    total_t1 = 0
    total_t2 = 0
    average_nodes = 0
    average_edges = 0
    for sample in dict_save.keys():
        with torch.no_grad():
            #print(sample)
            data = dict_save[sample]
            if len(data) <= 2:
                continue
            PD, PI = torch.FloatTensor(np.array(data[0].tolist() + data[1].tolist())).cuda(), torch.FloatTensor(
                data[2]).cuda()
            filt_value, edge_index = torch.FloatTensor(data[3]).cuda().view(-1, 1), torch.LongTensor(data[4]).cuda()
            edge_index = remove_self_loops(edge_index)[0]
            edge_index = add_self_loops(edge_index, num_nodes=len(filt_value))[0]
            t = time.time()
            _, _, _, _, _, _, t1, t2 = model(filt_value, edge_index, PD, p=p, kernel=kernel, pair_diagonal=True, compute_loss = False, grad_PI = False)
            total_t += (time.time() - t)
            total_t1 += t1; total_t2 += t2
            print(t1)
            average_nodes += filt_value.size()[0]; average_edges += edge_index.size()[1]
    average_nodes /= len(dict_save.keys())
    average_edges /= len(dict_save.keys())
    return total_t, total_t1, total_t2, average_nodes, average_edges


if __name__ == "__main__":
    #type = 'LP'
    type = 'NC'
    d_names = ['PubMed']#['photo']#['Cora'] # recommend every time only use a dataset
    #filts = ['hks_10', 'hks_0.1', 'ricci', 'degree']
    filts = ['clustering', 'centrality']
    epochs = 20
    learning_rate = 0.002
    weight_decay = 0.01
    dropout = 0
    loss_interval = 1000
    epoch_pair_diagonal = epochs # for epochs before, the gt PD do not pair with diagonal, after pair with
    num_models = 3
    seed = 1234
    batch_size = 10
    new_node_feat = True
    use_edge_attn = True
    max_loop_len = 10 # length of the initial input cycle vector
    #model_type = 'GCN'
    #model_type = 'PDGNN'
    model_type = 'GAT'
    #model_type = 'GIN'
    p = 2 # p-wasserstein distance
    kernel = 'wasserstein' # loss for PDs
    save_model = True
    train_prop = None

    model = Teacher_Model(hidden_dim = 32, type = model_type, num_models=num_models, dropout = dropout, max_loop_len = max_loop_len, new_node_feat = new_node_feat, use_edge_attn = use_edge_attn).cuda()
    #model = Student_Model(hidden_dim=32, type=model_type, num_models=num_models, dropout=dropout,
    #                      max_loop_len=max_loop_len).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    Loss = torch.nn.MSELoss()

    test_0_losses = []; test_PI_losses = []; test_Total_losses = []



    '''
    # evaluate time
    #d_names = ['Cora', 'Citeseer', 'PubMed', 'photo', 'computers', 'CS', 'Physics']
    d_names = ['computers']
    for d_name in d_names:
        if os.path.exists("/home/yzy/GNN/CurvGN-github/Knowledge_Distillation/models/" + d_name + ".pt"):
            model.load_state_dict(torch.load("/home/yzy/GNN/CurvGN-github/Knowledge_Distillation/models/" + d_name + ".pt"))
        else:
            model.load_state_dict(torch.load("/home/yzy/GNN/CurvGN-github/Knowledge_Distillation/models/photo.pt"))
        #with open('/data1/curvGN_LP/data/data/KD/' + d_name + '_ricci_test.pkl', 'rb') as f:
        with open('/data1/curvGN_LP/data/data/KD/' + d_name + '_ricci_hop3_test.pkl', 'rb') as f:
            dict_save = pickle.load(f)
        print(evaluate_time())
    '''
    '''
    # evaluate time for centrality and clustering coefficient
    d_names = ['Cora', 'Citeseer', 'PubMed']
    model.load_state_dict(torch.load("/home/yzy/GNN/CurvGN-github/Knowledge_Distillation/models/photo.pt"))
    for d_name in d_names:
        for filt in ['centrality', 'clustering']:
            with open('/data1/curvGN_LP/data/data/KD/' + d_name + '_' + filt + '_NC.pkl', 'rb') as f:
                dict_save = pickle.load(f)
            print(evaluate_time())
    '''

    # evaluate time for large sparse graphs and SBM
    #d_names = ['Cora', 'Citeseer', 'PubMed']
    d_names = ['SBM']
    model.load_state_dict(torch.load("/home/yzy/GNN/CurvGN-github/Knowledge_Distillation/models/photo.pt"))
    for d_name in d_names:
        with open('/data1/curvGN_LP/data/data/KD/' + d_name + '_density_degree_total_test.pkl', 'rb') as f:
            dict_save = pickle.load(f)
        print(evaluate_time())


    '''
    # evaluate on transferable
    #model.load_state_dict(torch.load("/home/yzy/GNN/CurvGN-github/Knowledge_Distillation/models/photo.pt"))
    for d_name in ['Cora', 'Citeseer', 'PubMed', 'photo', 'computers']:
        model.load_state_dict(torch.load("/home/yzy/GNN/CurvGN-github/Knowledge_Distillation/models/Cora.pt"))
        filt = 'ricci'
        dict_save = load_dataset(d_name, filt, 10, type)
        num_sample = sum([1 for key in dict_save.keys()])
        random.seed(seed)
        train_sample = random.sample(list(range(num_sample)), int(0.8 * num_sample))
        test_sample = list(set(list(range(num_sample))) - set(train_sample))
        train()
        PD0_loss, PI_loss, test_xy0, test_xd0, test_yd0 = test()
        print("dataset: {}, PD loss: {}, PI loss: {}".format(d_name, PD0_loss, PI_loss))
    '''




    '''
    # default training and evaluation
    for epoch in range(epochs):
        Total_train_0 = 0; Total_train_PI = 0; Total_train_samples = 0
        Total_train_xy0 = 0; Total_train_xd0 = 0; Total_train_yd0 = 0
        Total_test_0 = 0; Total_test_PI = 0; Total_test_samples = 0
        Total_test_xy0 = 0; Total_test_xd0 = 0; Total_test_yd0 = 0
        for d_name in d_names:
            for filt in filts:
                if filt == 'hks_10':
                    dict_save = load_dataset(d_name, 'hks', 10, type)
                elif filt == 'hks_0.1':
                    dict_save = load_dataset(d_name, 'hks', 0.1, type)
                else:
                    dict_save = load_dataset(d_name, filt, 10, type)
                num_sample = sum([1 for key in dict_save.keys()])
                random.seed(seed)
                train_sample = random.sample(list(range(num_sample)), int(0.8 * num_sample))
                test_sample = list(set(list(range(num_sample))) - set(train_sample))

                # evaluate influence of numbers
                if train_prop is not None:
                    train_sample = random.sample(train_sample, int(num_sample * train_prop ))

                #test()
                train_loss_0, train_loss_PI, train_xy0, train_xd0, train_yd0= train()
                PD0_loss, PI_loss, test_xy0, test_xd0, test_yd0 = test()

                if filt == 'ricci' and save_model and model_type == 'GAT' and new_node_feat:
                    if train_prop is None:
                        torch.save(model.state_dict(), './models/' + d_name + '.pt')
                    else:
                        torch.save(model.state_dict(), './models/' + d_name + '_' + str(train_prop) + '.pt')

                Total_train_0 += train_loss_0 * len(train_sample)
                Total_train_PI += train_loss_PI * len(train_sample); Total_train_samples += len(train_sample)
                Total_train_xy0 += train_xy0 * len(train_sample); Total_train_xd0 += train_xd0 * len(train_sample); Total_train_yd0 += train_yd0 * len(train_sample)
                Total_test_0 += PD0_loss * len(test_sample)
                Total_test_PI += PI_loss * len(test_sample); Total_test_samples += len(test_sample)
                Total_test_xy0 += test_xy0 * len(test_sample); Total_test_xd0 += test_xd0 * len(test_sample); Total_test_yd0 += test_yd0 * len(test_sample)
                print("Data: {}, filt: {}, Epoch: {}, Training 0-dim Loss: {}, PI loss: {}, Test 0-dim Loss: {}, PI loss: {}".format(d_name, filt, epoch, train_loss_0, train_loss_PI, PD0_loss, PI_loss))
        Total_train_0 /= Total_train_samples; Total_train_PI /= Total_train_samples
        Total_train_xy0 /= Total_train_samples; Total_train_xd0 /= Total_train_samples; Total_train_yd0 /= Total_train_samples
        Total_test_0 /= Total_test_samples; Total_test_PI /= Total_test_samples
        Total_test_xy0 /= Total_test_samples; Total_test_xd0 /= Total_test_samples; Total_test_yd0 /= Total_test_samples

        test_0_losses.append(Total_test_0)
        test_PI_losses.append(Total_test_PI)
        test_Total_losses.append(Total_test_0 + Total_test_PI)

        train_info = "Summary: Epoch: {}, Training 0-dim Loss: {}, xy loss: {}, xd loss: {}, yd loss: {}, PI loss: {}".format(epoch, Total_train_0, Total_train_xy0, Total_train_xd0, Total_train_yd0, Total_train_PI)
        test_info = "Test 0-dim Loss: {}, xy loss: {}, xd loss: {}, yd loss: {}, PI loss: {}".format(Total_test_0, Total_test_xy0, Total_test_xd0, Total_test_yd0, Total_test_PI)
        print(train_info)
        print(test_info)
        with open("result/" + model_type + ".txt", "a") as f:
            f.write(str(datetime.datetime.now())+ ": ")
            f.write(train_info)
            f.write("\n")
            f.write(test_info)
            f.write("\n")

    arg_min_loss = np.argmin(test_Total_losses)
    total_info = "Summary after all, Min Test 0-dim Loss: {}, PI loss: {}".format(test_0_losses[arg_min_loss], test_PI_losses[arg_min_loss])
    print(total_info)
    with open("result/" + model_type + ".txt", "a") as f:
        f.write(str(datetime.datetime.now()) + ": ")
        f.write(total_info)
        f.write("\n")
    '''

