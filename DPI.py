import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
import numpy as np
from data.datasets import twin_datasets,ihdp_datasets,twin_datasets_csv,jobs_datasets
from torch.utils.data import DataLoader
from models.network import EmbddingNet_1,EmbddingNet_2,TNet,YNet,OutNet_t0,OutNet_t1,T_YNet
import argparse
import random
import itertools
import matplotlib.pyplot as plt
import math
from utils.metrics import PEHE, ATE,Jobs_metric,ATT,ROL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)
def compute_mean_and_stddev(lst):
    n = len(lst)
    mean = sum(lst) / n
    stddev = math.sqrt(sum([(x - mean) ** 2 for x in lst]) / (n - 1))
    return mean, stddev

def mmd_loss(x, y):
    # 计算x和y的高斯核矩阵
    Kxx = gaussian_kernel(x, x)
    Kyy = gaussian_kernel(y, y)
    Kxy = gaussian_kernel(x, y)
    
    # 计算MMD值
    mmd = torch.mean(Kxx) - 2 * torch.mean(Kxy) + torch.mean(Kyy)
    
    return mmd

# 定义高斯核函数
def gaussian_kernel(x, y, sigma=1.0):
    # 计算每个样本与其他样本之间的距离平方
    x_norm_sq = torch.sum(x ** 2, dim=1, keepdim=True)
    y_norm_sq = torch.sum(y ** 2, dim=1, keepdim=True)
    dist_sq = torch.exp(- (x_norm_sq + y_norm_sq.t() - 2 * torch.matmul(x, y.t())) / (2 * sigma ** 2))
    
    return dist_sq

def mainTwins(args):
    # 构造数据集
    train_dataset = twin_datasets(isTrain=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    embedding_1= EmbddingNet_1(args.input_dims,args.hid_dims, 40).to(device)
    embedding_2 = EmbddingNet_2(args.input_dims,args.hid_dims, 40).to(device)
    tnet = TNet(args.input_dims,args.hid_dims, 1).to(device)
    t_ynet = T_YNet(args.input_dims,args.hid_dims, 40).to(device)
    ynet = YNet(41,args.hid_dims, 1).to(device)
    outnet_t0= OutNet_t0(args.hid_dims,1).to(device)
    outnet_t1= OutNet_t1(args.hid_dims,1).to(device)
    optimizer_P = Adam(itertools.chain(embedding_1.parameters(),embedding_2.parameters(),tnet.parameters(),t_ynet.parameters(),ynet.parameters(),outnet_t0.parameters(),outnet_t1.parameters()), lr=args.lr)
    min_pehe = 9999
    min_ate = 9999
    for epoch in range(args.epoch):
        epoch_pehe=9999
        epoch_ate=9999
        for steps, [train_x, train_t, train_y, train_potential_y] in enumerate(train_dataloader):
            train_x = train_x.float().to(device).squeeze()
            train_t = train_t.float().to(device).squeeze()
            train_y = train_y.float().to(device).squeeze()
            input_y0 = train_y[train_t == 0].float().to(device)
            input_y1 = train_y[train_t == 1].float().to(device)
            x0 = embedding_1(train_x)
            x1 = embedding_2(train_x)
            pred_T = tnet(x1)
            pred_T_Y =t_ynet(x1) 
            Y_T0= outnet_t0(pred_T_Y[train_t==0])
            Y_T1= outnet_t1(pred_T_Y[train_t==1])
            pred_Y = ynet(torch.cat((x0,pred_T),dim=1))
            # loss M
            loss_M=mmd_loss(x0,x1)
            # loss prediction
            loss_T= F.mse_loss(pred_T,train_t).to(device)

            loss_y =F.mse_loss(Y_T0,input_y0).to(device)+F.mse_loss(Y_T1,input_y1).to(device)+F.mse_loss(pred_Y,train_y).to(device)
            # total loss
            loss_other = args.rescon * loss_y+10*loss_M+loss_T
            optimizer_P.zero_grad()
            loss_other.backward()
            optimizer_P.step()
            if steps % args.print_steps == 0 or steps == 0:
                print(
                    "Epoches: %d, step: %d, loss_Y:%.3f,loss_M:%.3f,loss_T:%.3f"
                    % (epoch, steps, loss_y.detach().cpu().numpy(),loss_M.detach().cpu().numpy(),
                        loss_T.detach().cpu().numpy()))
                # ---------------------
                #         Test
                # ---------------------
                embedding_1.eval()
                embedding_2.eval()
                tnet.eval()
                t_ynet.eval()
                ynet.eval()
                outnet_t0.eval()
                outnet_t1.eval()
                total_test_potential_y = torch.Tensor([]).to(device)
                total_test_potential_y_hat = torch.Tensor([]).to(device)
                total_test_y_hat = torch.Tensor([]).to(device)
                total_test_y = torch.Tensor([]).to(device)
                test_dataset = twin_datasets(isTrain=False)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
                for steps, [test_x, test_t,test_y,test_potential_y] in enumerate(test_dataloader):
                    test_x = test_x.float().to(device).squeeze()
                    test_t = test_t.float().to(device).squeeze()
                    test_y = test_y.float().to(device)
                    test_potential_y= test_potential_y.float().to(device)
                    test_x0=embedding_1(test_x)
                    test_x1=embedding_2(test_x)
                    test_T_Y= t_ynet(test_x1)
                    test_T = tnet(test_x1)
                    test_Y_T0 = outnet_t0(test_T_Y)
                    test_Y_T1 = outnet_t1(test_T_Y)
                    test_Y = ynet(torch.cat((test_x0,test_T),dim=1))
                    test_potential_y_hat = torch.cat([test_Y_T0, test_Y_T1], dim=-1)
                    total_test_potential_y = torch.cat([total_test_potential_y, test_potential_y_hat], dim=0)
                    total_test_y_hat = torch.cat([test_Y, test_potential_y], dim=-1)
                    total_test_y = torch.cat([total_test_y, total_test_y_hat], dim=0)
                pehe = PEHE(total_test_potential_y.cpu().detach().numpy(),
                                total_test_y.cpu().detach().numpy())
                ate = ATE(total_test_potential_y.cpu().detach().numpy(),
                              total_test_y.cpu().detach().numpy())

                print("Train_PEHE:", pehe)
                print("Train_ATE:", ate)
                epoch_pehe= min(pehe, epoch_pehe)
                epoch_ate = min(ate, epoch_ate)
                min_pehe = min(pehe, min_pehe)
                min_ate = min(ate, min_ate)
                embedding_1.train()
                embedding_2.train()
                tnet.train()
                ynet.train()
                t_ynet.train()
                outnet_t0.train()
                outnet_t1.train()
        print("Test_PEHE:", min_pehe)
        print("Test_ATE:", min_ate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--input_dims", default=40, type=int)
    parser.add_argument("--hid_dims", default=40, type=int)
    parser.add_argument("--out_dims", default=1, type=int)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--print_steps", default=10, type=int)
    parser.add_argument("--lr", default=0.01, type=int)
    parser.add_argument("--rescon", type=int, default=10, help="weights of rescon loss")

    args = parser.parse_args()
    print(args)

    if (torch.cuda.is_available()):
        print("GPU is ready \n")
    else:
        print("CPU is ready \n")
    mainTwins(args)