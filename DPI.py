import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
import numpy as np
from data.datasets import twin_datasets,ihdp_datasets,twin_datasets_csv,jobs_datasets
from torch.utils.data import DataLoader
from models.network import EmbddingNet_1,EmbddingNet_2,TNet,YNet,OutNet_1,OutNet_2
import argparse
import random
import itertools
import matplotlib.pyplot as plt
import math
from utils.metrics import PEHE, ATE,Jobs_metric,ATT,ROL
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
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

def mainTwins(args):
    # 构造数据集
    train_dataset = twin_datasets(isTrain=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    embedding_1= EmbddingNet_1(args.input_dims,args.hid_dims, 40).to(device)
    embedding_2 = EmbddingNet_2(args.input_dims,args.hid_dims, 40).to(device)
    tnet = TNet(args.input_dims,args.hid_dims, 1).to(device)
    ynet = YNet(41,args.hid_dims, 1).to(device)
    optimizer_P = Adam(itertools.chain(embedding_1.parameters(),embedding_2.parameters(),tnet.parameters(),ynet.parameters()), lr=args.lr)
    min_pehe = 9999
    min_ate = 9999
    for epoch in range(args.epoch):
        epoch_pehe=9999
        epoch_ate=9999
        for steps, [train_x, train_t, train_y, train_potential_y] in enumerate(train_dataloader):
            train_x = train_x.float().to(device).squeeze()
            train_t = train_t.float().to(device).squeeze()
            train_y = train_y.float().to(device).squeeze()
            x0 = embedding_1(train_x)
            x1 = embedding_2(train_x)
            pred_T = tnet(x1)
            pred_Y = ynet(torch.cat((x0,train_t.unsqueeze(1)),dim=1))
            # loss M
            loss_M=torch.mean(torch.pow(x0 - torch.mean(x0), 2))+torch.mean(torch.pow(x1 - torch.mean(x1), 2))
            # loss prediction
            loss_T= F.mse_loss(pred_T,train_t).to(device)
            loss_y = F.mse_loss(pred_Y,train_y).to(device)
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
                ynet.eval()
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
                    test_T= tnet(test_x1)
                    test_Y = ynet(torch.cat((test_x0,test_t.unsqueeze(1)),dim=1))
                    test_potential_y_hat = torch.cat([test_T, test_Y], dim=-1)
                    total_test_potential_y = torch.cat([total_test_potential_y, test_potential_y_hat], dim=0)
                    total_test_y_hat = torch.cat([test_y, test_potential_y], dim=-1)
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