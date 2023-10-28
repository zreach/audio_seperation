#!/usr/bin/env python

from criterion import cal_loss

import os
import time
from tqdm import tqdm
import torch

class Solver(object):
    def __init__(self,data,model,optimizer,args) -> None:
        self.train_loader = data['tr_loader']
        self.val_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer
        self.train_loss = []
        self.val_loss = []
        self.best_val_loss = float("inf")
        self.prev_val_loss = float("inf")
        self.args = args
    def train(self):
        for epoch in range(self.args.epochs):
            #train
            train_loss = self.run_epoch(self.args)

            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1,time.time(), train_loss))
            print('-' * 85)

            file_path = os.path.join(
                    self.args.save_folder, 'epoch_%d.pth.tar' % (epoch + 1))
            torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                tr_loss=self.train_loss,
                                                cv_loss=self.val_loss),
                           file_path)
            if(epoch>3):
                old_file_path = os.path.join(
                    self.args.save_folder, 'epoch_%d.pth.tar' % (epoch - 3))
                if os.path.exists(old_file_path):
                        # 删除文件
                    os.remove(old_file_path)
            val_loss = self.run_epoch(self.args,val=True)

            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() , val_loss))
            print('-' * 85)

            # self.train_loss[epoch] = train_loss
            # self.val_loss[epoch] = val_loss
        file_path = os.path.join(
                    self.args.save_folder, 'final.pth.tar')
        torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                tr_loss=self.train_loss,
                                                cv_loss=self.val_loss),
                           file_path)

    def run_epoch(self,args,val=0):
        all_loss = 0

        if val:
            dataloader = self.val_loader
            self.model.eval()
        else:
            dataloader = self.train_loader
            self.model.train()
        for i,data in tqdm(enumerate(dataloader)):
            mixture, mixture_lengths, source = data
            if args.use_cuda:
                mixture = mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                source = source.cuda()

            pred_source = self.model(mixture,mixture_lengths)
            loss,_,_,_ = cal_loss(source, pred_source, mixture_lengths)

            
            if not val:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        all_loss +=loss.item()

        return all_loss / (i + 1)
            
            