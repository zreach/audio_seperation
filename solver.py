from criterion import cal_loss
from myNet import TasNet
import os
import time
from tqdm import tqdm


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

    def train(self,epochs):
        for epoch in range(epochs):
            #train
            train_loss = self.run_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, train_loss))
            print('-' * 85)

            val_loss = self.run_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() , val_loss))
            print('-' * 85)

            self.train_loss[epoch] = train_loss
            self.val_loss[epoch] = val_loss


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
            if args.cuda:
                mixture = mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                source = source.cuda()

            pred_source = self.model(mixture,mixture_lengths,source)
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(source, pred_source, mixture_lengths)
            #这些哪些是有用的？

            if not val:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        all_loss +=loss.item()

        return all_loss / (i + 1)
            
            