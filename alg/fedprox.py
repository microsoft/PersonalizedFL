# coding=utf-8
from alg.fedavg import fedavg
from util.traineval import train, train_prox


class fedprox(fedavg):
    def __init__(self, args):
        super(fedprox, self).__init__(args)

    def client_train(self, c_idx, dataloader, round):
        if round > 0:
            train_loss, train_acc = train_prox(
                self.args, self.client_model[c_idx], self.server_model, dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        else:
            train_loss, train_acc = train(
                self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc
