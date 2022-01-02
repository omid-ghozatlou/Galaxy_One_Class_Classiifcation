from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import skimage.measure 
import logging
import time
import torch
import torch.optim as optim
import numpy as np
from kymatio import Scattering2D


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 200, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, normal_class=0,outlier_class=1):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.normal_classes = tuple([normal_class])
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.scattering = Scattering2D(J=2, shape=(32, 32))
        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        # _, val_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        _, _, apply_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c %.3f initialized.' % self.c[0] )

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()

        '''
        #load model if it exists
        if os.path.isfile('./net_model.pt'):
            load_path = './net_model.pt'
            net.load_state_dict(torch.load(load_path, map_location='cpu'))
            logger.info('Model from ./net_model.pt loaded')
        '''
        loss_epochs = list()
        loss_val_epochs = list()
        for epoch in range(self.n_epochs):


            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, labels, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                # outputs = net(torch.reshape(self.scattering(inputs), (5,243,8, 8)))
                if labels[0] == 0:
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                # else:
                #     dist = 1/(torch.sum((outputs - self.c) ** 2, dim=1))
                # dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    loss = torch.mean(dist)

                    loss.backward()
                    optimizer.step()
                    loss_epoch += loss.item()
                    n_batches += 1
            loss_epochs.append(loss_epoch / n_batches)
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

            scheduler.step()
            # val_loss = 0.0
            # n_apply = 0
            # net.eval()
            # with torch.no_grad():
            #     for data in val_loader:
            #         inputs, labels, idx = data
            #         inputs = inputs.to(self.device)
            #         outputs = net(inputs)
            #         # outputs = net(torch.reshape(self.scattering(inputs), (5,243,8, 8)))
            #         if labels[0] == 0:
            #             dist = torch.sum((outputs - self.c) ** 2, dim=1)
            #         # else:
            #         #     dist = 1/(torch.sum((outputs - self.c) ** 2, dim=1))
            #             val_loss += torch.mean(dist)
            #             n_apply += 1
            # loss_val_epochs.append(val_loss / n_apply)

        # for epoch in range(2):


        #     if epoch in self.lr_milestones:
        #         logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

        #     loss_epoch = 0.0
        #     n_batches = 0
        #     epoch_start_time = time.time()
        #     for data in train_loader:
        #         inputs, labels, _ = data
        #         inputs = inputs.to(self.device)

        #         # Zero the network parameter gradients
        #         optimizer.zero_grad()

        #         # Update network parameters via backpropagation: forward + backward + optimize
        #         outputs = net(inputs)
        #         if labels[0] == 1:
        #             dist = 1/(torch.sum((outputs - self.c) ** 2, dim=1))
        #         # else:
        #         #     dist = torch.sum((outputs - self.c) ** 2, dim=1)
        #         # dist = torch.sum((outputs - self.c) ** 2, dim=1)
        #             loss = torch.mean(dist)

        #             loss.backward()
        #             optimizer.step()
        #             loss_epoch += loss.item()
        #             n_batches += 1
        #     loss_epochs.append(loss_epoch / n_batches)
        #     # log epoch statistics
        #     epoch_train_time = time.time() - epoch_start_time
        #     logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
        #                 .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        #     scheduler.step()
        #     val_loss = 0.0
        #     n_apply = 0
        #     net.eval()
        #     with torch.no_grad():
        #         for data in val_loader:
        #             inputs, labels, idx = data
        #             inputs = inputs.to(self.device)
        #             outputs = net(inputs)
        #             if labels[0] == 1:
        #                 dist = 1/(torch.sum((outputs - self.c) ** 2, dim=1))
        #                 val_loss += torch.mean(dist)
        #                 n_apply += 1
        #     loss_val_epochs.append(val_loss / n_apply)
        # for epoch in range(self.n_epochs):


        #     if epoch in self.lr_milestones:
        #         logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

        #     loss_epoch = 0.0
        #     n_batches = 0
        #     epoch_start_time = time.time()
        #     for data in train_loader:
        #         inputs, labels, _ = data
        #         inputs = inputs.to(self.device)

        #         # Zero the network parameter gradients
        #         optimizer.zero_grad()

        #         # Update network parameters via backpropagation: forward + backward + optimize
        #         outputs = net(inputs)
        #         if labels[0] == 0:
        #             dist = torch.sum((outputs - self.c) ** 2, dim=1)
        #         # else:
        #         #     dist = 1/(torch.sum((outputs - self.c) ** 2, dim=1))
        #         # dist = torch.sum((outputs - self.c) ** 2, dim=1)
        #             loss = torch.mean(dist)

        #             loss.backward()
        #             optimizer.step()
        #             loss_epoch += loss.item()
        #             n_batches += 1
        #     loss_epochs.append(loss_epoch / n_batches)
        #     # log epoch statistics
        #     epoch_train_time = time.time() - epoch_start_time
        #     logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
        #                 .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        #     scheduler.step()
        #     val_loss = 0.0
        #     n_apply = 0
        #     net.eval()
        #     with torch.no_grad():
        #         for data in val_loader:
        #             inputs, labels, idx = data
        #             inputs = inputs.to(self.device)
        #             outputs = net(inputs)
        #             if labels[0] == 0:
        #                 dist = torch.sum((outputs - self.c) ** 2, dim=1)
        #             # else:
        #             #     dist = 1/(torch.sum((outputs - self.c) ** 2, dim=1))
        #                 val_loss += torch.mean(dist)
        #                 n_apply += 1
        #     loss_val_epochs.append(val_loss / n_apply)    
        # plot losses                
        # plt.plot(loss_epochs, label='Training Loss')
        # plt.plot(loss_val_epochs, label='Validation Loss')
        # plt.legend()
        # plt.savefig('D:/Omid/UPB/SVM/Galaxy-classification-master_0/imgs/scattering/class6/plot_loss_2.png' )
        # plt.close()
        
        # for epoch in range(self.n_epochs):


        #     if epoch in self.lr_milestones:
        #         logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            
        #     loss_epoch = 0.0
        #     n_batches = 0
        #     epoch_start_time = time.time()
        #     for data in train_loader:
        #         inputs, labels, _ = data
        #         inputs = inputs.to(self.device)

        #         # Zero the network parameter gradients
        #         optimizer.zero_grad()

        #         # Update network parameters via backpropagation: forward + backward + optimize
        #         outputs = net(inputs)
        #         dist = 1/ (torch.sum((outputs - self.c) ** 2, dim=1))              
        #         loss = torch.mean(dist)
        #         loss.backward()
        #         optimizer.step()

        #         # Update hypersphere radius R on mini-batch distances
        #         if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
        #             self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

        #         loss_epoch += loss.item()
        #         n_batches += 1

        #     # log epoch statistics
        #     epoch_train_time = time.time() - epoch_start_time
        #     logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
        #                 .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        #     scheduler.step()
            
        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, _,apply_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in apply_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                # outputs = net(torch.reshape(self.scattering(inputs), (200,243,8, 8)))
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        
        # roc curve for models
        fpr, tpr, thresh = roc_curve(labels, scores, pos_label=1)
        self.labels = labels
        self.scores = scores
        self.fpr = np.array(fpr)
        self.tpr = np.array(tpr)
        self.thresh = np.array(thresh)
        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        # matplotlib
        plt.style.use('seaborn')
        
        # plot roc curves
        plt.plot(fpr, tpr, linestyle='--',color='orange', label=' %g epochs' % self.n_epochs)
        # title
        plt.title('ROC curve')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')    
        plt.legend(loc='best')
        plt.savefig('D:/Omid/UPB/SVM/Galaxy-classification-master_0/imgs/scattering/class6/plot_ROC_2-test.png' )
        plt.close()
        logger.info('Finished testing.')

    # extra section added for applying model to unlabelled data
    def apply_model(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get apply_model data loader
        _, _, apply_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Applying model
        logger.info('Starting Deep SVDD application.')
        start_time = time.time()
        idx_score = []
        net.eval()
        with torch.no_grad():
            for data in apply_loader:
                inputs, nolabels, idx = data  # nolables are NaN
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                # outputs = net(torch.reshape(self.scattering(inputs), (5,243,8, 8)))
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_score += list(zip(idx.cpu().data.numpy().tolist(),
                                      nolabels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.apply_time = time.time() - start_time
        logger.info('Deep SVDD application time: %.3f' % self.apply_time)
        # Compute AUC
        _, nolabels, scores = zip(*idx_score)
        nolabels = np.array(nolabels)
        scores = np.array(scores)

        self.nolabels = nolabels
        self.scores = scores
        from sklearn.metrics import roc_curve
        # roc curve for models
        fpr, tpr, thresh = roc_curve(nolabels, scores, pos_label=1)
        # print ('FPR %s \t TPR %s\t Threshold %s \t' %fpr %tpr %thresh)
        
        self.apply_auc = roc_auc_score(nolabels, scores)
        logger.info('Apply set AUC: {:.2f}%'.format(100. * self.apply_auc))
        ind,_, scores = zip(*idx_score)
        self.ind = np.array(ind)
        self.scores = np.array(scores)
        self.fpr = np.array(fpr)
        self.tpr = np.array(tpr)
        self.thresh = np.array(thresh)
        # matplotlib
        plt.style.use('seaborn')
        
        # plot roc curves
        plt.plot(fpr, tpr, linestyle='--',color='orange', label=' %g epochs' % self.n_epochs)
        # title
        plt.title('ROC curve')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')    
        plt.legend(loc='best')
        plt.savefig('D:/Omid/UPB/SVM/Galaxy-classification-master_0/imgs/scattering/class6/plot_ROC_2.png' )
        plt.close()
        # return fpr,tpr,thresh
        # plt.savefig('ROC',dpi=300)
        # plt.show();
        logger.info('Finished Deep SVDD application.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        # logger = logging.getLogger()
        entropies=[]
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, labels, _ = data
                # logger.info('  Label %s' % labels[0])
                # logger.info('  norm %s' % self.normal_classes)
                if labels[0] == 0:
                    inputs = inputs.to(self.device)
                    # outputs = net(inputs)
                    # outputs = net(torch.reshape(self.scattering(inputs), (5,243,8, 8)))
                    # n_samples += outputs.shape[0]
                    # c += torch.sum(outputs, dim=0)
                    entropies.append(skimage.measure.shannon_entropy((inputs)))
        entropy_avr = np.mean(entropies)
        entropy_std = np.std(entropies)            
        c /= n_samples
        print('input max & min:%s & %s' %(np.max((inputs).tolist()) ,np.min((inputs).tolist())))
        # print(entropy_avr,entropy_std)
        # print('Scattered input max & min:%s & %s' %(np.max(self.scattering(inputs).tolist()) ,np.min(self.scattering(inputs).tolist())))
        # print('\n output max & min:%s & %s' %(np.max((outputs).tolist()) ,np.min((outputs).tolist())))
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
