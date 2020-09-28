import wandb
import torch
from .archs import *
from .data.make_data import *
from .data.cluster_eval import *
import torch.nn.functional as F

class IIC_Pipeline():
    def __init__(self, config):
        super(IIC_Pipeline, self).__init__()
        self.config = config
        self.start_epoch = 0
        self.net = ClusterNetMulHead(config)
        self.net.to(config.device)
        self.iic_optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr_iic)
        self.simclr_optimizer = torch.optim.Adam(self.net.parameters(),lr=config.lr_simclr)
        self.iic_loss = IICLoss()
        self.simclr_loss = NTXentLoss(config.device, config.simclr_bs, 0.5, True)
        self.simclr_scheduler = None
        self.iic_scheduler = None
        if config.lr_gamma_iic is not None:
            self.iic_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.iic_optimizer, milestones=config.lr_milestones_iic, gamma=config.lr_gamma_iic)
    
    #Saves all state_dicts along with model parameteres and epoch number
    def save_checkpoint(self, epoch, path):
        """ Save a checkpoint of the training process on the given path. This includes the parameter setting of the model and the optimizer 
            as well as the epoch number.
        """
        check = {
            'epoch': epoch,
            'model': self.net.state_dict(),
            'iic_optimizer': self.iic_optimizer.state_dict(),
            'simclr_optimizer': self.simclr_optimizer.state_dict(),           
            'iic_scheduler' : None if (self.iic_scheduler is None) else self.iic_scheduler.state_dict(),
            'simclr_scheduler' : None if (self.simclr_scheduler is None) else self.simclr_scheduler.state_dict()
        }
        torch.save(check, path)

    #Loads state_dicts to automatically continue training from a certain epoch
    def load_checkpoint(self, path):
        """ Loads a checkpoint from a given file to resume training.
        """
        device = torch.device('cuda')
        check = torch.load(path, map_location=device)
        self.net.load_state_dict(check['model'])
        self.iic_optimizer.load_state_dict(check['iic_optimizer'])
        self.simclr_optimizer.load_state_dict(check['simclr_optimizer'])
        if check['iic_scheduler'] is not None:
            self.iic_scheduler.load_state_dict(check['iic_scheduler'])
        if check['simclr_scheduler'] is not None:
            self.simclr_scheduler.load_state_dict(check['simclr_scheduler'])
        self.start_epoch = check['epoch'] + 1

    def train_iic_head(self, head, dataloaders, n_epochs=1):
        """
        Trains a specific IIC head for a number of epochs and returns loss metrics
        """
        self.net.train()
        config = self.config
        avg_loss = 0.           # over heads and head_epochs (and sub_heads)
        avg_loss_no_lamb = 0.
        avg_loss_count = 0
        for head_i_epoch in range(n_epochs):
            iterators = (d for d in dataloaders)
            b_i = 0
            #For each batch
            for tup in zip(*iterators):
                self.iic_optimizer.zero_grad()
                #in_channels-1 because images haven't been sobeled yet
                all_imgs = torch.zeros(config.iic_bs, config.in_channels - 1, config.input_sz, config.input_sz).cuda()
                all_imgs_tf = torch.zeros(config.iic_bs, config.in_channels - 1, config.input_sz, config.input_sz).cuda()
                # Tup[0][0] and tup[0][1] are the lists of original images and label 
                # All subsequent tup[N][0] and tup[N][1] contains the same images with different random transforms applied
                imgs_curr = tup[0][0]
                curr_batch_sz = imgs_curr.size(0)
                for d_i in range(config.num_dataloaders):
                    #Iterates through all modified images
                    imgs_tf_curr = tup[1 + d_i][0]  # from 2nd to last
                    assert (curr_batch_sz == imgs_tf_curr.size(0))
                    actual_batch_start = d_i * curr_batch_sz
                    actual_batch_end = actual_batch_start + curr_batch_sz
                    #batches are linearized
                    all_imgs[actual_batch_start:actual_batch_end, :, :, :] = imgs_curr.cuda()
                    all_imgs_tf[actual_batch_start:actual_batch_end, :, :, :] = imgs_tf_curr.cuda()

                curr_total_batch_sz = curr_batch_sz * config.num_dataloaders
                #get actual batch size
                all_imgs = all_imgs[:curr_total_batch_sz, :, :, :]
                all_imgs_tf = all_imgs_tf[:curr_total_batch_sz, :, :, :]
                if config.sobel:
                    all_imgs = sobel_process(all_imgs, config.include_rgb)
                    all_imgs_tf = sobel_process(all_imgs_tf, config.include_rgb)

                x_outs = self.net(all_imgs, head=head)
                x_tf_outs = self.net(all_imgs_tf, head=head)

                avg_loss_batch = 0  # avg over the sub_heads
                avg_loss_no_lamb_batch = 0
                for i in range(config.num_sub_heads):
                    loss, loss_no_lamb = self.iic_loss(x_outs[i], x_tf_outs[i])
                    avg_loss_batch += loss
                    avg_loss_no_lamb_batch += loss_no_lamb

                avg_loss_batch /= config.num_sub_heads
                avg_loss_no_lamb_batch /= config.num_sub_heads

                avg_loss += avg_loss_batch.item()
                avg_loss_no_lamb += avg_loss_no_lamb_batch.item()
                avg_loss_count += 1

                avg_loss_batch.backward()
                self.iic_optimizer.step()
                #used for debugging: only first batch is used
                b_i += 1
                if b_i == 2 and config.test_code:
                  break
        
        if self.iic_scheduler is not None:
            self.iic_scheduler.step()
        avg_loss = float(avg_loss / avg_loss_count)
        avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)
        return (avg_loss, avg_loss_no_lamb)    

    def train_simclr_head(self, dataloader, n_epochs=1):
        """
        Trains the simclr head for a number of epochs, returning loss metric
        """
        self.net.train()
        config = self.config
        avg_loss = 0.
        avg_loss_count = 0
        for head_i_epoch in range(n_epochs):
            b_i = 0
            for (img1, img2), _ in dataloader:
                self.simclr_optimizer.zero_grad()
                img1 = img1.to(config.device)
                img2 = img2.to(config.device)
                if config.sobel:
                    img1 = sobel_process(img1, False, False) #Images are sobeled
                    img2 = sobel_process(img2, False, False)            
                out1 = self.net(img1, head='C')              #Processed by the net
                out2 = self.net(img2, head='C')
                ris1 = F.normalize(out1[0], dim=1)      #THen normalized
                ris2 = F.normalize(out2[0], dim=1)
                batch_loss = self.simclr_loss(ris1, ris2)
                avg_loss += batch_loss.item()
                avg_loss_count += 1
                batch_loss.backward()
                self.simclr_optimizer.step()
                #used for debugging: only first batch is used
                b_i += 1
                if b_i == 2 and config.test_code:
                    break
        if self.simclr_scheduler is not None:
            self.simclr_scheduler.step()
        avg_loss = float(avg_loss/ avg_loss_count)
        return avg_loss 
        
    def exec_pipeline(self, config_log, save_path=None, do_clustering=True, do_overclustering=True, do_simclr=True):
        """ Runs a standard train-evaluation loop. If a save path is given, a checkpoint is saved at each training epoch.
        If this function is preceeded by a checkpoint load, it will resume training from last saved epoch.
        """
        config = self.config
        print('Running on: ', torch.cuda.get_device_name(0))
        print('ClusterNet ready')
        d_A, d_B, d_C, d_ass, d_test = cluster_twohead_create_dataloaders(config)
        print('Dataloaders ready')
        if self.start_epoch < config.warmup:
            print('Starting ',config.warmup, ' epochs warmup:')
            for i in range(0, config.warmup):
                stat = self.train_simclr_head(d_C, config.num_epochs_head)
                print(' Epoch ', i, ' : ', stat)
        #Wandb support
        if config_log.log:
            wandb.init(config=config._asdict(), name=config_log.run, project=config_log.project)
        print('Starting training-evalutation loop: ')
        begin = max(config.warmup, self.start_epoch)
        for i in range(begin, config.num_epochs):
            print(' Epoch ', i,)
            #Go with iic overclustering head
            if do_overclustering:
                iic_over_perf, _ = self.train_iic_head(head='A', dataloaders=d_A, n_epochs=config.num_epochs_head)
                print('  IIC overclustering loss: ', iic_over_perf)
                if config_log.log:
                    wandb.log({
                        'overclustering_loss': iic_over_perf
                    }, step=i)
            #Go with iic clustering head
            if do_clustering: 
                iic_perf, _ = self.train_iic_head(head='B', dataloaders=d_B, n_epochs=config.num_epochs_head)
                print('  IIC loss: ',iic_perf)
                if config_log.log:
                    wandb.log({
                        'clustering_loss': iic_perf
                    }, step=i)
            #Go with simclr head
            if do_simclr:   
                simclr_perf = self.train_simclr_head(dataloader=d_C, n_epochs=config.num_epochs_head)
                print('  SimCLR loss: ',simclr_perf)
                if config_log.log:
                    wandb.log({
                        'simclr_loss': simclr_perf
                    }, step=i)
            #go with evaluation
            eval_avg, eval_best = cluster_eval(config, self.net, d_ass, d_test, sobel=config.sobel)
            if config_log.log:
                wandb.log({
                    'valid_acc_avg': eval_avg,
                    'valid_acc_best': eval_best
                }, step=i)
            print('  Average acc: ', eval_avg)
            print('  Best acc: ', eval_best)
            if save_path is not None:  
                self.save_checkpoint(i, save_path)

    def single_test(self):
        config = self.config
        print('Running on: ', torch.cuda.get_device_name(0))
        print('ClusterNet ready')
        d_A, d_B, d_C, d_ass, d_test = cluster_twohead_create_dataloaders(config)
        eval_avg, eval_best = cluster_eval(config, self.net, d_ass, d_test, sobel=config.sobel)
        print('Average acc: ', eval_avg)
        print('Best acc: ', eval_best)