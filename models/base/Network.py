import argparse

import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
import numpy as np
from tqdm import tqdm
from utils import *


from .helper import *
from utils import *
from dataloader.data_utils import *



class MYNET(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args

        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20(argu=args)
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(argu=args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, argu=args)# pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        

        
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        
        self.mem_mode = 'feat'
        

        self.features = torch.zeros(200, args.n_clusters).cuda()
        self.covs = torch.zeros(200, args.n_clusters, args.n_clusters).cuda()
        self.centers = torch.zeros(args.n_clusters, args.n_clusters).cuda()
        self.register_buffer('mem_feat', self.features)
        self.register_buffer('mem_cov', self.covs)
        self.register_buffer('mem_centers', self.centers)


        
    def forward_metric(self, x, without_dynamic):
        if without_dynamic == True:
            x, g_out = self.encode(x, without_dynamic)
        elif without_dynamic == False:
            x, g_out, attention, identity = self.encode(x, without_dynamic)
        g_label = x
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
        if without_dynamic==True:
            return x, g_label
        if without_dynamic==False:
            return x, g_label, attention, identity

    def encode(self, x, without_dynamic):
        if without_dynamic == True:
            x = self.encoder(x, without_dynamic)
            g_out = x
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
            return x, g_out

        elif without_dynamic == False:
            identity, attention, x = self.encoder(x, without_dynamic)
            g_out = x
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
            return x, g_out, attention, identity


    def forward(self, input, without_dynamic):
        if self.mode != 'encoder':
            if without_dynamic==False:
                input, g_label, attention, identity = self.forward_metric(input, without_dynamic)
                return input, g_label, attention, identity
            elif without_dynamic==True:
                input, g_label = self.forward_metric(input, without_dynamic)
                return input, g_label


        elif self.mode == 'encoder':
            if without_dynamic == True:
                input,g_out = self.encode(input, True)
                return input, g_out

            elif without_dynamic == False:
                input, g_out, attention, identity = self.encode(input, False)
                return input, g_out, attention, identity

        else:
            raise ValueError('Unknown mode')
            
    def update_fc(self, dataloader,class_list,session, args):
        i = 0
        for batch in dataloader:
            i+=1
            raw_data, label = [_.cuda() for _ in batch]
            if args.without_dynamic == False:
                data,g_out,attention,identity=self.encode(raw_data, False)#.detach()
            elif args.without_dynamic == True:
                data,_ = self.encode(raw_data, True)
            print('counting data:', data.shape)
            data = data.detach()
        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            print('update average')
            old_class_num = self.args.base_class + self.args.way * (session - 1)
            if args.without_dynamic == True:
                attention = 0
            new_fc = self.update_fc_avg(data, attention, label, class_list, old_class_num,args)
            self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)
            
        # if 'ft' in self.args.new_mode:  # further finetune
        if True:
            print('start finetuning')
            max_acc = [0 for i in range(8)]
            max_epoch = 0
            self.update_fc_ft(new_fc,raw_data,data,attention,label,session,max_acc,max_epoch)
            

    def update_fc_avg(self,data,attention,label,class_list, old_class_num,args):
        new_fc=[]
        print('check class list', class_list)
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)

            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto


        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            print('x.shape in get_logits :', x.shape)
            return F.linear(x,fc)        
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))


    def update_fc_ft(self,new_fc,raw_data,old_out,attention,label,session,max_acc,max_epoch):

        print('start finetuning!!!')
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True

        print('new fc', new_fc.shape)
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr':0.1},
                                     {'params': self.encoder.dynamic_conv.attention.parameters(), 'lr': 0.1},
                                    ],
                                     momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():


            old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :]
            old_logits = self.get_logits(old_out, old_fc).detach()

            for epoch in range(self.args.epochs_new):
                if self.args.without_dynamic == False:
                    new_out, _, _, _ = self.encode(raw_data, without_dynamic=False)
                elif self.args.without_dynamic == True:
                    new_out, _ = self.encode(raw_data, without_dynamic=True)
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :] #.detach()
                fc = torch.cat([old_fc, new_fc], dim=0)

                logits = self.get_logits(new_out,fc)  


                ############### distillation loss ##########################################
                T = 0.2
                distill_loss = -(F.log_softmax(logits[:,:old_logits.shape[1]]/T, dim=1) * F.softmax(old_logits/T, dim=1)).sum(-1).mean(-1)

                ce_loss = F.cross_entropy(logits, label)

                loss = ce_loss*1 + distill_loss * 30

                optimizer.zero_grad()
                loss.backward()


                optimizer.step()
                self.fc.weight.data[:self.args.base_class + self.args.way * session, :].copy_(fc.data)
      
                from models.base.helper import test
                from copy import deepcopy
                trainset, trainloader, testloader = get_new_dataloader(self.args, session)
                tsl, tsa,_ = test(self, testloader, epoch, self.args, session, without_dynamic=self.args.without_dynamic)
                if (tsa * 100) >= max_acc[session-1]:
                    max_acc[session-1] = float('%.3f' % (tsa * 100))
                    max_epoch = epoch
                    sh = self.args.sh
                    save_model_dir = os.path.join('session' + str(session) + '_' + str(sh) + '_max_acc.pth')
                    torch.save(dict(params=self.state_dict()), save_model_dir)
                    self.best_model_dict = deepcopy(self.state_dict())
                    print('********A better model is found!!**********')
                    print('Saving model to :%s' % save_model_dir)
                print('best epoch {}, best test acc={:.3f}'.format(max_epoch, max_acc[session-1]))

    
        
        

        
