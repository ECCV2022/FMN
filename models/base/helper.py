from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import time
import torch.nn as nn

def replace_base_fc(trainset, transform, model, args, without_dynamic, init=False):
    # replace fc.weight with the embedding average of train data    
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,    
                                              num_workers=8, pin_memory=True, shuffle=False)    
    trainloader.dataset.transform = transform    
    embedding_list = []
    label_list = []
    attention_embedding_list = []
    identity_embedding_list = []

    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            if without_dynamic==True:
                embedding,_ = model(data, True)
            elif without_dynamic==False:
                embedding,_,attention,identity = model(data, False)
                attention_embedding_list.append(attention.cpu())
                identity_embedding_list.append(identity)
            embedding_list.append(embedding.cpu())
            label_list.append(label)
    embedding_list = torch.cat(embedding_list, dim=0) #(30000,512)

    label_list = torch.cat(label_list, dim=0)
    proto_list = []
    cov_list = []
    feature = []
    # args.base_class = 100
    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        feature.append(embedding_this.cpu().detach().numpy()[:500])
        cov_this = torch.tensor(np.cov(embedding_this.cpu().numpy().T))
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
        cov_list.append(cov_this)
    feature = np.array(feature)
    proto_list = torch.stack(proto_list, dim=0)  # (60,512)
    cov_list = torch.stack(cov_list, dim=0)  # (60,512,512)
    print(model.module.fc.weight.data[:args.base_class].shape, proto_list.shape)
    model.module.fc.weight.data[:args.base_class] = proto_list

    if without_dynamic==False:
        attention_proto_list = []
        attention_cov_list = []
        attention = []

        label = []
        for class_index in range(args.base_class):
            data_index = (label_list == class_index).nonzero()
            label.append(np.ones(500)*class_index)
        label = np.array(label)
        feature = np.array(feature)
        print(feature.shape)
        predict = np.argmax(feature, axis=-1)
        print('feature.shape:', feature.shape)
        print('predict.shape:', predict.shape)
        print('label.shape:', label.shape)

    return model

def base_train(basis, model, trainloader, optimizer, scheduler, epoch, args, dis_flag = False):
    total_loss = 0
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    trainiter = iter(trainloader)
    tqdm_gen = tqdm(range(len(trainloader)))
    for i, batch in enumerate(tqdm_gen, 1):
        start = time.time()
        batch = trainiter.next()
        start = time.time()
        data, train_label = [var.cuda() for var in batch]
        if args.without_dynamic == False:
            logits, g_label,attention,_ = model(data, False)
        elif args.without_dynamic == True:
            logits, g_label = model(data, True)

        # orthogonality loss
        cos_sim = 0      
        for i, basis_1 in enumerate(basis):
            for j, basis_2 in enumerate(basis):
                if i != j :
                    cos_sim += torch.cosine_similarity(basis_1, basis_2)
        ortho_loss = torch.sum(cos_sim / 2.0)/args.batch_size_base

        ce_loss = F.cross_entropy(logits, train_label)
        
        if args.without_dynamic == False:
            loss = ce_loss
        elif args.without_dynamic == True:
            loss = ce_loss
        acc = count_acc(logits, train_label)
        total_loss = total_loss + loss


        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta

def test(model, testloader, epoch, args, session, without_dynamic):
    test_class = args.base_class + session * args.way

    model = model.eval()
    vl = Averager()
    va = Averager()
    acc_list = list()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]

            if without_dynamic==True:
                logits, _ = model(data, without_dynamic)
            elif without_dynamic==False:
                logits, _,_,_ = model(data, without_dynamic)
            # logits = logits[:, :test_class]

            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            # print(logits.shape, test_label.shape)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            acc_list.append(acc)


        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va, acc_list




