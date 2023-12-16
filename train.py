import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
from os.path import join
import json
from Beta_Mixture_mode import BetaMixture1D
import torch.nn.functional as F

from affinity_layer import Affinity, sinkhorn_rpm, one_hot, BCEFocalLoss, MultiHeadAttention
from gcn import GCN



def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    with open(join('/data/zxf/home/disk/code/Opt2SAR/data/data_remote_sensing_our_1/info.json'), 'r') as fp:
        info = json.load(fp)
    name_classes = np.array(info['label'], dtype=np.str)
    num_classes = np.int(info['classes'])
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    c = (predict == all_label).squeeze()
    for i in range(len(all_label)):
        _label = all_label[i]
        class_correct[int(_label)] += c[i].item()
        class_total[int(_label)] += 1
    for i in range(num_classes):
        # print('Accuracy of %5s : %2d %%' % (
        #     name_classes[i], 100 * class_correct[i] / class_total[i]))
        print('Accuracy of %5s : %5f %%' % (
            name_classes[i], 100 * class_correct[i] / class_total[i]))
    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    prep_config = config["prep"]
    print('hello world')
    if "webcam" in data_config["source"]["list_path"] or "dslr" in data_config["source"]["list_path"]:
        prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    else:
        prep_dict["source"] = prep.image_target(**config["prep"]['params'])

    if "webcam" in data_config["target"]["list_path"] or "dslr" in data_config["target"]["list_path"]:
        prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    else:
        prep_dict["target"] = prep.image_target(**config["prep"]['params'])

    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                       transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some CDANs
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
    ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                                                 base_network, test_10crop=prep_config["test_10crop"])
            temp_model = base_network
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)
        if i % config["snapshot_interval"] == 0:
            torch.save(base_network, osp.join(config["output_path"], \
                                                             "iter_{:05d}_model.pth.tar".format(i)))

        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)

        softmax_src = nn.Softmax(dim=1)(outputs_source)
        softmax_tgt = nn.Softmax(dim=1)(outputs_target)
        softmax_out = torch.cat((softmax_src, softmax_tgt), dim=0)

        if config['CDAN'] == 'CDAN+E':
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
        elif config['CDAN'] == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
        else:
            raise ValueError('Method cannot be recognized.')

        _, s_tgt, _ = torch.svd(softmax_tgt)
        if config["method"] == "BNM":
            method_loss = -torch.mean(s_tgt)
        elif config["method"] == "BFM":
            method_loss = -torch.sqrt(torch.sum(s_tgt * s_tgt) / s_tgt.shape[0])
        elif config["method"] == "ENT":
            method_loss = -torch.mean(torch.sum(softmax_tgt * torch.log(softmax_tgt + 1e-8), dim=1)) / torch.log(
                softmax_tgt.shape[1])
        elif config["method"] == "NO":
            method_loss = 0

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        _, labels_target_fake = torch.max(nn.Softmax(dim=1)(outputs_target), 1)

        # clust_pseu_labels = pred_label
        clust_pseu_labels = labels_target_fake

        features_target = torch.as_tensor(features_target)
        features_source = torch.as_tensor(features_source)
        features_target = features_target.cuda()
        features_source = features_source.cuda()

        gcn = GCN(256, 256).cuda()
        features_source = gcn(features_source)
        features_target = gcn(features_target)

        # print(features_source.size())
        # features_source_graph =
        batchSize = features_target.size(0)
        # trainFeatures_s = torch.rand(features_source.size(0), features_source.size(1)).cuda()
        trainFeatures = torch.rand(features_target.size(0), features_target.size(1)).cuda()

        # trainFeatures = features_source

        # trainFeatures = trainFeatures / torch.norm(trainFeatures, p=2, dim=1, keepdim=True)
        features_source_norm = features_source / torch.norm(features_source, p=2, dim=1, keepdim=True)
        features_target_norm = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)

        with torch.no_grad():
            # features_target_norm = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
            # trainFeatures_s = trainFeatures_s / torch.norm(trainFeatures_s, p=2, dim=1, keepdim=True)
            trainFeatures = trainFeatures / torch.norm(trainFeatures, p=2, dim=1, keepdim=True)


        clust_pseu_labels = torch.as_tensor(clust_pseu_labels)

        trainNoisyLabels = clust_pseu_labels
        # dist_s = torch.mm(features_source_norm, trainFeatures_s.t())
        dist = torch.mm(features_target_norm, trainFeatures.t())

        # yd_s, yi_s = dist_s.topk(5, dim=1, largest=True,
        #                   sorted=True)  ## Top-K similar scores and corresponding indexes
        yd, yi = dist.topk(5, dim=1, largest=True,
                           sorted=True)  ## Top-K similar scores and corresponding indexes

        # yi = yi[:, 0:]  # batch x K
        # trainFeatures_near = trainFeatures[yi].view(yd.shape[0], -1)
        # trainFeatures_s_near = features_source_norm[yi_s].view(yd_s.shape[0], -1)
        trainFeatures_near = features_target_norm[yi].view(yd.shape[0], -1)

        # trainFeatures_re = trainFeatures.unsqueeze(0).expand(trainFeatures_near.shape[0], -1, -1)  # batch x n x dim
        # trainFeatures_s_re = trainFeatures_s_near.t()
        trainFeatures_re = trainFeatures_near.t()
        # dist_s = torch.mm(trainFeatures_s_near, trainFeatures_s_re)
        dist_t = torch.mm(trainFeatures_near, trainFeatures_re)

        # node classification layers
        affinity = Affinity(512)
        cross_domain_feature = MultiHeadAttention(256, 1, dropout=0.1, version='v2')  # Cross Graph Interaction
        intra_domain_feature = MultiHeadAttention(256, 1, dropout=0.1, version='v2')  # Intra-domain graph aggregation
        intra_f_s = \
        intra_domain_feature(features_source_norm.cpu(), features_source_norm.cpu(), features_source_norm.cpu())[0]
        intra_f_t = \
        intra_domain_feature(features_target_norm.cpu(), features_target_norm.cpu(), features_target_norm.cpu())[0]

        cross_f_s = cross_domain_feature(intra_f_t, intra_f_t, intra_f_s)[0]
        cross_f_t = cross_domain_feature(intra_f_s, intra_f_s, intra_f_t)[0]
        prob = affinity(cross_f_s, cross_f_t)
        InstNorm_layer = nn.InstanceNorm2d(1)
        M = InstNorm_layer(prob[None, None, :, :])
        M = sinkhorn_rpm(M[:, 0, :, :], n_iters=20).squeeze().exp()

        # match = (idx_source == idx_target).sum(-1).float()  # batch x K
        # _, yi_s_near_near = dist_s.topk(5, dim=1, largest=True, sorted=True)
        #
        yd_near_near, yi_near_near = dist_t.topk(5, dim=1, largest=True, sorted=True)

        candidates = trainNoisyLabels.view(1, -1).expand(batchSize,
                                                         -1)  ##Replicate the labels per row to select
        candidates = candidates.cuda()
        retrieval = torch.gather(candidates, 1,
                                 yi_near_near)  ## From replicated labels get those of the top-K neighbours using the index yi (from top-k operation)
        # C = xbm_targets.max() + 1

        retrieval_one_hot_train = torch.zeros(5, class_num).cuda()
        retrieval_one_hot_train.resize_(batchSize * 5, class_num).zero_()
        ## Generate the K*batchSize one-hot encodings from neighboring labels ("retrieval"), i.e. each row in retrieval
        # (set of neighbouring labels) is turned into a one-hot encoding
        retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd_near_near.clone().div_(0.1).exp_()  ## Apply temperature to scores
        yd_transform[...] = 1.0  ##To avoid using similarities
        probs_corrected = torch.sum(
            torch.mul(retrieval_one_hot_train.view(batchSize, -1, class_num),
                      yd_transform.view(batchSize, -1, 1)), 1)

        # new_pseudo_labels = torch.max(nn.Softmax(dim=1)(probs_corrected), 1)[1]

        confidence, new_pseudo_labels = torch.max(nn.Softmax(dim=1)(probs_corrected), 1)

        # id = (new_pseudo_labels == clust_pseu_labels)

        new_labels = new_pseudo_labels
        # contra_loss = InfoNCELoss()
        # con_loss = contra_loss(dynamic_similarity1, labels_source, new_labels)

        bmm_model = BetaMixture1D(max_iters=10)
        # entropy_t = entropy[batchSize:].cpu()
        confidence = confidence.cpu()
        # c = np.asarray(entropy)
        c = np.asarray(confidence.detach().numpy())
        c, c_min, c_max = bmm_model.outlier_remove(c)
        c = bmm_model.normalize(c, c_min, c_max)
        bmm_model.fit(c)
        bmm_model.create_lookup(1)  # 0: noisy, 1: clean
        # get posterior
        c = np.asarray(confidence.detach().numpy())
        # c = np.asarray(entropy.numpy())
        c = bmm_model.normalize(c, c_min, c_max)
        p = bmm_model.look_lookup(c)
        p = torch.from_numpy(p)
        # p[p < torch.rand_like(p)] = 0.0
        # p = 1+torch.exp(-p)
        # p[p < 0.1] = 0.0

        # p = 1 + torch.exp(-p)
        weight = torch.tensor(p, dtype=torch.float32).cuda()

        # weight[weight < 0.1] = 0.0
        # weight[weight < 0.7] = 0.0

        # weight = weight / torch.sum(weight).item()
        new_labels = new_labels.cuda()

        one_hot_s = one_hot(labels_source.cpu(), class_num)
        one_hot_t = one_hot(new_labels.cpu(), class_num)
        one_hot_s = torch.as_tensor(one_hot_s)
        one_hot_t = torch.as_tensor(one_hot_t)
        matching_target = torch.mm(one_hot_s, one_hot_t.t())
        TP_mask = (matching_target == 1).float()
        indx = (M * TP_mask).max(-1)[1]
        TP_samples = M[range(M.size(0)), indx].view(-1, 1)
        TP_target = torch.full(TP_samples.shape, 1, dtype=torch.float, device=TP_samples.device).float()

        FP_samples = M[matching_target == 0].view(-1, 1)
        FP_target = torch.full(FP_samples.shape, 0, dtype=torch.float, device=FP_samples.device).float()
        matching_loss = BCEFocalLoss()
        TP_loss = matching_loss(TP_samples, TP_target.float()) / len(TP_samples)
        FP_loss = matching_loss(FP_samples, FP_target.float()) / torch.sum(FP_samples).detach()
        # print('FP: ', FP_loss, 'TP: ', TP_loss)
        M = M.cuda()
        R = torch.mm(M, features_source_norm) - torch.mm(M, features_target_norm)
        quadratic_loss = torch.nn.L1Loss(reduction='mean')
        loss_quadratic = quadratic_loss(R, R.new_zeros(R.size()))
        TP_loss = TP_loss.cuda()
        FP_loss = FP_loss.cuda()
        matching_loss = TP_loss + FP_loss + loss_quadratic
        matching_loss = matching_loss.cuda()
        outputs_target = outputs_target.cuda()
        loss_t = nn.CrossEntropyLoss()(outputs_target, new_labels)
        
        weight[weight < 0.5] = 0.0


        weight = weight / torch.sum(weight).item()
        if torch.sum(weight) > 1e-10:
            loss_t_weight = torch.sum(weight * loss_t) / (torch.sum(weight) + 1e-10)
          
        else:
            loss_t_weight = 0.0
            # loss_in_weight = 0.0

       
        total_loss = loss_params["trade_off"] * transfer_loss \
                     + args.cls_weight * classifier_loss \
                     + args.self_coeff * loss_t_weight + args.matching_weight * matching_loss + loss_params["lambda_method"] * method_loss
        
        total_loss.backward()
        optimizer.step()
        if i % config['print_num'] == 0:
            log_str = "iter: {:05d}, classification: {:.5f}, transfer: {:.5f}, method: {:.5f}".format(i,
              classifier_loss, transfer_loss, method_loss)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            if config['show']:
                print(log_str)

        if i % args.print_freq == 0:
            
            print("iter: {:05d},transfer_loss:{:.6f}, classifier_loss:{:.6f}, loss_t_weight:{:6f}, matching_loss:{:6f}, method_loss:{:.6f},total_loss:{:.6f}" \
                  .format(i, loss_params["trade_off"] * transfer_loss.item(), args.cls_weight * classifier_loss.item(),
                          args.self_coeff * loss_t_weight, args.matching_weight * matching_loss, loss_params["lambda_method"] * method_loss, total_loss))
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--CDAN', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E'])

    parser.add_argument('--method', type=str, default='BNM', choices=['BNM', 'BFM', 'ENT', 'NO'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50',
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13",
                                 "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    
    parser.add_argument('--s_dset_path', type=str,
                        default='/data/zxf/home/disk/code/Opt2SAR/data/data_remote_sensing_our_1/source_NWPU_da.txt',
                        help="The source dataset path list")
 
    parser.add_argument('--t_dset_path', type=str,
                        default='/data/zxf/home/disk/code/Opt2SAR/data/data_remote_sensing_our_1/target_WHU-SAR_da.txt',
                        help="The target dataset path list")

    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--print_num', type=int, default=100, help="print num ")
    parser.add_argument('--batch_size', type=int, default=36, help="number of batch size ")

    parser.add_argument('--num_iterations', type=int, default=20000, help="total iterations")
    parser.add_argument('--snapshot_interval', type=int, default=500, help="interval of two continuous output model")
    
    parser.add_argument('--output_dir', type=str,
                        default='/data/zxf/home/disk/code/Opt2SAR/snapshot_our_graph/NWPU2WHU-SAR')
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--trade_off', type=float, default=1.0, help="parameter for CDAN")
    # parser.add_argument('--lambda_method', type=float, default=0.1, help="parameter for method")
    parser.add_argument('--lambda_method', type=float, default=0.1, help="parameter for method")

    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--show', type=bool, default=False, help="whether show the loss functions")
    
    parser.add_argument('--self_coeff', type=float, default=0.001)
    

    parser.add_argument("--cls_weight", type=float, default=1.0)

    parser.add_argument('--momentum', default=1.0, type=float)
    parser.add_argument('--print_freq', type=int, default=1)
    
    parser.add_argument("--matching_weight", type=float, default=0.1)
   


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config['CDAN'] = args.CDAN
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations
    config["print_num"] = args.print_num
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["show"] = args.show
    # config["output_path"] = args.dset + '/' + args.output_dir
    config["output_path"] = args.output_dir
    config["matching_weight"] = args.matching_weight
    config["self_coeff"] = args.self_coeff

    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop": False, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    config["loss"] = {"trade_off": args.trade_off, "lambda_method": args.lambda_method}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": args.batch_size}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": args.batch_size}, \
                      "test": {"list_path": args.t_dset_path, "batch_size": args.batch_size}}

    if config["dataset"] == "office":
        if ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
        config["network"]["params"]["class_num"] = 6
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 6
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
        config["network"]["params"]["class_num"] = 6
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 6
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')

    seed = random.randint(1, 10000)
    # seed = 1349
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # uncommenting the following two lines for reproducing
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
