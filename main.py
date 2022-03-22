import os
import numpy as np 
import pandas as pd
from sympy import arg, evaluate 
import torch 
import torch.nn as nn
import argparse
import random
import utils
import logging
import datetime
from torch.utils.data import DataLoader
from dataset import GetData
from model.feature_net import FeatureNet
import random
from pprint import pprint
# import pprint
# def pprint(*text):
#     time = '['+str(datetime.datetime.utcnow() +
#             datetime.timedelta(hours=8))[:19]+'] -'
#     print(time, *text, flush=True)

def seed_set(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# def norm(data, dim, ntype):
#     x = utils.Norm_data(data, ntype, dim)
#     norm_x = x.norm()
#     return x, norm_x
def L2loss(model, lamda=1.5e-3):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    return reg_loss

def evaluation(pre, real):
    '''
    inp:
        pre: [b, pre_len, nodes]
        real: same
    oup: metrics
    '''
    eps = 1
    a, b, c = np.where(real == 0)
    pre[a, b, c] = 1
    real[a, b, c] = 1

    mae = np.abs(pre-real).sum() / (real.size - a.size)
    rmse = np.sqrt(((pre-real)**2).sum() / (real.size - a.size))
    mape = (np.abs(pre-real) / real).sum() / (real.size - a.size)
    acc = 1 - mape

    # mae = np.array(np.mean(np.abs(pre-real)), dtype=float)
    # rmse = np.array(np.sqrt(np.mean(((pre-real)**2))), dtype=float)
    # mape = np.array(np.mean((np.abs(pre-real)/(real))), dtype=float)
    # acc = np.array(np.mean((1- (np.abs(real - pre) / (real)))), dtype=float)
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'acc': acc}

def get_data(src_data, trg_data, seq_len, pre_len, norm_type, batch_size, attr:list=None):
    '''
    inp:
        src_data: [time_len, src_nodes]
        trg_data: [time_len, trg_nodes]
        seq_len: int
        pre_len: int
    oup: dataset, dataloader
    '''
    dataset = GetData(src_data, trg_data, seq_len, pre_len, norm_type, attr)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=0)
    return dataset, dataloader

def main(**kwargs):
    
    args = kwargs

    # ----- Data load ----- #
    logger.info('DATA LOAD')
    print('DATA LOAD')

    src_raw_data, trg_raw_data = np.load(args['src_file']), np.load(args['trg_file'])
    src_adj, src_speed = src_raw_data['adj'], src_raw_data['speed']
    trg_adj, trg_speed = trg_raw_data['adj'], trg_raw_data['speed']
    adjs = [src_adj, trg_adj]
    trg_data = trg_speed[args['day_start']*288:(args['day_start']+args['day_inter']+1)*288]
    src_data = src_speed[args['day_start']*288:(args['day_start']+args['day_inter']+1)*288]
    assert trg_data.shape[0] == src_data.shape[0]
    train_src_data, val_src_data = src_data[:288], src_data[288:]
    train_trg_data, val_trg_data = trg_data[:288], trg_data[288:]
    
    src_attr = {'mean': np.mean(src_speed), 
                'std': np.std(src_speed),
                'max': np.max(src_speed),
                'min': np.min(src_speed),
                'med': np.median(src_speed)}
    trg_attr = {'mean': np.mean(train_trg_data), 
                'std': np.std(train_trg_data),
                'max': np.max(train_trg_data),
                'min': np.min(train_trg_data),
                'med': np.median(train_trg_data)}

    # ----- Data process ----- #
    logger.info('DATA PROCESS')
    print('DATA PROCESS')
    train_dataset, train_dataloader = get_data(train_src_data, train_trg_data, args['seq_len'], args['pre_len'], args['norm_type'], args['batch_size'], [src_attr, src_attr])
    val_dataset, val_dataloader = get_data(val_src_data, val_trg_data, args['seq_len'], args['pre_len'], args['norm_type'], args['batch_size'], [src_attr, src_attr])
    
    # trg_train_dataset, trg_train_dataloader = get_data(trg_train, args['seq_len'], args['pre_len'], args['norm_type'], args['batch_size'])
    # trg_val_dataset, trg_val_dataloader = get_data(trg_val, args['seq_len'], args['pre_len'], args['norm_type'], args['batch_size'])
    

    logger.info('\n---------Init information---------- \
                \ndatafile: %s \
                \nSrcTrainDataShape: %s \nSrcValDataShape: %s \
                \nTrgTrainDataShape: %s \nTrgValDataShape: %s \
                \nTrainDatasetLen: %d \nValDatasetLen: %d \
                \n-----------------------------------'\
                %(args['data_file'], \
                str(train_src_data.shape), str(val_src_data.shape), \
                str(train_trg_data.shape), str(val_trg_data.shape), \
                train_dataset.__len__(), \
                val_dataset.__len__()))
    logger.info('\n\nSET INFO: %s'%(args))
    print('\n---------Init information---------- \
                \ndatafile: %s \
                \nSrcTrainDataShape: %s \nSrcValDataShape: %s \
                \nTrgTrainDataShape: %s \nTrgValDataShape: %s \
                \nTrainDatasetLen: %d \nValDatasetLen: %d \
                \n-----------------------------------'\
                %(args['data_file'], \
                str(train_src_data.shape), str(val_src_data.shape), \
                str(train_trg_data.shape), str(val_trg_data.shape), \
                train_dataset.__len__(), \
                val_dataset.__len__()))



    # ------ init model ------ #
    logger.info('INIT MODEL')
    print('INIT MODEL')
    model = FeatureNet(adjs, args['hidden_size'], args['pre_len']).cuda('cuda:'+args['gpu_id'])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['opt_lr'])
    criterion = nn.L1Loss()
        # metric set
    ttl_metric = {'mae': [], 'rmse': [], 'mape': [],  'acc': []}
    bst_dic = {'bst_mape': None, 'bst_model': None,
               'bst_optimizer': None, 'bst_epoch': None, 'bst_loss': None}

    epoch_metric = {'SrcPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []},\
                    'TrgPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []}}
    epoch_loss = {'loss_hidden_src_trg': [], 'loss_pre_src_trg': [], 'loss_pretrg_trueY': [], \
                    'loss_presrc_trueY': [], 'ttl_loss':[]}

    val_epoch_metric = {'SrcPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []},\
                    'TrgPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []}}
    val_epoch_loss = {'loss_hidden_src_trg': [], 'loss_pre_src_trg': [], 'loss_pretrg_trueY': [], \
                        'loss_presrc_trueY': [], 'ttl_loss':[]}

    global_metric = {'SrcPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []},\
                    'TrgPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []}}
    global_loss = {'loss_hidden_src_trg': [], 'loss_pre_src_trg': [], 'loss_pretrg_trueY': [], \
                    'loss_presrc_trueY':[], 'ttl_loss':[]}
    
    val_global_metric = {'SrcPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []},\
                    'TrgPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []}}
    val_global_loss = {'loss_hidden_src_trg': [], 'loss_pre_src_trg': [], 'loss_pretrg_trueY': [], \
                        'loss_presrc_trueY': [], 'ttl_loss':[]}

    loss_rt =  args['loss_rt']
    # ------ TRAIN MODEL ------ #
    logger.info('START TRAIN')
    print('START TRAIN')
    model.train()
    for epoch in range(args['epochs']):
        iter_metric = {'SrcPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []},\
                        'TrgPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []}}
        train_cnt_loss = {'loss_hidden_src_trg': [], 'loss_pre_src_trg': [], 'loss_pretrg_trueY': [], \
                            'loss_presrc_trueY': [], 'ttl_loss': []}
        
        for iter, (train_X, train_Y) in enumerate(train_dataloader):
            '''
            inp: 
                train_X [SrcX, TrgX]      SrcX [b, seq_len, src_node+1] TrgX [b, seq_len, trg_node+1]
                train_Y [SrcY, TrgY]      SrcY [b, pre_len, src_node+1] TrgY [b, pre_len, trg_node+1]
            '''
            # src_x, src_y = (src_speed, args['seq_len'],args['batch_size'],args['pre_len'])
            src_x, trg_x = train_X[0].cuda('cuda:'+args['gpu_id']), train_X[1].cuda('cuda:'+args['gpu_id'])
            src_y, trg_y = train_Y[0].to(src_x.device), train_Y[1].to(src_x.device)

            optimizer.zero_grad()
            output_model = model(src_x, trg_x)
            # TODO: 调整的
            loss_hidden_src_trg = criterion(output_model[0][0], output_model[0][1]) # loss in hidden src and trg
            loss_pre_src_trg = criterion(output_model[1][0], output_model[1][1]) # loss in pre src and trg
            loss_presrc_trueY = criterion(output_model[1][0], trg_y[:,:,1:]) # loss in pre_src and tru_y
            loss_pretrg_trueY = criterion(output_model[1][1], trg_y[:,:,1:]) # loss in pre_y and tru_y
            # if epoch > 100:
            #     ttl_loss = 0.2*loss_hidden_src_trg+0.2*loss_pre_src_trg+0.2*loss_presrc_trueY + 0.4*loss_pretrg_trueY
            # else: 
            #     ttl_loss = 0.2*loss_hidden_src_trg+0.2*loss_pre_src_trg+0.2*loss_presrc_trueY + 0.4*loss_pretrg_trueY
            ttl_loss = loss_rt[0]*loss_hidden_src_trg + loss_rt[1]*loss_pre_src_trg \
                        +loss_rt[2]*loss_presrc_trueY + loss_rt[3]*loss_pretrg_trueY + loss_rt[4]*L2loss(model)
            # ttl_loss = 0.9*loss_pretrg_trueY + 0.1*L2loss(model)

            pre_src_inv = train_dataset.inverse(output_model[1][0])
            pre_trg_inv = train_dataset.inverse(output_model[1][1])
            y = train_dataset.inverse(trg_y[:,:,1:])
            metric_dic_PreSrc_TrueY = evaluation(pre_src_inv.detach().cpu().numpy(), \
                                    y.detach().cpu().numpy())
            metric_dic_PreTrg_TrueY = evaluation(pre_trg_inv.detach().cpu().numpy(), \
                                    y.detach().cpu().numpy())

            for key in metric_dic_PreTrg_TrueY.keys():
                iter_metric['SrcPre_TrgY'][key].append(metric_dic_PreSrc_TrueY[key])
                iter_metric['TrgPre_TrgY'][key].append(metric_dic_PreTrg_TrueY[key])
        

            train_cnt_loss['loss_hidden_src_trg'].append(loss_hidden_src_trg.item())
            train_cnt_loss['loss_pre_src_trg'].append(loss_pre_src_trg.item())
            train_cnt_loss['loss_pretrg_trueY'].append(loss_pretrg_trueY.item())
            train_cnt_loss['loss_presrc_trueY'].append(loss_presrc_trueY.item())
            train_cnt_loss['ttl_loss'].append(ttl_loss.item())

            ttl_loss.backward()

            # ----- save loss and metric each iter for show ----- #
            for key in global_loss.keys():
                global_loss[key].append(train_cnt_loss[key][-1])
            for key in metric_dic_PreTrg_TrueY.keys():
                global_metric['SrcPre_TrgY'][key].append(metric_dic_PreSrc_TrueY[key])
                global_metric['TrgPre_TrgY'][key].append(metric_dic_PreTrg_TrueY[key])

            np.save(args['log_file']+'/train_loss.npy', global_loss)
            np.save(args['log_file']+'/train_metric.npy', global_metric)
            # --------------------------------------------------- #

        optimizer.step()
        # epoch loss save
        for key in epoch_loss.keys():
            epoch_loss[key].append(np.mean(train_cnt_loss[key]))
        
        # epoch metric save
        for key in metric_dic_PreTrg_TrueY.keys():
            epoch_metric['SrcPre_TrgY'][key].append(np.mean(iter_metric['SrcPre_TrgY'][key]))
            epoch_metric['TrgPre_TrgY'][key].append(np.mean(iter_metric['TrgPre_TrgY'][key]))
        
        if epoch % args['print_epoch'] == 0:
            logger.info('Epoch: %02d --------------------'%(epoch))
            logger.info('SrcPre_TrgY MAE: %.4f, RMSE: %.4f, MAPE: %.4f, ACC: %.4f'% \
                (epoch_metric['SrcPre_TrgY']['mae'][-1],
                epoch_metric['SrcPre_TrgY']['rmse'][-1],
                epoch_metric['SrcPre_TrgY']['mape'][-1],
                epoch_metric['SrcPre_TrgY']['acc'][-1]))
            logger.info('TrgPre_TrgY MAE: %.4f, RMSE: %.4f, MAPE: %.4f, ACC: %.4f \n'% \
                (epoch_metric['TrgPre_TrgY']['mae'][-1],
                epoch_metric['TrgPre_TrgY']['rmse'][-1],
                epoch_metric['TrgPre_TrgY']['mape'][-1],
                epoch_metric['TrgPre_TrgY']['acc'][-1]))
            logger.info('\n')
            print('Epoch: %02d --------------------'%(epoch))
            print('SrcPre_TrgY MAE: %.4f, RMSE: %.4f, MAPE: %.4f, ACC: %.4f'% \
                (epoch_metric['SrcPre_TrgY']['mae'][-1],
                epoch_metric['SrcPre_TrgY']['rmse'][-1],
                epoch_metric['SrcPre_TrgY']['mape'][-1],
                epoch_metric['SrcPre_TrgY']['acc'][-1]))
            print('TrgPre_TrgY MAE: %.4f, RMSE: %.4f, MAPE: %.4f, ACC: %.4f \n'% \
                (epoch_metric['TrgPre_TrgY']['mae'][-1],
                epoch_metric['TrgPre_TrgY']['rmse'][-1],
                epoch_metric['TrgPre_TrgY']['mape'][-1],
                epoch_metric['TrgPre_TrgY']['acc'][-1]))

        # ------- save epoch loss & metric to file ------ #
        np.save(args['log_file']+'/train_epoch_loss.npy', epoch_loss)
        np.save(args['log_file']+'/train_epoch_metric.npy', epoch_metric)
        # ----------------------------------------------- #


        # ------- VALIDATION -------- #
        if epoch % args['val_inter_epoch'] == 0:
            model.eval()

            logger.info('######### VALIDATION #########')
            print('######### VALIDATION #########')

            val_iter_metric = {'SrcPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []},\
                        'TrgPre_TrgY': {'mae': [], 'rmse': [], 'mape': [], 'acc': []}}
            val_cnt_loss = {'loss_hidden_src_trg': [], 'loss_pre_src_trg': [], 'loss_pretrg_trueY': [], \
                            'loss_presrc_trueY': [], 'ttl_loss': []}

            for iter, (Val_X, Val_Y) in enumerate(val_dataloader):
                '''
                inp: 
                    VAL_X [SrcX, TrgX]      SrcX [b, seq_len, src_node+1] TrgX [b, seq_len, trg_node+1]
                    VAL_Y [SrcY, TrgY]      SrcY [b, pre_len, src_node+1] TrgY [b, pre_len, trg_node+1]
                '''
                # src_x, src_y = (src_speed, args['seq_len'],args['batch_size'],args['pre_len'])
                src_x, trg_x = Val_X[0].cuda('cuda:'+args['gpu_id']), Val_X[1].cuda('cuda:'+args['gpu_id'])
                src_y, trg_y = Val_Y[0].to(src_x.device), Val_Y[1].to(src_x.device)

                # optimizer.zero_grad()
                output_model = model(src_x, trg_x)
                # TODO: 调整的
                loss_hidden_src_trg = criterion(output_model[0][0], output_model[0][1]) # loss in hidden src and trg
                loss_pre_src_trg = criterion(output_model[1][0], output_model[1][1]) # loss in pre src and trg
                loss_presrc_trueY = criterion(output_model[1][0], trg_y[:,:,1:]) # loss in pre src and tru_y
                loss_pretrg_trueY = criterion(output_model[1][1], trg_y[:,:,1:]) # loss in pre_y and tru_y
                # if epoch > 100:
                #     ttl_loss = 0.2*loss_hidden_src_trg+0.2*loss_pre_src_trg+0.2*loss_presrc_trueY + 0.4*loss_pretrg_trueY
                # else:
                    # ttl_loss = 0.2*loss_hidden_src_trg+0.2*loss_pre_src_trg+0.2*loss_presrc_trueY + 0.4*loss_pretrg_trueY

                ttl_loss = 0.1*loss_hidden_src_trg+0.2*loss_pre_src_trg+0.4*loss_presrc_trueY + 0.3*loss_pretrg_trueY 

                pre_src_inv = train_dataset.inverse(output_model[1][0])
                pre_trg_inv = train_dataset.inverse(output_model[1][1])
                y = train_dataset.inverse(trg_y[:,:,1:])
                val_metric_dic_PreSrc_TrueY = evaluation(pre_src_inv.detach().cpu().numpy(), \
                                        y.detach().cpu().numpy())
                val_metric_dic_PreTrg_TrueY = evaluation(pre_trg_inv.detach().cpu().numpy(), \
                                        y.detach().cpu().numpy())

                for key in metric_dic_PreTrg_TrueY.keys():
                    val_iter_metric['SrcPre_TrgY'][key].append(val_metric_dic_PreSrc_TrueY[key])
                    val_iter_metric['TrgPre_TrgY'][key].append(val_metric_dic_PreTrg_TrueY[key])

                val_cnt_loss['loss_hidden_src_trg'].append(loss_hidden_src_trg.item())
                val_cnt_loss['loss_pre_src_trg'].append(loss_pre_src_trg.item())
                val_cnt_loss['loss_pretrg_trueY'].append(loss_pretrg_trueY.item())
                val_cnt_loss['loss_presrc_trueY'].append(loss_presrc_trueY.item())
                val_cnt_loss['ttl_loss'].append(ttl_loss.item())

                # ----- save val loss and metric each iter for show ----- #
                for key in val_global_loss.keys():
                    val_global_loss[key].append(val_cnt_loss[key][-1])
                for key in val_metric_dic_PreTrg_TrueY.keys():
                    val_global_metric['SrcPre_TrgY'][key].append(val_metric_dic_PreSrc_TrueY[key])
                    val_global_metric['TrgPre_TrgY'][key].append(val_metric_dic_PreTrg_TrueY[key])

                np.save(args['log_file']+'/val_loss.npy', val_global_loss)
                np.save(args['log_file']+'/val_metric.npy', val_global_metric)
                # ------------------------------------------------------- #

            # 计算val loss / metric
            for key in epoch_loss.keys():
                val_epoch_loss[key].append(np.mean(val_cnt_loss[key]))

            for key in metric_dic_PreTrg_TrueY.keys():
                val_epoch_metric['SrcPre_TrgY'][key].append(np.mean(val_iter_metric['SrcPre_TrgY'][key]))
                val_epoch_metric['TrgPre_TrgY'][key].append(np.mean(val_iter_metric['TrgPre_TrgY'][key]))

            # ------- save val epoch loss & metric to file ------ #
            np.save(args['log_file']+'/val_epoch_loss.npy', val_epoch_loss)
            np.save(args['log_file']+'/val_epoch_metric.npy', val_epoch_metric)
            # --------------------------------------------------- #

            logger.info('VALIDATION------------------')
            logger.info('SrcPre_TrgY MAE: %.4f, RMSE: %.4f, MAPE: %.4f, ACC: %.4f'% \
                (val_epoch_metric['SrcPre_TrgY']['mae'][-1],
                val_epoch_metric['SrcPre_TrgY']['rmse'][-1],
                val_epoch_metric['SrcPre_TrgY']['mape'][-1],
                val_epoch_metric['SrcPre_TrgY']['acc'][-1]))
            logger.info('TrgPre_TrgY MAE: %.4f, RMSE: %.4f, MAPE: %.4f, ACC: %.4f'% \
                (val_epoch_metric['TrgPre_TrgY']['mae'][-1],
                val_epoch_metric['TrgPre_TrgY']['rmse'][-1],
                val_epoch_metric['TrgPre_TrgY']['mape'][-1],
                val_epoch_metric['TrgPre_TrgY']['acc'][-1]))
            print('VALIDATION------------------')
            print('SrcPre_TrgY MAE: %.4f, RMSE: %.4f, MAPE: %.4f, ACC: %.4f'% \
                (val_epoch_metric['SrcPre_TrgY']['mae'][-1],
                val_epoch_metric['SrcPre_TrgY']['rmse'][-1],
                val_epoch_metric['SrcPre_TrgY']['mape'][-1],
                val_epoch_metric['SrcPre_TrgY']['acc'][-1]))
            print('TrgPre_TrgY MAE: %.4f, RMSE: %.4f, MAPE: %.4f, ACC: %.4f'% \
                (val_epoch_metric['TrgPre_TrgY']['mae'][-1],
                val_epoch_metric['TrgPre_TrgY']['rmse'][-1],
                val_epoch_metric['TrgPre_TrgY']['mape'][-1],
                val_epoch_metric['TrgPre_TrgY']['acc'][-1]))

            # VALIDATION ENDING
            logger.info('########## VAL END ###########\n')
            print('########## VAL END ###########\n')






def get_args():
    parser = argparse.ArgumentParser()
    # data para
    parser.add_argument('--model_name', type=str, default='standard_net')
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--pre_len', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_file', type=str, default='./data')
    parser.add_argument('--src_city', type=str, default='bj')
    parser.add_argument('--trg_city',type=str, default='sh')
    parser.add_argument('--day_start',type=int, default=30)
    parser.add_argument('--day_inter', type=int, default=10)
    # model para
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--gpu_id', type=str, default='7')
    parser.add_argument('--opt_lr', type=float, default=0.001)
    parser.add_argument('--oup_file', type=str, default='./outputs')
    parser.add_argument('--norm_type', type=str, default='zscore')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--loss_rt', type=list, default=[0.1, 0.2, 0.4, 0.2, 0.1])
    parser.add_argument('--print_epoch', type=int, default=5)
    parser.add_argument('--val_inter_epoch', type=int, default=20)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # para get
    args = get_args()
    # config = laod_para(args)
    # seed set
    seed_set(args.seed)
    # log set
    args.log_file = args.oup_file + '/' + args.model_name + '/' \
                    + 'predays_' + str(args.day_inter) + '/' \
                    + args.src_city +'_' + args.trg_city + '/' \
                    + 'start_days_at_'+str(args.day_start) + '/' \
                    + str(args.seq_len)+'_'+str(args.pre_len)
    args.src_file = args.data_file + '/' + args.src_city + '_data.npz'
    args.trg_file = args.data_file + '/' + args.trg_city + '_data.npz'

    utils.dir_exist(args.log_file)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        filename=args.log_file+'/log.log')
    logger = logging.getLogger(__name__)
    
    # main
    main(**vars(args))
