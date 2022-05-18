#-*- encoding:utf-8 -*-
"""
This program will run in conda env wupy37
"""
import argparse
import logging

import torch
import json
from tqdm import trange,tqdm
import torch.nn as nn
import numpy as np
from sklearn import metrics
import time
import torchvision.models as models
import os
import transformers
from transformers import get_linear_schedule_with_warmup
import warnings
import random
import utils
from utils import build_loader
import models

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total: ',total_num, 'Trainable: ',trainable_num)
    #Model: CSAMM
    #Total:  249935962 Trainable:  249935962 2分类 显存占用：10317+7175M
    #Time cost: 10360.765762965195
    #Total:  249936265 Trainable:  249936265 5分类 显存占用：10305+7175M 10347+7217M
    #Time cost: 6129.022078233771
    #Model: multi_DRMM
    #Total:  139407578 Trainable:  139407578 5分类 显存占用：5311+3517M
    #Time cost: 2899.0146614946425
    #Total:  139179176 Trainable:  139179176 2分类 显存占用：5339+3517M
    #Time cost: 4847.417582822964

    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #if args.n_gpu > 0:
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    
def print_metrics(true_labels,pre_labels,average=None):
    """true_labels为正确的类别标签列表，pre_labels为正确的类别标签列表
    此处数据格式为tensor int"""
    s = ''
    if average == None:
        s = 'average=None:'
        print(s)
    elif average=='micro':
        s = 'average=micro:'
        print(s)
    elif average=='macro':
        s = 'average=macro:'
        print(s)
    elif average=='weighted':
        s = 'average=weighted:'
        print(s)
    p = metrics.precision_score(true_labels,pre_labels,average=average)
    r = metrics.recall_score(true_labels,pre_labels,average=average)
    f1 = metrics.f1_score(true_labels,pre_labels,average=average)
    s += 'P:{} R:{} F1:{}'.format(str(p),str(r),str(f1))
    s += '\n'
    print('P:{} R:{} F1:{}'.format(str(p),str(r),str(f1)))

    return s
    
def evaluation(model,args,dataloader,Test=False):
    
    if Test:
        print('Test the model and load the model:')
        color = "RED"
        description = "Testing"
        
    else:
        print('Select parameters on the dev dataset:')
        color = "GREEN"
        description = "Evaluating"
        
    criterion = nn.CrossEntropyLoss()
    model.eval()
    show_att = True
    with torch.no_grad():
        correct = 0
        total = 0
        dev_loss=[]
        pre_labels = None
        true_labels = None
        N = 0
        epoch_iterator = tqdm(dataloader, desc=description,colour=color)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            labels = batch[-1].cpu()
            
            if args.use_two_sentences:
                if args.use_frcnn_features:
                    outputs = model(batch[0],batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7],batch[8])
                else:
                    outputs = model(batch[0],batch[1],batch[2],batch[3],batch[4],batch[5],batch[6])
            else:
                if args.use_frcnn_features:
                    outputs = model(batch[0],batch[1],batch[2],batch[3],batch[4],batch[5])
                else:
                    outputs = model(batch[0],batch[1],batch[2],batch[3])
            
            pred_labels = outputs.cpu()
            
            loss = criterion(pred_labels, labels)
            _,pre = torch.max(pred_labels.data,1)
            
            total += labels.size(0)
            correct += (pre==labels).sum().item()
            dev_loss.append(loss.item())
            if step != 0:
                pre_labels = torch.cat((pre_labels,pre))
                true_labels = torch.cat((true_labels,labels))
            else:
                pre_labels = pre
                true_labels = labels
            N += 1
            if Test:
                if N==4*6:
                    print('pre_labels',pre_labels)
                    print('true_labels',true_labels)
            #if Test and show_att:
            #    show_att = False
            #    print(labels[0])
            #    print(att1[0].size(),att2[0].size(),att3[0].size())
            #    print(att1[0],att2[0],att3[0])

        print('total',total,'correct',correct)
        dev_loss = np.mean(dev_loss)
        if Test:
            print('Accuracy on Test:{:.2f}%'.format(100*correct/total))
            print('Test loss:', dev_loss)
        else:
            print('Accuracy on Dev:{:.2f}%'.format(100*correct/total))
            print('Dev loss:', dev_loss)
    if Test:
        print('metrics on Test:')
        s0 = 'total:'+str(total)+' correct:'+str(correct)+' Accuracy:'+str(metrics.accuracy_score(true_labels,pre_labels))+'\n'
        print_metrics(true_labels, pre_labels)
        print_metrics(true_labels, pre_labels, average='micro')
        print_metrics(true_labels, pre_labels, average='macro')
        print_metrics(true_labels, pre_labels, average='weighted')

    else:
        print('metrics on Dev:')
        print_metrics(true_labels,pre_labels)
        print_metrics(true_labels,pre_labels,average='micro')
        print_metrics(true_labels, pre_labels, average='macro')
        print_metrics(true_labels, pre_labels, average='weighted')
    return (dev_loss,metrics.accuracy_score(true_labels,pre_labels),true_labels,pre_labels)

def train(model,args,train_loader,dev_loader,test_loader):

    model_save_base_path = './checkpoints'
    model_save_path = ""
    if args.model_name=="CSAMM":
        model_save_path = os.path.join(model_save_base_path, "CSAMM")
    
    print("model_save_path",model_save_path)
    
    if len(os.listdir(model_save_path))!=0:
        print("save path is not empty! Romove the dir and rebuild!")
        import shutil
        shutil.rmtree(model_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    criterion = nn.CrossEntropyLoss()
    max_steps = -1
    num_train_epochs = 30
    #gradient_accumulation_steps = 1
    warmup_steps = 0
    weight_decay = 0.05#DMBERT中为0，Oscar中为0.05
    max_grad_norm = 1.0
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_loader) // \
                args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_loader) // args.gradient_accumulation_steps \
                * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    use_ods_labels = args.use_frcnn_ods or args.use_mrcnn_ods or args.use_scenes
    use_caption = args.use_oscar_ic_coco or args.use_oscar_ic
    #use_frcnn_features

    acc_last_improve_epoch = 0
    loss_last_improve_epoch = 0
    acc_last_improve_steps = 0
    loss_last_improve_steps = 0
    dev_best_acc = float('-inf')
    dev_best_loss = float('inf')
    Acc = False
    Loss = False
    global_step = 0
    train_iterator = trange(args.num_train_epochs, desc="Epoch")
    train_start = time.perf_counter()
    for epoch in train_iterator:
        model.train()
        loss_list = []
        train_correct = 0
        train_total = 0
        start = time.perf_counter()
        epoch_iterator = tqdm(train_loader, desc="Training",colour="BLUE")
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            if args.use_two_sentences:
                if args.use_frcnn_features:
                    labels = batch[9]
                    outputs = model(batch[0],batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7],batch[8])
                else:
                    labels = batch[7]
                    outputs = model(batch[0],batch[1],batch[2],batch[3],batch[4],batch[5],batch[6])
            else:
                if args.use_frcnn_features:
                    labels = batch[6]
                    outputs = model(batch[0],batch[1],batch[2],batch[3],batch[4],batch[5])
                else:
                    labels = batch[4]
                    outputs = model(batch[0],batch[1],batch[2],batch[3])
            pre_labels = outputs
            _, predicted = torch.max(pre_labels.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            loss = criterion(pre_labels, labels)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            loss_list.append(loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    eval_loss,eval_acc,_,_ = evaluation(model, args, dev_loader, Test=False)
                    if eval_acc > dev_best_acc:
                        print("Do test during training!")
                        dev_best_acc = eval_acc
                        best_steps = global_step
                        test_loss,test_acc,_,_ = evaluation(model, args, test_loader, Test=True)
                        print("Test loss:{}".format(test_loss))
                        print("Test acc:{}".format(test_acc))
                if(args.save_steps > 0 and global_step % args.save_steps == 0):
                    
                    model_save_name = os.path.join(model_save_path,"checkpoint-{}.ckpt".format(global_step))
                    print("Save model to {}".format(model_save_name))
                    torch.save(model.state_dict(), model_save_name)
    train_end = time.perf_counter()
    train_time = train_end - train_start
    checkpoints = os.listdir(model_save_path)
    print("Evaluate the following checkpoints: %s", checkpoints)
    test_best_acc = float('-inf')
    test_best_loss = float('inf')
    test_best_loss_acc = float('-inf')
    test_best_steps = 0
    test_loss_steps = 0
    test_acc_steps = 0
    for checkpoint in checkpoints:
        checkpoint = os.path.join(model_save_path,checkpoint)
        model.load_state_dict(torch.load(checkpoint))
        test_loss,test_acc,_,_ = evaluation(model, args, test_loader, Test=True)
        if test_loss<test_best_loss:
            test_best_loss = test_loss
            test_best_loss_acc = test_acc
            test_loss_steps = int(checkpoint.split("/")[-1].split("-")[-1].split(".")[0])
        if test_acc>test_best_acc:
            test_best_acc = test_acc
            test_acc_steps = int(checkpoint.split("/")[-1].split("-")[-1].split(".")[0])
    if test_best_loss_acc>test_best_acc:
        test_best_acc = test_best_loss_acc
        test_best_steps = test_loss_steps
    else:
        test_best_steps = test_acc_steps
    model_best_checkpoint = os.path.join(model_save_path,"checkpoint-{}.ckpt".format(test_best_steps))
    model.load_state_dict(torch.load(model_best_checkpoint))
    _,_,true_labels,pre_labels = evaluation(model, args, test_loader, Test=True)
    best_model_save_path = os.path.join(model_save_base_path, "best_model_checkpoint")
    best_model_save_name = os.path.join(best_model_save_path,"{}-{}.ckpt".format(args.model_name,args.task_name))
    torch.save(model.state_dict(), best_model_save_name)
    
    s0 = ' Accuracy:'+str(metrics.accuracy_score(true_labels,pre_labels))+'\n'
    s1 = print_metrics(true_labels, pre_labels)
    s2 = print_metrics(true_labels, pre_labels, average='micro')
    s3 = print_metrics(true_labels, pre_labels, average='macro')
    s4 = print_metrics(true_labels, pre_labels, average='weighted')
    s = 'task:' + args.task_name + ' ' + 'model:' + args.model_name +" Training time:"+ str(train_time)+"\n"
    
    s += "Use ods labels:"
    if args.use_frcnn_ods:
        s += "use_frcnn_ods"
    elif args.use_mrcnn_ods:
        s += "use_mrcnn_ods"
    if args.use_oscar_ic:
        s += "use_oscar_ic"
    elif args.use_oscar_ic_coco:
        s += "use_oscar_ic_coco"
    if args.use_scenes:
        s += "use_scenes"
    if (not use_ods_labels) and (not use_caption):
        s += "None"
    s += "\n"
    s = s + s0 + s1 + s2 + s3 + s4

    s5 = 'store time:'+utils.get_asctime()+'\n'
    s = s5+s
    with open('./results_save.txt','a+') as f:#'a+'打开一个文件用于读写。若该文件已存在，文件指针将会放在文件的结尾。文件打开时会是追加模式。若文件不存在，创建新文件用于读写。
        f.write(s)
    test_end = time.perf_counter()
    print("best step",test_best_steps)
    print('Time cost:',test_end-train_start)




def main(model_name):
    """This function is from Swin transformer"""
    parser = argparse.ArgumentParser('CrisisMMD training,evaluation and testing script', add_help=False)

    # easy config modification
    
    parser.add_argument('--data_path', type=str,default="./data",help='The input data dir with all required files.')
    #--task_name Humanitarian
    parser.add_argument('--task_name', type=str,default="Informative",help='')
    parser.add_argument('--use_newdata',default=False,action='store_true',help='use data ')
    parser.add_argument("--output_dir", default='./output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument('--bert_path',default="./bert_pretrain",help='path to load pretrained BERT')
    #文本长度
    parser.add_argument("--max_seq_length", default=50, type=int,
                        help="The maximum text input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    
    #图像标签序列  长度 this is from Oscar
    #
    parser.add_argument("--use_two_sentences", default=False,action='store_true', help='Original sentence and added features sentence.')
    parser.add_argument("--use_oscar_ic", default=False,action='store_true', help='use oscar crisis finetue image captions')
    parser.add_argument("--use_oscar_ic_coco", default=False,action='store_true', help='Oscar image captions')
    parser.add_argument("--use_ics", default=False,action='store_true', help='image captions')
    parser.add_argument("--use_mrcnn_ods", default=False,action='store_true', help='mrcnn objects')
    parser.add_argument("--use_frcnn_ods", default=False,action='store_true', help='frcnn objects')
    parser.add_argument("--use_scenes", default=False,action='store_true', help='scenes')
    parser.add_argument("--remove_same_tags", default=False,action='store_true', help='remove same tags in object labels')
    parser.add_argument("--use_frcnn_features", default=False,action='store_true', help='frcnn features')
    
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help= "Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=3,
                        help="The max number of masked tokens per sentence.")
                  
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument('--batch_size', type=int,default=32,help="batch size for single GPU")
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=150, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=150, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--num_train_epochs", default=30, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument("--max_steps", default=-1, type=int, 
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument("--seed", type=int, default=2020, help="random seed for initialization")
    parser.add_argument("--num_workers", default=8, type=int, help="Workers in dataloader.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training.")#-1,0
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    
    args, unparsed = parser.parse_known_args()#args = parser.parse_args()
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    
    #Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )
    
    # Set seed
    set_seed(args.seed)
    
    #Load data
    data_loader_train,data_loader_dev,data_loader_test = build_loader(args)
    print(args.task_name)
    if args.task_name=="Informative":
        num_classes = 2
    elif args.task_name=="Humanitarian":
        num_classes = 5
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_name = model_name
    print("Model:",args.model_name)
    if args.model_name == "CSAMM":
        model = models.CSAMM(num_classes,args.bert_path)
    
    model = nn.DataParallel(model)
    model.to(args.device)
    get_parameter_number(model)
    train(model,args,data_loader_train,data_loader_dev,data_loader_test)
if __name__=="__main__":
    main(model_name="CSAMM")