#-*- encoding:utf-8 -*-
import json
import time
import os
import torch
from transformers import BertModel,BertTokenizer
import torchvision.transforms as transforms
import pickle
import numpy as np
from PIL import Image

def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),#将图片短边缩放至x，长宽比保持不变。
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )
    
class MultiBertDataset(torch.utils.data.Dataset):

    def __init__(self,args,texts,image_paths,labels,texts_ods=None,texts_ods_len=40):
        self.use_frcnn_features = args.use_frcnn_features
        self.use_two_sentences = args.use_two_sentences
        self.texts = texts
        self.ids = []
        self.image_paths = image_paths
        self.labels = labels
        self.texts_ods = texts_ods
        self.task_name = args.task_name
        self.transforms = get_transforms()
        bert_path = args.bert_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_seq_a_len = 50#60得到90.68 70得到90.94 80得到90.68
        #old 70得到90.74 50得到90.94 60得到90.87
        self.max_seq_b_len = texts_ods_len#od40(实际上最多为8种类型) scenes40 ic25 frcnn：60

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        #文本
        
        if self.texts_ods!=None:
            text_o = self.texts_ods[item]
        else:
            text_o = None
        text = self.texts[item]
        if self.use_two_sentences and self.texts_ods!=None:
            #使用两条句子表示文本信息时必须使用额外特征
            sent1_tokens_ids,sent1_attention_mask,sent1_segment = self.bert_text2ids(text=text,
                                                                                     text_len=self.max_seq_a_len)
            sent2_tokens_ids,sent2_attention_mask,sent2_segment = self.bert_text2ids(text=text_o,
                                                                                     text_len=self.max_seq_b_len)
        else:
            tokens_ids,attention_mask,segment = self.bert_text2ids(text=text,
                                                                   text_len=self.max_seq_a_len,
                                                                   text_b=text_o,
                                                                   text_b_len=self.max_seq_b_len)#Informative任务的句子分词最大数47/42/39
        
        #图片
        #从npy文件读入
        image_path = self.image_paths[item]
        npy_path = image_path.split('/')[-1]
        img_name = npy_path[:-4]
        if self.task_name.lower()=='informative':
            npy_path = './image_data_numpy/informative/' + img_name + '.npy'
        else:
            npy_path = './image_data_numpy/humanitarian/' + img_name + '.npy'
        with open(npy_path, 'rb') as f:
            img_npy = pickle.load(f, encoding='iso-8859-1')
        # 转化为Tensor
        image_npy = torch.tensor(img_npy, dtype=torch.float)#shape为[224,224,3]
        # 模型要的输入为[3,224,224]
        image_npy = image_npy.permute(2, 0, 1)#调整维度顺序
        
        #读入FRCNN特征
        frcnn_features_path = './features/frcnn/'+'features/'+img_name+ '_frcnn.npy'
        frcnn_spatial_path = './features/frcnn/spatial_features/' + img_name + '_frcnn_spatial.npy'
        with open(frcnn_features_path, 'rb') as f:
            frcnn_features = pickle.load(f, encoding='iso-8859-1')
        with open(frcnn_spatial_path, 'rb') as f:
            frcnn_spatial = pickle.load(f, encoding='iso-8859-1')
        frcnn_features = torch.tensor(frcnn_features[0],dtype=torch.float)#[1,36,2048]->[36,2014]
        frcnn_spatial = torch.tensor(frcnn_spatial,dtype=torch.float)#[36,6]
        
        #使用PIL读入
        image_PIL = Image.open(image_path).convert("RGB")
        image_PIL = self.transforms(image_PIL)
        image_PIL = torch.tensor(image_PIL, dtype=torch.float)#shape为[224,224,3]
        
        #标签
        image_label = self.labels[item]
        image_label = torch.tensor(image_label, dtype=torch.long)
        if self.use_two_sentences and self.texts_ods!=None:
            if self.use_frcnn_features:
                return sent1_tokens_ids,sent1_attention_mask,sent1_segment,sent2_tokens_ids,sent2_attention_mask,\
                sent2_segment,image_npy,frcnn_features,frcnn_spatial, image_label
            else:
                return sent1_tokens_ids,sent1_attention_mask,sent1_segment,sent2_tokens_ids,sent2_attention_mask,\
                sent2_segment,image_npy,image_label
        else:
            if self.use_frcnn_features:
                return tokens_ids,attention_mask,segment,image_npy,frcnn_features,frcnn_spatial, image_label
            else:
                return tokens_ids,attention_mask,segment,image_npy,image_label

    def bert_text2ids(self,text,text_len,text_b=None,text_b_len=0,cls_token_segment_id=0, pad_token_segment_id=0,sequence_a_segment_id=0,sequence_b_segment_id=1):
        """将一条句子文本用BERT分词并转换为BERT词表id"""
        max_seq_a_len = text_len
        max_seq_b_len = text_b_len
        
        tokens = self.tokenizer.tokenize(text)
        cls = self.tokenizer.cls_token
        sep = self.tokenizer.sep_token
        if len(tokens) > max_seq_a_len - 2:
            tokens = tokens[: (max_seq_a_len  - 2)]
        tokens = [cls] + tokens + [sep]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b!=None:
            padding_a_len = max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)
            #len(tokens)=self.max_seq_a_len
            
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > max_seq_b_len - 1:
                tokens_b = tokens_b[: (max_seq_b_len - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        seq_len = len(tokens)
        if text_b!=None:
            max_len = max_seq_a_len + max_seq_b_len
        else:
            max_len = max_seq_a_len
        padding_len = max_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        attention_mask = torch.zeros(max_len, dtype=torch.long)
        a_start, a_end = 0, seq_a_len
        attention_mask[a_start: a_end] = 1
        
        if text_b!=None:
            b_start, b_end = max_seq_a_len, seq_len
            attention_mask[b_start: b_end] = 1
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        #print(input_ids.size())
        #print(attention_mask.size())torch.Size([180])
        return input_ids,attention_mask,segment_ids

        
def get_asctime():
    """"获取时间，将'Sat Mar 27 16:12:58 2021'转换为'2021-Mar-27-16_12_58'"""
    t = time.asctime()
    #print(t)
    t = t.split(' ')
    #print(t)
    t = t[-1]+' '+t[1]+' '+t[2]+' '+t[3]
    #print(t)
    t = t.replace(' ','-')
    #print(t)
    t = t.replace(':','_')
    #print(t)
    return t
def load_json(path):
    """从json格式文件中读入数据并返回"""
    """json文件中是python列表"""
    with open(path,'r') as f:
        data = json.load(f)
    return data

def save_json_data(data, path):
    with open(path, "w") as f:
        json.dump(data, f)
        
def read_texts_ids_paths_labels(file):

    """从json文件中读入数据，json文件是对齐的0：文本句子:1：文本句子对应的预训练词表id，2：图片的路径，3：one-hot格式的标签"""
    data = load_json(file)
    texts = data[0]#英文单词组成的句子
    #tokens_ids = data[1]#将每个单词替换为词表索引组成的序列
    image_paths = data[2]#句子对应图片的路径0
    one_hot_label = data[3]#标签 二分类对应[informative,not_informative] 五分类对应[affected_individuals,infrastructure_and_utility_damage,not_humanitarian,other_relevant_information,rescue_volunteering_or_donation_effort]

    label = []
    for l in one_hot_label:
        for i in range(len(l)):
            if l[i]==1:
                label.append(i)

    return (texts,image_paths,label)#[句子，句子对应的图片路径，句子对应的标签]
    
def get_data(args,filetdt="train"):
    """从json文件中读入数据，返回句子，句子对应的图片路径，句子对应的标签"""
    task_name = args.task_name.lower()
    #texts imgs labels
    if args.use_newdata:
        data_base_dir = "./data/data_20211220"
    else:
        data_base_dir = "./data/data_20210422"#
        
    data_file = os.path.join(data_base_dir,"{}_{}_data.json".format(task_name,filetdt))
    print("Loading data from:",data_file)
    sents,image_paths,labels = read_texts_ids_paths_labels(data_file)
    print("max_len",get_maxlen(args,sents))
    return (sents,image_paths,labels)

def read_data(path):
    with open(path,"r") as f:
        data = f.read()
        data = data.split("\n")
    print(len(data))
    return data
    
def get_maxlen(args,sent_list):
    bert_path = args.bert_path
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    l = 0
    for i in range(len(sent_list)):
        tokens = tokenizer.tokenize(sent_list[i])
        l = max(l,len(tokens))
    return l

def setLabels(args,sent_list):
    new_sent_list = []
    for i in range(len(sent_list)):
        sent = sent_list[i].split()
        sent = list(set(sent))
        new_sent_list.append(" ".join(sent))
    return new_sent_list
        
def get_scenes(args):
    if args.task_name=="Informative":
        train_scenes = "./json_file/Informative_train_scenes.json"
        dev_scenes = "./json_file/Informative_dev_scenes.json"
        test_scenes = "./json_file/Informative_test_scenes.json"
    train_scenes = load_json(train_scenes)
    dev_scenes = load_json(dev_scenes)
    test_scenes = load_json(test_scenes)
    
    #移除重复标签
    if args.remove_same_tags:
        train_scenes = setLabels(args,train_scenes)
        dev_scenes = setLabels(args,dev_scenes)
        test_scenes = setLabels(args,test_scenes)
    
    train_len = get_maxlen(args,train_scenes)
    dev_len = get_maxlen(args,dev_scenes)
    test_len = get_maxlen(args,test_scenes)
    print("Scenes nums:",train_len,dev_len,test_len)
    return train_scenes,dev_scenes,test_scenes,max(train_len,dev_len,test_len)
    
def get_object(args):
    if args.task_name.lower()=="informative":
        #不移除重复标签为90.38
        train_ob = "./json_file/Informative_train_obj.json"
        dev_ob = "./json_file/Informative_dev_obj.json"
        test_ob = "./json_file/Informative_test_obj.json"
    train_ob = load_json(train_ob)
    dev_ob = load_json(dev_ob)
    test_ob = load_json(test_ob)
    
    #移除重复标签
    if args.remove_same_tags:
        train_ob = setLabels(args,train_ob)
        dev_ob = setLabels(args,dev_ob)
        test_ob = setLabels(args,test_ob)
    
    train_len = get_maxlen(args,train_ob)
    dev_len = get_maxlen(args,dev_ob)
    test_len = get_maxlen(args,test_ob)
    print("Objects nums:",train_len,dev_len,test_len)
    return train_ob,dev_ob,test_ob,max(train_len,dev_len,test_len)
    
def get_frcnn_object(args):
    if args.task_name=="Informative":
        train_ob = "./json_file/Informative_frcnn_objs_train.json"
        dev_ob = "./json_file/Informative_frcnn_objs_dev.json"
        test_ob = "./json_file/Informative_frcnn_objs_test.json"
    train_ob = load_json(train_ob)
    dev_ob = load_json(dev_ob)
    test_ob = load_json(test_ob)
    
    #移除重复标签
    if args.remove_same_tags:
        train_ob = setLabels(args,train_ob)
        dev_ob = setLabels(args,dev_ob)
        test_ob = setLabels(args,test_ob)
    
    train_len = get_maxlen(args,train_ob)
    dev_len = get_maxlen(args,dev_ob)
    test_len = get_maxlen(args,test_ob)
    print("FRCNN objects nums:",train_len,dev_len,test_len)
    return train_ob,dev_ob,test_ob,max(train_len,dev_len,test_len)
    
def get_ics(args):
    if args.task_name=="Informative":
        train_ics = "./pre_imagecaption_file/Informative_pretrained_imagecaption_train.json"
        dev_ics = "./pre_imagecaption_file/Informative_pretrained_imagecaption_dev.json"
        test_ics = "./pre_imagecaption_file/Informative_pretrained_imagecaption_test.json"
    train_ics = load_json(train_ics)
    dev_ics = load_json(dev_ics)
    test_ics = load_json(test_ics)
    train_len = get_maxlen(args,train_ics)
    dev_len = get_maxlen(args,dev_ics)
    test_len = get_maxlen(args,test_ics)
    print("Imagecaption nums:",train_len,dev_len,test_len)
    return train_ics,dev_ics,test_ics,max(train_len,dev_len,test_len)
    
def oscar_captions(oscar_dict_list):
    oscar_captions = []
    for i in range(len(oscar_dict_list)):
        caption = oscar_dict_list[i]["caption"]
        oscar_captions.append(caption)
    return oscar_captions
    
def get_oscar_ics(args):
    if args.task_name.lower()=="informative":#train_train.json:91.20
        if args.use_oscar_ic:
            train_ics = "./json_file/oscar_caption/train_train.json"
            dev_ics = "./json_file/oscar_caption/train_dev.json"
            test_ics = "./json_file/oscar_caption/train_test.json"
        elif args.use_oscar_ic_coco:
            #训练
            #train_ics = "./json_file/oscar_caption/train_coco_In_max20_train.json"
            #dev_ics = "./json_file/oscar_caption/train_coco_In_max20_dev.json"
            #test_ics = "./json_file/oscar_caption/train_coco_In_max20_test.json"
            #微调
            train_ics = "./json_file/oscar_caption/finetune_coco_In_max20_train.json"
            dev_ics = "./json_file/oscar_caption/finetune_coco_In_max20_dev.json"
            test_ics = "./json_file/oscar_caption/finetune_coco_In_max20_test.json"
    else:
        if args.use_oscar_ic:
            train_ics = "./json_file/oscar_caption/crisis_human_train30_train.json"
            dev_ics = "./json_file/oscar_caption/crisis_human_train30_dev.json"
            test_ics = "./json_file/oscar_caption/crisis_human_train30_test.json"
        elif args.use_oscar_ic_coco:
            #训练
            #train_ics = "./json_file/oscar_caption/train_coco_In_max20_train.json"
            #dev_ics = "./json_file/oscar_caption/train_coco_In_max20_dev.json"
            #test_ics = "./json_file/oscar_caption/train_coco_In_max20_test.json"
            #微调
            train_ics = "./json_file/oscar_caption/coco_human_finetune20_train.json"
            dev_ics = "./json_file/oscar_caption/coco_human_finetune20_dev.json"
            test_ics = "./json_file/oscar_caption/coco_human_finetune20_test.json"
    train_ics = load_json(train_ics)
    dev_ics = load_json(dev_ics)
    test_ics = load_json(test_ics)
    #[{"image_id": "data_image/hurricane_harvey/15_9_2017/908804643064496128_0.jpg", \
    #"caption": "if irma doesn kill me tryna go on tour with"},...]
    train_ics = oscar_captions(train_ics)
    dev_ics = oscar_captions(dev_ics)
    test_ics = oscar_captions(test_ics)
    print("train_ics[0]",train_ics[0])
    train_len = get_maxlen(args,train_ics)
    dev_len = get_maxlen(args,dev_ics)
    test_len = get_maxlen(args,test_ics)
    print("Imagecaption nums:",train_len,dev_len,test_len)
    return train_ics,dev_ics,test_ics,max(train_len,dev_len,test_len)
    
def get_one(data1,data2):
    assert len(data1)==len(data2)
    new_data = []
    for i in range(len(data1)):
        sent = data1[i] + " " + data2[i]
        new_data.append(sent)
    return new_data
    
def get_od_labels(args):
    if args.use_mrcnn_ods:
        o_train_data,o_dev_data,o_test_data,o_len = get_object(args)
        if args.use_scenes:
            o_train_data1,o_dev_data1,o_test_data1,o_len1 = get_scenes(args)
            o_train_data,o_dev_data,o_test_data = get_one(o_train_data,o_train_data1),get_one(o_dev_data,o_dev_data1),get_one(o_test_data,o_test_data1)
            o_len = o_len+o_len1
        #print("o_train_data[6]",o_train_data[6])
    elif args.use_frcnn_ods:
        o_train_data,o_dev_data,o_test_data,o_len = get_frcnn_object(args)
        if args.use_scenes:
            o_train_data1,o_dev_data1,o_test_data1,o_len1 = get_scenes(args)
            o_train_data,o_dev_data,o_test_data = get_one(o_train_data,o_train_data1),get_one(o_dev_data,o_dev_data1),get_one(o_test_data,o_test_data1)
            o_len = o_len + o_len1
    elif args.use_scenes:
        o_train_data,o_dev_data,o_test_data,o_len = get_scenes(args)
    elif args.use_ics or args.use_oscar_ic or args.use_oscar_ic_coco:
        o_train_data,o_dev_data,o_test_data,o_len = get_oscar_ics(args)
    else:
        o_train_data,o_dev_data,o_test_data,o_len = None,None,None,0
    if o_len!=0:
        o_len = o_len//10*10 + 10
    print("o_len",o_len)
    return o_train_data,o_dev_data,o_test_data,o_len
    
def build_loader(args):
    train_sents,train_img_paths,train_labels = get_data(args,filetdt="train")
    dev_sents,dev_img_paths,dev_labels = get_data(args,filetdt="dev")
    test_sents,test_img_paths,test_labels = get_data(args,filetdt="test")
    
    #ods_labels
    o_train_data,o_dev_data,o_test_data,o_len = get_od_labels(args)
    
    dataset_train = MultiBertDataset(args,train_sents,train_img_paths,train_labels,o_train_data,o_len)
    dataset_dev = MultiBertDataset(args,dev_sents,dev_img_paths,dev_labels,o_dev_data,o_len)
    dataset_test = MultiBertDataset(args,test_sents,test_img_paths,test_labels,o_test_data,o_len)
    model_add_BN = False
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=model_add_BN,
    )
    data_loader_dev = torch.utils.data.DataLoader(
        dataset_dev,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return data_loader_train,data_loader_dev,data_loader_test
    
if __name__=="__main__":
    base = "./data/data_20210422/"
    train_file=base+"informative_train_data.json"
    train_texts,train_paths,train_labels = read_texts_ids_paths_labels(train_file)
    oscar_train = "./json_file/oscar_caption/finetune_train.json"
    oscar_train = load_json(oscar_train)
    #[{"image_id": "data_image/hurricane_harvey/15_9_2017/908804643064496128_0.jpg", \
    #"caption": "if irma doesn kill me tryna go on tour with"},...]
    for i in range(len(train_paths)):
        path = train_paths[i]
        oscar = oscar_train[i]#
        if path == oscar["image_id"]:
            print(i)
    