#-*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from transformers import BertModel,BertTokenizer,ViTModel
import timm
from timm.models.vision_transformer import VisionTransformer,vit_small_patch16_224,vit_base_patch16_224,vit_large_patch16_224

#text models
class text_BERT(nn.Module):
    def __init__(self, num_classes,bert_path):
        super(text_BERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.02)
        self.fc1 = nn.Linear(768, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(768, num_classes)
    def forward(self, tokens_ids,mask,segment_ids,image_data):
        bert_out = self.bert(tokens_ids,attention_mask=mask)
        pooled = bert_out[1]
        
        text_image = pooled
        #text_image = self.dropout1(text_image)
        #text_image = self.fc1(text_image)
        #text_image = self.dropout2(text_image)
        #text_image = self.fc2(text_image)
        #text_image = self.dropout3(text_image)
        text_image = self.fc3(text_image)
        return text_image
        
class image_ViTs(nn.Module):#0.906 验证集上最佳0.899 训练轮次13 去除之后pad均为64，90.29

    def __init__(self, num_classes,bert_path):
        super(image_ViTs, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.vit = vit_small_patch16_224(pretrained=True)#输出是1000
        self.vit_fc = nn.Linear(1000,768)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.02)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, tokens_ids,mask,segment_ids,image_data):
        bert_out = self.bert(tokens_ids, attention_mask=mask,token_type_ids=segment_ids)#[batch_size,768]
        pooled = bert_out[1]
        image = self.vit(image_data)
        #image = self.vit_fc(image)

        image_text = torch.cat((image, pooled), -1)
        image_text = self.dropout1(image)
        image_text = self.fc1(image_text)
        image_text = self.dropout2(image_text)
        image_text = self.fc2(image_text)
        image_text = self.dropout3(image_text)
        image_text = self.fc3(image_text)
        return image_text  
class image_ViTb(nn.Module):#0.906 验证集上最佳0.899 训练轮次13 去除之后pad均为64，90.29

    def __init__(self, num_classes,bert_path):
        super(image_ViTb, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.vit = vit_base_patch16_224(pretrained=True)#输出是1000
        self.vit_fc = nn.Linear(1000,768)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.02)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, tokens_ids,mask,segment_ids,image_data):
        bert_out = self.bert(tokens_ids, attention_mask=mask,token_type_ids=segment_ids)#[batch_size,768]
        pooled = bert_out[1]
        image = self.vit(image_data)
        #image = self.vit_fc(image)

        image_text = torch.cat((image, pooled), -1)
        image_text = self.dropout1(image)
        image_text = self.fc1(image_text)
        image_text = self.dropout2(image_text)
        image_text = self.fc2(image_text)
        image_text = self.dropout3(image_text)
        image_text = self.fc3(image_text)
        return image_text 

class multi_BERT_ViTs_s(nn.Module):

    def __init__(self, num_classes,bert_path):
        super(multi_BERT_ViTs_s, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.vit = vit_small_patch16_224(pretrained=True)
        self.vit_fc = nn.Linear(1000,768)
        #self.tvit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.02)
        self.fc1 = nn.Linear(768+768, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, tokens_ids,mask,segment_ids,image_data):
        bert_out = self.bert(tokens_ids, attention_mask=mask,token_type_ids=segment_ids)#[batch_size,768]
        #pooled = bert_out.pooler_output
        pooled = bert_out[1]
        image = self.vit(image_data)
        image = self.vit_fc(image)

        image_text = torch.cat((image, pooled), -1)
        image_text = self.dropout1(image_text)
        image_text = self.fc1(image_text)
        image_text = self.dropout2(image_text)
        image_text = self.fc2(image_text)
        image_text = self.dropout3(image_text)
        image_text = self.fc3(image_text)
        return image_text  
        
class SelfAttention(nn.Module):
    def __init__(self,hidden_size=768,num_attention_heads=12,attention_probs_dropout_prob=0.1,position_embedding_type=None):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        attention_head_size = int(hidden_size/num_attention_heads)
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads*attention_head_size

        self.query = nn.Linear(hidden_size,self.all_head_size)
        self.key = nn.Linear(hidden_size,self.all_head_size)
        self.value = nn.Linear(hidden_size,self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if(is_cross_attention):
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores,dim=-1)
        context_layer = torch.matmul(attention_probs,value_layer)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer,attention_probs) if output_attentions else (context_layer,)
        return outputs

class SelfOutput(nn.Module):
    def __init__(self,hidden_size=768,layer_norm_eps=1e-12,hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size,hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size,eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self,hidden_states,input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        #print("input_tensor.size()",input_tensor.size())
        #print("hidden_states.size()",hidden_states.size())
        hidden_states = self.LayerNorm(hidden_states+input_tensor)
        return hidden_states

class Attention(nn.Module):
    def __init__(self,hidden_size=768,num_attention_heads=12,attention_probs_dropout_prob=0.1,
                 position_embedding_type=None):
        super().__init__()
        self.self = SelfAttention(hidden_size,num_attention_heads,attention_probs_dropout_prob,position_embedding_type)
        self.output = SelfOutput()
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions=True
        )
        attention_output = self.output(self_outputs[0],hidden_states)
        outputs = (attention_output,)+self_outputs[1:]
        return outputs
        
class Pooler(nn.Module):
    def __init__(self,hidden_size=768):
        super().__init__()
        self.dense = nn.Linear(hidden_size,hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self,hidden_states):
        first_token_tensor = hidden_states[:,0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
        
class CrossAttentionEncoder(nn.Module):
    def __init__(self,hidden_size=768,num_attention_heads=12,attention_probs_dropout_prob=0.1,
                 position_embedding_type=None):
        super().__init__()
        self.attention = Attention(hidden_size=hidden_size,num_attention_heads=num_attention_heads,attention_probs_dropout_prob=attention_probs_dropout_prob,
                 position_embedding_type=position_embedding_type)
        self.pooler = Pooler(hidden_size=hidden_size)
    def forward(self,hidden_states1,hidden_states2):
        cross_attn = self.attention(hidden_states=hidden_states1,encoder_hidden_states=hidden_states2)
        cross_attn_pooled = self.pooler(cross_attn[0])
        return cross_attn_pooled

#two sentences
class CSAMM(nn.Module):
    def __init__(self, num_classes,bert_path):
        super(CSAMM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.vit = vit_small_patch16_224(pretrained=True)#输出是1000
        self.vit_fc = nn.Linear(1000,768)
        self.tvit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        self.cross_attention_encoder = CrossAttentionEncoder(num_attention_heads=12)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768*4, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, tokens_ids1,mask1,segment_ids1,tokens_ids2,mask2,segment_ids2,image_data):
        
        bert_out1 = self.bert(tokens_ids1,attention_mask=mask1,token_type_ids=segment_ids1)
        pooled1 = bert_out1[1]
        hidden_states1 = bert_out1[0]
        
        bert_out2 = self.bert(tokens_ids2,attention_mask=mask2,token_type_ids=segment_ids2)
        pooled2 = bert_out2[1]
        hidden_states2 = bert_out2[0]
        
        tvit_output = self.tvit(image_data)
        tvit_hidden_states = tvit_output[0]
        
        cross_attn_pooled1 = self.cross_attention_encoder(hidden_states1=hidden_states1,hidden_states2=tvit_hidden_states)
        cross_attn_pooled2 = self.cross_attention_encoder(hidden_states1=hidden_states2,hidden_states2=tvit_hidden_states)
        cross_attn_pooled3 = self.cross_attention_encoder(hidden_states1=hidden_states1,hidden_states2=hidden_states2)
     
        image = self.vit(image_data)
        image = self.vit_fc(image)
        cross_attn_pooled_t = cross_attn_pooled3*cross_attn_pooled2
        image_text = torch.cat((image, pooled1,cross_attn_pooled_t,cross_attn_pooled1), -1)
        image_text = self.dropout1(image_text)
        image_text = self.fc1(image_text)
        image_text = self.dropout2(image_text)
        image_text = self.fc2(image_text)
        image_text = self.dropout3(image_text)
        image_text = self.fc3(image_text)

        return image_text