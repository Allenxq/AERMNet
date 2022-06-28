import torch
import torch.nn as nn
import torchvision
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# encoder    
class Encoder(nn.Module):

    def __init__(self, encoded_image_size=7):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet101(pretrained=False)  # pretrained ImageNet ResNet-101
        pretrain_path = '/zengxh_fix/wzq/My_models/resnet101-5d3b4d8f.pth'
        resnet.load_state_dict(torch.load(pretrain_path))
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: 图像张量(batch_size, 3, image_size, image_size)
        :return: 编码后的图像
        """
        out = self.resnet(images)  
        out = self.adaptive_pool(out)  
        out = out.permute(0, 2, 3, 1)  
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = fine_tune
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class AOAttention(nn.Module):
    def __init__(self,  D):  
        super(AOAttention, self).__init__()
        self.W_q = nn.Linear(2 * D, D, bias=True)
        self.W_v = nn.Linear(2 * D, D, bias=True)
        self.sigmoid = nn.Sigmoid()
    def f_similarity(self,target1, behaviored1): 
        scores = torch.zeros(target1.size(0), behaviored1.size(1))
        for j in range(target1.size(0)):
            target = target1[j]
            behaviored = behaviored1[j]
            attention_distribution = []
            for i in range(behaviored.size(0)):
                attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))
                attention_distribution.append(attention_score)
            attention_distribution = torch.Tensor(attention_distribution)
            score = attention_distribution / torch.sum(attention_distribution, 0)
            scores[j] = score
        return scores
    def Att(self,Q, K, V):
        V_ = torch.zeros(Q.size(0), Q.size(1), V.size(2))
        V_=V_.to(device) 
        soft = nn.Softmax(dim=1)
        for i in range((Q.size(1))):
            alpha_i_j = soft(self.f_similarity(Q[:, i, :].unsqueeze(1), K)) 
            alpha_i_j = alpha_i_j.unsqueeze(2)  
            alpha_i_j = alpha_i_j.expand_as(V)  
            alpha_i_j=alpha_i_j.to(device)
            V_[:, i, :] = torch.sum(alpha_i_j.mul(V), dim=1)  
        return V_  
    def forward(self,Q,K,V):
        V_=self.Att(Q,K,V)
        s = torch.cat((V_, Q), 2)
        image_feats_on_att= self.sigmoid(self.W_v(s)).mul(self.W_q(s))
        return image_feats_on_att

class multi_head_att(nn.Module):
    def __init__(self, D):
        super(multi_head_att, self).__init__()
        self.W_Q = nn.Linear(D, D)
        self.W_K = nn.Linear(D, D)
        self.W_V = nn.Linear(D, D)
        self.softmax = nn.Softmax(dim=2)
        self.layernorm=nn.LayerNorm(D,eps=1e-05, elementwise_affine=True)
        self.AOAttention=AOAttention(D)
    def slice(self, Q, K, V):  
        d = Q.size(2) // 8
        list_Q = []
        list_k = []
        list_v = []
        for i in range(8):
            tempQ=Q[:,:,d * i:d * (i + 1)]
            tempK=K[:,:,d * i:d * (i + 1)]
            tempV=V[:,:,d * i:d * (i + 1)]
            list_Q.append(tempQ)
            list_k.append(tempK)
            list_v.append(tempV)

        return list_Q,list_k,list_v

    def forward(self,B,A):
        list_Q,list_k,list_v=self.slice(self.W_Q(B),self.W_K(A),self.W_V(A))
        head=[]
        for i in range(len(list_v)):
            f_dot_att=self.softmax(list_Q[i].matmul(list_k[i].permute(0,2,1))/8)
            f_dot_att=f_dot_att.matmul(list_v[i])
            head.append(f_dot_att)
        f_mh_att=torch.cat((head[0],head[1],head[2],head[3],head[4],head[5],head[6],head[7]),dim=2)
        AOAttention = self.AOAttention( self.W_Q(B) ,f_mh_att ,f_mh_att)
        return  self.layernorm(AOAttention+A)

class MogrifierLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mogrify_steps,k=5):
        super(MogrifierLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])  # q
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])  # r

    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i + 1) % 2 == 0:
                h = (2 * torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2 * torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct

# not working   
class X_Linear_block(nn.Module):
    def __init__(self,N=196,D_q=1024,D_k=1024,D_v=1024,D_c=512,D_B=1024):
        super(X_Linear_block, self).__init__()
        self.N=N
        self.D_q = D_q
        self.D_k = D_k
        self.D_v = D_v
        self.D_c = D_c
        self.D_B = D_B
        self.ELU=nn.ELU()
        self.RELU=nn.ReLU()
        self.softMax = nn.Softmax(dim=1)
        self.sigMoid=nn.Sigmoid()
        self.laynorm = nn.LayerNorm(D_B)
        self.W_k=nn.Linear(D_k,D_B)
        self.W_q_k=nn.Linear(D_q,D_B)
        self.W_k_B=nn.Linear(D_B,D_c)
        self.W_b=nn.Linear(D_c,1)
        self.W_e=nn.Linear(D_c,D_B)
        self.W_v=nn.Linear(D_v,D_B)
        self.W_v_q=nn.Linear(D_q,D_B)
        self.W_k_m = nn.Linear(D_v + D_k, D_k)
        self.W_v_m = nn.Linear(D_v + D_v, D_v)

    def forward(self, Q,K,V):
        batch=Q.size(0)
        B_k=torch.zeros(batch, self.N, self.D_B).to(device) 
        B_k_pie=torch.zeros(batch, self.N, self.D_c).to(device) 
        b_s=torch.zeros(batch,self.N).to(device) 
        B_v=torch.zeros(batch, self.N, self.D_B).to(device) 
        B_k=torch.mul(self.ELU(self.W_k(K)),self.ELU(self.W_q_k(Q).unsqueeze(1)))   
        B_k_pie=self.RELU(self.W_k_B(B_k))
        b_s=self.W_b(B_k_pie).squeeze(2)       
        beita_s=self.softMax(b_s)  
        B_gang=torch.mean(B_k_pie,1)     
        beita_c=self.sigMoid(self.W_e(B_gang))      
        B_v=torch.mul(self.W_v(V),self.W_v_q(Q).unsqueeze(1)) 
        v_MAO=torch.mul(torch.mean(torch.mul(beita_s.unsqueeze(2),B_v),dim=1 ) ,beita_c)
        v_MAO_m=torch.zeros_like(K)
        v_MAO_m=v_MAO_m+v_MAO.unsqueeze(1)
        K_m=self.laynorm(self.RELU(self.W_k_m(torch.cat((v_MAO_m,K),dim=2),))+K)
        V_m=self.laynorm(self.RELU(self.W_v_m(torch.cat((v_MAO_m,V),dim=2),))+V)
        return v_MAO , K_m ,V_m   

class Attention(nn.Module):
    """
    注意网络
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: 图像编码的特征大小
        :param decoder_dim: RNN的解码大小
        :param attention_dim: 注意力网络的维度
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(decoder_dim, attention_dim) 
        self.decoder_att = nn.Linear(decoder_dim, attention_dim) 
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  
        att2 = self.decoder_att(decoder_hidden)  
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  
        alpha = self.softmax(att)  
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  
        return attention_weighted_encoding, alpha

# not working
class MLC(nn.Module):
    def __init__(self,
          classes=11,
          sementic_features_dim=1024,
          fc_in_features=2048,
          k=1):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = nn.Softmax()
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
        tags = self.softmax(self.classifier(avg_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags, semantic_features

class RelationMemory(nn.Module):
    def __init__(self, D):
        super(RelationMemory, self).__init__()
        self.W_Q = nn.Linear(D, D)
        self.W_K = nn.Linear(D, D)
        self.W_V = nn.Linear(D, D)
        self.softmax = nn.Softmax(dim=2) 
        self.layernorm = nn.LayerNorm(D, eps=1e-05, elementwise_affine=True)
        self.AOAttention = AOAttention(D)
        self.mlp = nn.Sequential(nn.Linear(1024, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 1024),
                                 nn.ReLU())
        self.W = nn.Linear(1024, 1024 * 2)
        self.U = nn.Linear(1024, 1024 * 2)
    def slice(self, Q, K, V): 
        d = Q.size(2) // 8
        list_Q = []
        list_k = []
        list_v = []
        for i in range(8):
            tempQ = Q[:, :, d * i:d * (i + 1)]
            tempK = K[:, :, d * i:d * (i + 1)]
            tempV = V[:, :, d * i:d * (i + 1)]
            list_Q.append(tempQ)
            list_k.append(tempK)
            list_v.append(tempV)
        return list_Q, list_k, list_v

    def forward(self, B, A):
        list_Q, list_k, list_v = self.slice(self.W_Q(B), self.W_K(A), self.W_V(A))
        head = []
        for i in range(len(list_v)):
            f_dot_att = self.softmax(list_Q[i].matmul(list_k[i].permute(0, 2, 1)) / 8)
            f_dot_att = f_dot_att.matmul(list_v[i])
            head.append(f_dot_att)
        f_mh_att = torch.cat((head[0], head[1], head[2], head[3], head[4], head[5], head[6], head[7]),
                             dim=2)  
        AOA = self.AOAttention(self.W_Q(B), f_mh_att, f_mh_att)
        Z = self.layernorm(AOA + A) 
        next_memory = B + Z
        next_memory = next_memory + self.mlp(next_memory)
        gates = self.W(B) + self.U(torch.tanh(A))
        gates = torch.split(gates, split_size_or_sections=1024, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * B
        next_memory = next_memory.squeeze(1)
        return next_memory

class DecoderWithAttention(nn.Module):
    """
    Report generator
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, word2idx, encoder_dim=2048, dropout=0.5,
                 max_seq_len=52):
        """
        embed_dim  decoder_dim  设置为1024
        :param attention_dim: 注意网络大小
        :param embed_dim: 嵌入大小
        :param decoder_dim: RNN解码器大小
        :param vocab_size: 词表大小
        :param encoder_dim: 编码图像的特征尺寸
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()       
        self.AOA=AOAttention(1024)
        self.multi_head=multi_head_att(1024)
        self.X_Linear=X_Linear_block()
        self.feat_embed=nn.Linear(encoder_dim,1024)
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout_num = dropout
        self.ctx_dim = 1024
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.W_G=nn.Linear(1024*2,1024)
        self.embedding = nn.Embedding(vocab_size, embed_dim)  
        self.dropout = nn.Dropout(p=self.dropout_num)
        self.decode_step = MogrifierLSTMCell(embed_dim + decoder_dim, decoder_dim, 4)  
        #self.decode_step = nn.LSTMCell(embed_dim + decoder_dim, decoder_dim)  
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.f_beta = nn.Linear(decoder_dim, decoder_dim) 
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  
        self.fc1 = nn.Linear(decoder_dim * 3, decoder_dim)
        self.init_weights()
        self.rm = RelationMemory(1024)
        self.input_embed = nn.Linear(encoder_dim, 1024)

    def init_weights(self):
        """
        均匀分布初始化
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        加载预训练好的词嵌入
        :param embeddings: 预训练好的词嵌入
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        是否微调嵌入层（只有在不使用预训练的嵌入层时使用）。
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        初始化h,c
        """
        mean_encoder_out = encoder_out.mean(dim=1)  
        h = self.init_h(mean_encoder_out)  
        c = self.init_c(mean_encoder_out)
        return h, c

    def _forward_step(self, it, t, encoder_out, h, c, embeddings, alphas):
        attention_weighted_encoding, alpha = self.attention(encoder_out, h)
        gate = self.sigmoid(self.f_beta(h))  
        attention_weighted_encoding = gate * attention_weighted_encoding
        h, c = self.decode_step(
            torch.cat([embeddings(it), attention_weighted_encoding], dim=1), (h, c)) 
        preds = self.fc(self.dropout(h))  
        alphas[:, t, :] = alpha
        logprobs = F.log_softmax(preds, dim=1)

        return logprobs, alphas, h, c

    def forward(self, all_feats, encoded_captions,caption_lengths ):  
        """
        Forward propagation.
        :param encoded_captions: 字幕编码,  (batch_size, max_caption_length)
        :param caption_lengths: 字幕长度,  (batch_size, 1)
        :return: scores for vocabulary, 已排序的字幕编码, 解码长度, weights, sort indices
        """
        if 1:
            batch_size = all_feats.size(0)
            encoder_dim = all_feats.size(-1)
            vocab_size = self.vocab_size

            all_feats = all_feats.view(batch_size, -1, encoder_dim)  
            CNN_feats= self.dropout(self.feat_embed(all_feats))       
            num_pixels = all_feats.size(1)  
            caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)  
            encoded_captions = encoded_captions[sort_ind]
            CNN_feats = CNN_feats[sort_ind]
            all_feats=all_feats[sort_ind]
            embeddings = self.embedding(encoded_captions)             
            h, c = self.init_hidden_state(all_feats)  
            h1, c1 = self.init_hidden_state(all_feats)   
            decode_lengths = (caption_lengths - 1).tolist()
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)  
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)  
            a_mean=torch.mean(CNN_feats,dim=1)
            ctx_=torch.zeros_like(a_mean)           
            memory = torch.zeros_like(a_mean)           
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                Input = torch.cat((a_mean[:batch_size_t]+ctx_[:batch_size_t], embeddings[:batch_size_t, t, :]), dim=1)                
                input_rm = torch.cat((memory[:batch_size_t], embeddings[:batch_size_t, t, :]), dim=1)
                input_rm = self.input_embed(input_rm) 
                memory = self.rm(memory[:batch_size_t].unsqueeze(1), input_rm.unsqueeze(1))     
                attention_weighted_encoding, alpha = self.attention(CNN_feats[:batch_size_t], h[:batch_size_t])
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  
                attention_weighted_encoding = gate * attention_weighted_encoding              
                h, c = self.decode_step(Input,(h[:batch_size_t], c[:batch_size_t]))  
                a_mao=self.multi_head(h.unsqueeze(1),CNN_feats[:batch_size_t])
                ctx_=self.AOA(h.unsqueeze(1),a_mao,a_mao).squeeze(1)               
                a_mao = (a_mao * alpha.unsqueeze(2)).sum(dim=1)            
                input2 = torch.cat((a_mao, h), dim=1)
                h1, c1 = self.decode_step(input2, (h1[:batch_size_t], c1[:batch_size_t]))
                ctx1 = h1
                ctx_ = torch.cat([ctx1, ctx_], dim=1)
                ctx_ = torch.cat([ctx_, memory], dim=1)
                ctx_ = self.fc1(ctx_)                  
                preds = self.fc(self.dropout(ctx_))  
                predictions[:batch_size_t, t, :] = preds
                alphas[:batch_size_t, t, :] = alpha
            return predictions, encoded_captions, decode_lengths, alphas, sort_ind

