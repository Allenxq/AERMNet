import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
import torch.nn.functional as F
from tqdm import tqdm

# MAMNet_RM (i.e. AERMNet) 
data_name = 'AERMNet'

# Load the trained model
checkpoint = '/content/drive/MyDrive/AERMNet/checkpoint/AERMNet.pth.tar'

word_map_file = '/content/drive/MyDrive/AERMNet/data/WORDMAP.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Load model
checkpoint = torch.load(checkpoint)
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
# not working
mlc = checkpoint['mlc']
mlc = mlc.to(device)
mlc.eval()

decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r', encoding='utf-8') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])

def evaluate(loadpath):
    """
    Evaluation
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_name, 'TEST', loadpath),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    references = list()
    hypotheses = list()

    for i, (all_feats) in enumerate(
        tqdm(loader, desc = "EVALUATING AT BEAM SIZE " + str(1))):
        all_feats = all_feats.to(device)
        all_feats = encoder(all_feats)  
        b = all_feats.size(0)
        e = all_feats.size(3)
        mean_feats = torch.mean(all_feats.view(b, -1, e), dim=1)
        mean_feats = mean_feats.to(device)
        pre_tag, semantic_features = mlc(mean_feats)
        semantic_features = semantic_features.to(device)
        k = 1
        # Encoder
        encoder_dim = all_feats.size(3)
        encoder_out = all_feats.view(1, -1, encoder_dim)  
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)        
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['Start']]] * k).to(device)  
        seqs = k_prev_words  
        top_k_scores = torch.zeros(k, 1).to(device) 
        complete_seqs = list()
        complete_seqs_scores = list()
        all_feats = all_feats.view(1, -1, encoder_dim)
        CNN_feats = decoder.feat_embed(all_feats)  
        Q = torch.mean(CNN_feats, dim=1)  
        step = 1
        h, c = decoder.init_hidden_state(all_feats)
        h = h.expand(k,h.size(1))
        c = c.expand(k,h.size(1))
        h1, c1 = decoder.init_hidden_state(all_feats)
        h1 = h1.expand(k,h.size(1))
        c1 = c1.expand(k,h.size(1))        
        ctx_ = torch.zeros_like(h) 
        ctx_ = ctx_.expand(k,  1024)       
        ctx1 = torch.zeros_like(h1)
        ctx1 = ctx1.expand(k, 1024)
        a_mean = torch.mean(CNN_feats,dim=1)
        a_mean = a_mean.expand(k,  1024)
        memory = torch.zeros_like(h1)
        memory = memory.expand(k, 1024)       
        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  
            Input = torch.cat((a_mean + ctx_, embeddings), dim=1)
            input_rm = torch.cat((memory, embeddings), dim=1)
            input_rm = decoder.input_embed(input_rm)         
            memory = decoder.rm(memory.unsqueeze(1), input_rm.unsqueeze(1))      
            attention_weighted_encoding, alpha = decoder.attention(CNN_feats, h)
            h, c = decoder.decode_step(Input, (h, c))  
            a_mao = decoder.multi_head(h.unsqueeze(1), CNN_feats)
            ctx_ = decoder.AOA(h.unsqueeze(1), a_mao, a_mao).squeeze(1)           
            a_mao = (a_mao * alpha.unsqueeze(2)).sum(dim=1)  
            input2 = torch.cat((a_mao, h), dim=1)    
            h1, c1 = decoder.decode_step(input2, (h1, c1))
            ctx1 = h1            
            ctx_ = torch.cat([ctx1, ctx_], dim=1)            
            ctx_ = torch.cat([ctx_, memory], dim=1)           
            ctx_ = decoder.fc1(ctx_)             
            scores = decoder.fc(ctx_)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores  
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True) 
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  
   
            prev_word_inds = top_k_words // vocab_size  
            next_word_inds = top_k_words % vocab_size  
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['End']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  
            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]           
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]           
            semantic_features = semantic_features[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            ctx_ = ctx_[prev_word_inds[incomplete_inds]]
            a_mean = a_mean[prev_word_inds[incomplete_inds]]
            memory = memory[prev_word_inds[incomplete_inds]] 
            if step > 200:
                break
            step += 1
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]      
        pre_caps = [rev_word_map[w] for w in seq if w not in {word_map['Start'], word_map['End'], word_map['Pad']}]
        pre_caps_num=[w for w in seq if w not in {word_map['Start'], word_map['End'], word_map['Pad']}]
        str_res = ''
        seq_res=''
        res = ""
        for i in range(len(pre_caps)):
          res += pre_caps[i]
          res += ' '
          str_res += pre_caps[i] + ' '
          seq_res += str(pre_caps_num[i])+' '
        a = loadpath.split("/")
        with open('/content/drive/MyDrive/AERMNet/report/report.txt', 'a+', encoding = 'utf-8') as f:
          f.write(str(a[-1]) + '\t' + res.strip(' ') + '\n')
        return res 

if __name__ == '__main__':
    beam_size = 1
    # Load image path
    image_path = '/content/drive/MyDrive/AERMNet/test_picture/CXR640_IM-2219-1001.png'
    print(evaluate(image_path))
