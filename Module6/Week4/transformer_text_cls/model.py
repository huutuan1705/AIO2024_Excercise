import torch
import torch.nn as nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, max_length, 
                 num_layers, num_heads, ff_dim, dropout=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = TransformerEncoder(src_vocab_size, embed_dim, max_length, num_layers, 
                                            num_heads, ff_dim, dropout=dropout, device=device)
        self.decoder = TransformerDecoder(tgt_vocab_size, embed_dim, max_length, num_layers, 
                                            num_heads, ff_dim, dropout=dropout, device=device)
        self.fc = nn.Linear(embed_dim, tgt_vocab_size)
        
    def generate_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        
        src_mask = torch.zeros(
            (src_seq_len, src_seq_len), device=self.device
        ).type(torch.bool)
        
        tgt_mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=self.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(
            tgt_mask==0, float('-inf')
        ).masked_fill(tgt_mask==1, float(0.0))
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        
        return output