#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super(Transformer, self).__init__()
        
        for k, v in kwargs.items():
            print(f" * {k}={v}")
        
        self.vocab_size = kwargs.get('vocab_size')
        self.model_dim = kwargs.get('model_dim')
        self.dropout = kwargs.get('dropout')
        self.n_encoder_layers = kwargs.get('n_encoder_layers')
        self.n_decoder_layers = kwargs.get('n_decoder_layers')
        self.n_heads = kwargs.get('n_heads')
        self.batch_size = kwargs.get('batch_size')
        self.PAD_IDX = kwargs.get('pad_idx', 0)

        self.encoder = Encoder(
            self.vocab_size, self.model_dim, self.dropout, self.n_encoder_layers, self.n_heads)
        self.decoder = Decoder(
            self.vocab_size, self.model_dim, self.dropout, self.n_decoder_layers, self.n_heads)
        self.fc = nn.Linear(self.model_dim, self.vocab_size)
        

    @staticmethod    
    def generate_square_subsequent_mask(size: int):
            """Generate a triangular (size, size) mask. From PyTorch docs."""
            mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask


    def encode(
            self, 
            x: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Input
            x: (B, S) with elements in (0, C) where C is num_classes
        Output
            (B, S, E) embedding
        """

        mask = (x == self.PAD_IDX).float()
        encoder_padding_mask = mask.masked_fill(mask == 1, float('-inf'))
        
        # (B, S, E)
        encoder_output = self.encoder(
            x, 
            padding_mask=encoder_padding_mask
        )  
        
        return encoder_output, encoder_padding_mask
    
    
    def decode(
            self, 
            tgt: torch.Tensor, 
            memory: torch.Tensor, 
            memory_padding_mask=None
        ) -> torch.Tensor:
        """
        B = Batch size
        S = Source sequence length
        L = Target sequence length
        E = Model dimension
        
        Input
            encoded_x: (B, S, E)
            y: (B, L) with elements in (0, C) where C is num_classes
        Output
            (B, L, C) logits
        """
        
        mask = (tgt == self.PAD_IDX).float()
        tgt_padding_mask = mask.masked_fill(mask == 1, float('-inf'))

        decoder_output = self.decoder(
            tgt=tgt, 
            memory=memory, 
            tgt_mask=self.generate_square_subsequent_mask(tgt.size(1)), 
            tgt_padding_mask=tgt_padding_mask, 
            memory_padding_mask=memory_padding_mask,
        )  
        output = self.fc(decoder_output)  # shape (B, L, C)
        return output

        
        
    def forward(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
        ) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (B, L, C) logits
        """
        
        # Encoder output shape (B, S, E)
        encoder_output, encoder_padding_mask = self.encode(x)  

        # Decoder output shape (B, L, C)
        decoder_output = self.decode(
            tgt=y, 
            memory=encoder_output, 
            memory_padding_mask=encoder_padding_mask
        )  
        
        return decoder_output

    def predict(
            self,
            x: torch.Tensor,
            sos_idx: int=1,
            eos_idx: int=2,
            max_length: int=None
        ) -> torch.Tensor:
        """
        Method to use at inference time. Predict y from x one token at a time. This method is greedy
        decoding. Beam search can be used instead for a potential accuracy boost.

        Input
            x: str
        Output
            (B, L, C) logits
        """

        # Pad the tokens with beginning and end of sentence tokens
        x = torch.cat([
            torch.tensor([sos_idx]), 
            x, 
            torch.tensor([eos_idx])]
        ).unsqueeze(0)

        encoder_output, mask = self.transformer.encode(x) # (B, S, E)
        
        if not max_length:
            max_length = x.size(1)

        outputs = torch.ones((x.size()[0], max_length)).type_as(x).long() * sos_idx
        for step in range(1, max_length):
            y = outputs[:, :step]
            probs = self.transformer.decode(y, encoder_output)
            output = torch.argmax(probs, dim=-1)
            
            # Uncomment if you want to see step by step predicitons
            # print(f"Knowing {y} we output {output[:, -1]}")

            if output[:, -1].detach().numpy() in (eos_idx, sos_idx):
                break
            outputs[:, step] = output[:, -1]
            
        
        return outputs