import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
# PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# Hugging Face transformers
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import AutoTokenizer
from tqdm import tqdm

# Group assignment Python modules
from chatbot_common import *
from data_preparation import load_source_target_pair_dataset

# Pretrained models are available here: https://drive.google.com/drive/u/0/folders/1-vnMaiuIMTjABXTmIDhfHW8yGE_3e8Mh


# Learning and evaluation code that runs but was abandoned in favour of Hugging Face
# Heavily inspired on https://www.kaggle.com/code/moaazreda/chatbot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device {device.type}")
tokeniser = AutoTokenizer.from_pretrained("bert-base-uncased")
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
early_stop_count = DEFAULT_EARLY_STOP_COUNT
max_seq_len = 50


@dataclass
class ChatbotConfig:
    encoder_vocab_size: int
    decoder_vocab_size: int
    d_embed: int
    h: int
    d_ff: int
    N_encoder: int
    N_decoder: int
    dropout: float
    max_seq_len: int


# Encoder class
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_embed))
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.N_encoder)])
        self.drop = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.drop(x + x_pos)
        for layer in self.encoder_blocks:
            x = layer(x, mask)
       
        return self.norm(x)


# Decoder class
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.decoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_embed))
        self.drop = nn.Dropout(config.dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.N_decoder)])
        self.norm = nn.LayerNorm(config.d_embed)
        self.linear = nn.Linear(config.d_embed, config.decoder_vocab_size)

    def future_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len), diagonal=1) != 0).to(device)
        mask = mask.long()
        return mask.view(1,1,seq_len,seq_len)

    def forward(self, memory, src_mask, trg, trg_pad_mask):
        seq_len = trg.size(1)
        causal_mask = self.future_mask(seq_len).to(trg.device)
       
        trg_mask = torch.logical_or(trg_pad_mask, self.future_mask(seq_len))
               
        x = self.tok_embed(trg) + self.pos_embed[:, :trg.size(1), :]
        x = self.drop(x)
       
        for layer in self.decoder_blocks:
            x = layer(memory, src_mask, x, trg_mask)
            
        x = self.norm(x)
        logits = self.linear(x)
        
        return logits


class ResidualConnection(nn.Module):
    def __init__(self, d_embed, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        normed_x = self.layer_norm(x)
        sublayer_output = sublayer(normed_x)      
      
        return x + self.dropout(sublayer_output)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_embed, dropout=0.1):
        super().__init__()
        assert d_embed % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.num_heads = num_heads
        self.d_k = d_embed // num_heads  # Dimension of each head
        self.d_embed = d_embed

        # Linear layers for query, key, and value
        self.q_linear = nn.Linear(d_embed, d_embed)
        self.k_linear = nn.Linear(d_embed, d_embed)
        self.v_linear = nn.Linear(d_embed, d_embed)
        self.out = nn.Linear(d_embed, d_embed)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Perform linear projections
        query = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)             
        key = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)       
        value = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Ensure mask has the correct shape for multi-head attention
        if mask is not None:
            scores = scores.masked_fill(mask , float('-inf'))

        attention_filter = nn.functional.softmax(scores,dim=-1)
        attention_filter = self.attn_dropout(attention_filter)

        x = torch.matmul(attention_filter,value)

        # Concatenate heads and apply output linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_embed)
        
        return self.out(x)


# EncoderBlock class
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.atten = MultiHeadAttention(config.h, config.d_embed, config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual1 = ResidualConnection(config.d_embed, config.dropout)
        self.residual2 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.atten(x, x, x, mask = mask))
        x = self.residual2(x, lambda x: self.feed_forward(x))
        return x


# DecoderBlock class
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.atten1 = MultiHeadAttention(config.h, config.d_embed, config.dropout)
        self.atten2 = MultiHeadAttention(config.h, config.d_embed, config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residuals = nn.ModuleList([ResidualConnection(config.d_embed, config.dropout) for _ in range(3)])

    def forward(self, memory, src_mask, decoder_layer_input, trg_mask):
        x = memory
        y = decoder_layer_input
        y = self.residuals[0](y, lambda y: self.atten1(y, y, y, mask=trg_mask))
        y = self.residuals[1](y, lambda y: self.atten2(y, x, x, mask=src_mask))
        y = self.residuals[2](y, self.feed_forward)
        return y


# Transformer class
class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_mask, trg, trg_pad_mask):
        memory = self.encoder(src, src_mask)
        output = self.decoder(memory, src_mask, trg, trg_pad_mask)
        return output


def make_model(config):
    model = Transformer(
        encoder=Encoder(config),
        decoder=Decoder(config)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(device)


def make_batch_input(x, y):
    # Move inputs to the specified device
    x = x.clamp(0, 30000-1)  # Clamp source inputs
    y = y.clamp(0, 30000-1)
    src = x.to(device)
    trg_in = y[:, :-1].to(device)  # Target input excludes the last token
    trg_out = y[:, 1:].to(device)  # Target output excludes the first token

    # Create source padding mask
    src_pad_mask = (src == PAD).unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, seq_len]
    
    # Create target padding mask
    trg_pad_mask = (trg_in == PAD).unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, seq_len]

    return src, trg_in, trg_out, src_pad_mask, trg_pad_mask


def train_epoch(model, dataloaders, optimiser, scheduler):
    model.train()
    losses = []
    num_batches = len(dataloaders.train_loader)
    pbar = tqdm(enumerate(dataloaders.train_loader), total=num_batches)

    for idx, (x, y) in pbar:
        # Reset gradients
        optimiser.zero_grad()

        # Prepare inputs and masks
        src, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch_input(x, y)

        # Forward pass
        pred = model(src, src_pad_mask, trg_in, trg_pad_mask)
        pred = pred.view(-1, pred.size(-1))

        loss = loss_fn(pred, trg_out.reshape(-1))
        
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimiser.step()
        # Update the learning rate
        scheduler.step() 

        # Progress bar update every 50 steps
        if idx > 0 and idx % 50 == 0:
            pbar.set_description(
                f"train loss = {loss.item():.3f}, lr = {scheduler.get_last_lr()[0]:.6f}"
            )

    return np.mean(losses)


def validate(model,dataloaders):
    model.eval()
    losses=[]
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloaders.valid_loader):
            src, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch_input(x, y)
    
            pred = model(src, src_pad_mask, trg_in, trg_pad_mask)
            pred = pred.view(-1, pred.size(-1))
    
            loss = loss_fn(pred, trg_out.reshape(-1))
            
            losses.append(loss.item())

    return torch.tensor(losses).mean().item()


def train(model, dataloaders, epochs, optimiser, scheduler, warmup_steps):
    global early_stop_count
    best_valid_loss = float('inf')
    train_size = len(dataloaders.train_loader) * 128

    for ep in range(epochs):
        # Training phase
        train_loss = train_epoch(model, dataloaders, optimiser, scheduler)

        # Validation phase
        valid_loss = validate(model, dataloaders)
        print(f"Epoch {ep}: train loss = {train_loss:.5f}, valid loss = {valid_loss:.5f}")

        # Check for improvement in validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        else:
            # Early stopping mechanism
            if scheduler.last_epoch > 2 * warmup_steps:
                early_stop_count -= 1
                if early_stop_count <= 0:
                    return train_loss, valid_loss

    return train_loss, valid_loss


def translate(model, x):
    with torch.no_grad():
        batch_size = x.size(0)  # Get the batch size (50 in this case)
        seq_len = x.size(1)  # Get the input sequence length (50 in this case)
        
        # Initialize the output sequence with BOS token (e.g., 50) and pad the rest
        y = torch.full((batch_size, max_seq_len), fill_value=PAD, dtype=torch.long).to(device)
        y[:, 0] = 1  # Set the first token of each sequence in the batch to BOS

        # Create padding mask for the input sequence
        x_pad_mask = (x == PAD).unsqueeze(1).unsqueeze(2).to(device)

        # Pass input sequence through the encoder
        memory = model.encoder(x, x_pad_mask)
        
        # Generate the output sequence
        for t in range(1, max_seq_len):
            # Create padding mask for the output sequence up to the current timestep
            y_pad_mask = (y == PAD).unsqueeze(1).unsqueeze(2).to(device)
            
            # Decode using the decoder
            logits = model.decoder(memory, x_pad_mask, y[:, :t], y_pad_mask[:, :, :, :t])
            
            # Get the last token's predicted output
            last_output = logits[:, -1, :].argmax(-1)
            
            # Append the predicted token to the output sequence
            y[:, t] = last_output

    return y


def decode_sentence(tokenizer, sentence_ids):
    # Convert to list if sentence_ids is a tensor
    if not isinstance(sentence_ids, list):
        sentence_ids = sentence_ids.tolist()

   
    # sentence_ids = remove_pad(sentence_ids)

    # Decode and clean the sentence
    decoded_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
    return decoded_sentence.strip().replace(" .", ".")


def answer_question(model, question: str):
    # Tokenize the input text and add special tokens
    src_tokens = tokeniser.encode(
           question, max_length=50, truncation=True, padding="max_length", add_special_tokens=True)
    # input_tokens = [BOS] + tokenizers[SRC](text) + [EOS]
    input_tensor = torch.tensor(src_tokens, dtype=torch.long).to(device)
    input_tensor = input_tensor.unsqueeze(0).repeat(50, 1)
    # Translate the input using the model
    output = translate(model, input_tensor)
    
    # Decode the output tokens to get the translated sentence
    return decode_sentence(tokeniser, output[0])


class DialogueDataset(Dataset):
    source_target_pairs: pd.DataFrame
    #tokeniser
    max_len: int

    def __init__(self, source_target_pairs: pd.DataFrame, tokeniser: PreTrainedTokenizerBase, max_len):
        self.source_target_pairs = source_target_pairs
        self.tokeniser = tokeniser
        self.max_len = max_len

    def __len__(self):
        return len(self.source_target_pairs)

    def __getitem__(self, index: int):
        row = self.source_target_pairs.iloc[index]
        source = row["source"]
        target = row["target"]
        #print(f"__getitem__({index}): '{source=}', '{target=}'")

        # Tokenize and encode with padding/truncation
        source_tokens = self.tokeniser.encode(source, max_length=self.max_len, truncation=True, padding="max_length", add_special_tokens=True)
        target_tokens = self.tokeniser.encode(target, max_length=self.max_len, truncation=True, padding="max_length", add_special_tokens=True)

        # Convert to tensors and return
        return torch.tensor(source_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)


class DialogueDataLoaders:
    def __init__(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, tokeniser, batch_size: int, max_len: int = DEFAULT_MAX_LEN):
        self.train_dataset = DialogueDataset(train_data, tokeniser, max_len)
        self.valid_dataset = DialogueDataset(valid_data, tokeniser, max_len)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        src_seqs, trg_seqs = zip(*batch)
        
        return torch.stack(src_seqs), torch.stack(trg_seqs)


def main():
    os.makedirs(PYTORCH_MODELS_FOLDER, exist_ok=True)
    dataset_size = DatasetSize.SMALL
    model_file = os.path.join(PYTORCH_MODELS_FOLDER, f"chatbot_pytorch_{dataset_size}.model")
    if os.path.exists(model_file):
        print(f"Loading from '{model_file}'...")
        torch.serialization.add_safe_globals([Transformer, Encoder, nn.Embedding, nn.ModuleList, EncoderBlock,
                                              DecoderBlock, MultiHeadAttention, nn.Linear, nn.Dropout, nn.Sequential,
                                              nn.ReLU, ResidualConnection, nn.LayerNorm, Encoder, Decoder])
        model = torch.load(model_file)
    else:
        source_target_pairs = load_source_target_pair_dataset(dataset_size)
        print(f"{len(source_target_pairs)=}")
        print(source_target_pairs.head())
        
        #for id in ["31/1.tsv/0", "31/1.tsv/2", "278/1.tsv/0"]:
        #    row = source_target_pairs.loc[source_target_pairs["id"] == id].iloc[0]
        #    print()
        #    print(f"{id}: Question: {row.source}")
        #    print(f"{id}: Response: {row.target}")

        config = ChatbotConfig(
                encoder_vocab_size=30000,
                decoder_vocab_size=30000,
                d_embed=512,
                h=8,
                d_ff=2048,
                N_encoder=6,
                N_decoder=6,
                dropout=0.1,
                max_seq_len=50)

        train_data = source_target_pairs.reset_index(drop=True)
        val_data = source_target_pairs[:30000].reset_index(drop=True)
        dataloaders = DialogueDataLoaders(train_data, val_data, tokeniser, DEFAULT_BATCH_SIZE, MAX_LENGTH)
        print(len(dataloaders.train_loader))
        print(len(dataloaders.valid_loader))

        # Initialize model, optimiser, scheduler, and loss function
        model = make_model(config)
        warmup_steps = len(dataloaders.train_loader)*128
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, 
            lr_lambda = lambda step: config.d_embed ** -0.5 * min((step + 1) ** -0.5, (step + 1) * 3 ** -1.5))

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        history = train(model, dataloaders, NUM_EPOCHS, optimiser, scheduler, warmup_steps) 
        torch.save(model, model_file)
        print(history)

    for question in TEST_QUESTIONS:
        print(f"Question: {question} ... Response: {answer_question(model, question)}")


if __name__ == "__main__":
    main()
