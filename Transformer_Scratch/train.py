import torch
from torch import nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import random_split, DataLoader, Dataset

from dataset import BiLingualDataSet, casual_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings


def get_all_sentence(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentence(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("Helsinki-NLP/opus-100", f'{config["src_lang"]}-{config["tgt_lang"]}', split="train")

    # build tokenizers
    tokenizer_src = get_tokenizer(config, ds_raw, config["src_lang"])
    tokenizer_tgt = get_tokenizer(config, ds_raw, config["tgt_lang"])

    # keeping 90% for traning and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BiLingualDataSet(train_ds_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"])
    val_ds = BiLingualDataSet(val_ds_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["src_lang"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["tgt_lang"]]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print("Max len of source sentence: ", max_len_src)
    print("Max len of Target sentence: ", max_len_tgt)

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"])
    return model

def train_model(config):
    #Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device - {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # tensorboard session
    writter = SummaryWriter(config["epx_name"])

    optimzer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    gloabl_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print("Preloading Model: ", model_filename)
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1

        optimzer.load_state_dict(state["optimizer_state_dict"])
        gloabl_step = state["gloabl_step"]
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for e in range(initial_epoch, config["epochs"]):
        model.train()
        batch_itirator = tqdm(train_dataloader, desc=f"Processing Epoch {e:02d}")
        for batch in batch_itirator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            # run the tensor through transformer
            enocder_output = model.encode(encoder_input, encoder_mask) # batchm seq_len, d_model
            decoder_output = model.decode(enocder_output, encoder_mask, decoder_input, decoder_mask) #batch, seq_len , d_model
            proj_out = model.project(decoder_output) # b, seq_len, tgt_voc_size
            
            lable = batch["lable"].to(device)

            loss = loss_fn(proj_out.view(-1, tokenizer_tgt.get_vocab_size()), lable.view(-1))
            batch_itirator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writter.add_scalar("train_loss", loss.item(), gloabl_step)
            writter.flush()

            #back prop
            loss.backward()

            #update weight
            optimzer.step()
            optimzer.zero_grad()

            gloabl_step+=1
        
        model_filename = get_weights_file_path(config, f'{e:02d}')
        torch.save({
            "epoch": e,
            "model_sate_dict": model.state_dict(),
            "optimizer-state-dict": optimzer.state_dict(),
            "global_step": gloabl_step 
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)