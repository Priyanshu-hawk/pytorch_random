import torch
from torch import nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import random_split, DataLoader, Dataset
import torchmetrics

from dataset import BiLingualDataSet, casual_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import shutil

def greedy_decoder(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        #mask tgt decoder
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calc output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, val_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writter, num_examples=2):
    model.eval()
    count=0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.inference_mode():
        for batch in val_ds:
            count+=1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decoder(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            #print logs (This is of tqdm not a normal print because it interfear with tqdm progress bar)
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                break
    if writter:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writter.add_scalar('validation cer', cer, global_step)
        writter.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writter.add_scalar('validation wer', wer, global_step)
        writter.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writter.add_scalar('validation BLEU', bleu, global_step)
        writter.flush()

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
    ds_raw = load_dataset(config["HF_DS_Name"], f'{config["src_lang"]}-{config["tgt_lang"]}', split="train[:10%]") # taking 10% of ds, bcz for hindi opus it quite large

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

def is_colab():
    try:
        import google.colab
        return True
    except:
        return False

def train_model(config):
    #Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device - {device}")

    Path("{}_{}".format(config["HF_DS_Name"], config["model_folder"])).mkdir(parents=True, exist_ok=True)

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
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1

        optimzer.load_state_dict(state["optimizer_state_dict"])
        gloabl_step = state["global_step"]

        print("Resume Training from: {} epoch".format(gloabl_step))
    else:
        print("Starting a fresh Training")
    
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
        
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], 
                           device, lambda msg: batch_itirator.write(msg), gloabl_step, writter)
        
        model_filename = get_weights_file_path(config, f'{e:02d}')
        torch.save({
            "epoch": e,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimzer.state_dict(),
            "global_step": gloabl_step 
        }, model_filename)

        if is_colab():
            shutil.copy(model_filename, "/content/drive/MyDrive/Trasnformer_W/")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)