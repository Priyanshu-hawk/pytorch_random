from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "src_lang": "en",
        "tgt_lang": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "02",
        "tokenizer_file": "tokenizer_{0}.json",
        "epx_name": "runs/tmodel",
        "HF_DS_Name": "Helsinki-NLP/opus-100"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = "{}_{}".format(config["HF_DS_Name"], config["model_folder"])
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path(".") / model_folder / model_filename)