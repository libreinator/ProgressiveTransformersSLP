#!/usr/bin/env python3

import re
import sys
import os
import dill as pickle
import torch
from model import build_model
from helpers import (
    load_config,
    load_checkpoint,
    get_latest_checkpoint,
)
from prediction import validate_on_data
from training import TrainManager
from torchtext import data
from torchtext.data import Dataset
from constants import UNK_TOKEN, PAD_TOKEN, TARGET_PAD
from vocabulary import build_vocab, Vocabulary
from data import SignProdDataset


# shortened version of load_data
def load_data(
    cfg: dict,
) -> (Dataset, Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    data_cfg = cfg["data"]
    # Source, Target and Files postfixes
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    files_lang = data_cfg.get("files", "files")
    # Train, Dev and Test Path
    train_path = data_cfg["train"]
    test_path = data_cfg["test"]

    if os.path.isfile(test_path + ".pth"):
        os.unlink(test_path + ".pth")

    level = "word"
    lowercase = False
    max_sent_length = data_cfg["max_sent_length"]
    # Target size is plus one due to the counter required for the model
    trg_size = cfg["model"]["trg_size"] + 1
    # Skip frames is used to skip a set proportion of target frames, to simplify the model requirements
    skip_frames = data_cfg.get("skip_frames", 1)

    EOS_TOKEN = "</s>"
    tok_fun = lambda s: list(s) if level == "char" else s.split()

    # Source field is a tokenised version of the source words
    src_field = data.Field(
        init_token=None,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        tokenize=tok_fun,
        batch_first=True,
        lower=lowercase,
        unk_token=UNK_TOKEN,
        include_lengths=True,
    )

    # Files field is just a raw text field
    files_field = data.RawField()

    def tokenize_features(features):
        features = torch.as_tensor(features)
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]

    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    # Creating a regression target field
    # Pad token is a vector of output size, containing the constant TARGET_PAD
    reg_trg_field = data.Field(
        sequential=True,
        use_vocab=False,
        dtype=torch.float32,
        batch_first=True,
        include_lengths=False,
        pad_token=torch.ones((trg_size,)) * TARGET_PAD,
        preprocessing=tokenize_features,
        postprocessing=stack_features,
    )
    # Create the Training Data, using the SignProdDataset
    train_data = SignProdDataset(
        path=train_path,
        exts=("." + src_lang, "." + trg_lang, "." + files_lang),
        fields=(src_field, reg_trg_field, files_field),
        trg_size=trg_size,
        skip_frames=skip_frames,
        filter_pred=lambda x: len(vars(x)["src"]) <= max_sent_length
        and len(vars(x)["trg"]) <= max_sent_length,
    )

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)

    if os.path.isfile("src_vocab.pth"):
        print("loading binary src_vocab")
        with open("src_vocab.pth", "rb") as f:
            src_vocab = pickle.load(f)
            print("done")
    else:
        print("build_vocab")
        src_vocab = build_vocab(
            field="src",
            min_freq=src_min_freq,
            max_size=src_max_size,
            dataset=train_data,
            vocab_file=src_vocab_file,
        )

        print("saving src_vocab")
        with open("src_vocab.pth", "wb") as f:
            pickle.dump(src_vocab, f)
        print("done")

    # Create a target vocab just as big as the required target vector size -
    # So that len(trg_vocab) is # of joints + 1 (for the counter)
    trg_vocab = [None] * trg_size

    # Create the Testing Data
    test_data = SignProdDataset(
        path=test_path,
        exts=("." + src_lang, "." + trg_lang, "." + files_lang),
        trg_size=trg_size,
        fields=(src_field, reg_trg_field, files_field),
        skip_frames=skip_frames,
    )

    src_field.vocab = src_vocab

    return test_data, src_vocab, trg_vocab


# pylint: disable-msg=logging-too-many-args
def test(cfg_file, input_text: str, ckpt: str = None) -> None:

    print("inside test()")
    text = re.sub(r"[^\w\s]", "", input_text)
    text = text.lower()

    if text[0] == " ":
        text = text[1:]

    if text[-1] == " ":
        text = text[:-1]

    if text[-1] != ".":
        text += " ."

    with open("../data_aud_text/test.text", "w") as f:
        f.write(text)
    with open("../data_aud_text/test.file", "w") as f:
        f.write("inconnect_input")
    # Load the config file
    cfg = load_config(cfg_file)

    # Load the model directory and checkpoint
    model_dir = cfg["training"]["model_dir"]
    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        print("get_latest_checkpoint")
        ckpt = get_latest_checkpoint(model_dir, post_fix="_best")
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence")
    )
    use_cuda = cfg["training"].get("use_cuda", False)
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # load the data
    print("load_data")
    test_data, src_vocab, trg_vocab = load_data(cfg=cfg)

    # To produce testing results
    data_to_predict = {"test": test_data}
    # To produce validation results
    # data_to_predict = {"dev": dev_data}

    # Load model state from disk
    print("load_checkpoint")
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # Build model and load parameters into it
    print("build_model")
    model = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])
    # If cuda, set model as cuda
    if use_cuda:
        model.cuda()

    # print("save binary")
    # torch.save(model, "model.binary")
    # print("saved model")

    # Set up trainer to produce videos
    print("TrainManager")
    trainer = TrainManager(model=model, config=cfg, test=True)

    # For each of the required data, produce results
    for data_set_name, data_set in data_to_predict.items():
        print("data_set_name", data_set_name)
        print("data_set", data_set)

        # Validate for this data set
        score, loss, references, hypotheses, inputs, all_dtw_scores, file_paths = (
            validate_on_data(
                model=trainer.model,
                data=data_set,
                batch_size=batch_size,
                max_output_length=max_output_length,
                eval_metric=eval_metric,
                loss_function=None,
                batch_type=batch_type,
                type="val",
            )
        )

        # Set which sequences to produce video for
        display = list(range(len(hypotheses)))

        # Produce videos for the produced hypotheses
        trainer.produce_validation_video(
            output_joints=hypotheses,
            inputs=inputs,
            references=references,
            model_dir=model_dir,
            display=display,
            type="test",
            file_paths=file_paths,
            text=input_text,
        )


if __name__ == "__main__":
    test(cfg_file="Configs/Base.yaml", input_text=sys.argv[1])
