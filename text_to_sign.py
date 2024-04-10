#!/usr/bin/env python3

import argparse, torch
from model import Model, build_model
from helpers import (
    load_config,
    load_checkpoint,
    get_latest_checkpoint,
)
from prediction import validate_on_data
from data import load_data
from training import TrainManager


# pylint: disable-msg=logging-too-many-args
def test(cfg_file, input_text: str, ckpt: str = None) -> None:
    print("inside test()")

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
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)

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

    print("save binary")
    torch.save(model, "model.binary")
    print("saved model")

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

        print("score", score)
        print("loss", loss)
        print("references", references)
        print("hypotheses", hypotheses)
        print("inputs", inputs)
        print("all_dtw_scores", all_dtw_scores)
        print("file_paths", file_paths)

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
        )


if __name__ == "__main__":
    test(cfg_file="Configs/Base.yaml", input_text="All the world is a stage")
