import argparse
import importlib
import os
from argparse import RawTextHelpFormatter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from TTS.tts.datasets import TTSDataset

from TTS.config import load_config
# from TTS.tts.datasets.TTSDataset import TTSDataset
from TTS.tts.models import setup_model
# from TTS.tts.utils.text.characters import make_symbols, phonemes, symbols
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_checkpoint

if __name__ == "__main__":


    print("I'm here")

    # assert False
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Extract attention masks from trained Tacotron/Tacotron2 models.
These masks can be used for different purposes including training a TTS model with a Duration Predictor.\n\n"""
        """Each attention mask is written to the same path as the input wav file with ".npy" file extension.
(e.g. path/bla.wav (wav file) --> path/bla.npy (attention mask))\n"""
        """
Example run:
    CUDA_VISIBLE_DEVICE="0" python TTS/bin/compute_attention_masks.py
        --model_path /data/rw/home/Models/ljspeech-dcattn-December-14-2020_11+10AM-9d0e8c7/checkpoint_200000.pth
        --config_path /data/rw/home/Models/ljspeech-dcattn-December-14-2020_11+10AM-9d0e8c7/config.json
        --dataset_metafile metadata.csv
        --data_path /root/LJSpeech-1.1/
        --batch_size 32
        --dataset ljspeech
        --use_cuda True
""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to Tacotron/Tacotron2 model file ")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to Tacotron/Tacotron2 config file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        required=True,
        help="Target dataset processor name from TTS.tts.dataset.preprocess.",
    )

    parser.add_argument(
        "--dataset_metafile",
        type=str,
        default="",
        required=True,
        help="Dataset metafile inclusing file paths with transcripts.",
    )
    parser.add_argument("--data_path", type=str, default="", help="Defines the data path. It overwrites config.json.")
    parser.add_argument("--use_cuda", type=bool, default=False, help="enable/disable cuda.")

    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size for the model. Use batch_size=1 if you have no CUDA."
    )
    args = parser.parse_args()

    C = load_config(args.config_path)
    ap = AudioProcessor(**C.audio)

    # if the vocabulary was passed, replace the default
    # if "characters" in C.keys():
    #     symbols, phonemes = make_symbols(**C.characters)

    symbols = ['a', 'a_1', 'a_2', 'ã', 'ã_1', 'ã_2', 'b', 'b_1', 'b_2', 'bʰ', 'bʰ_1', 'bʰ_2', 'c', 'c_1', 'c_2', 'cʰ', 'cʰ_1', 'cʰ_2', 'd', 'd_1', 'd_2', 'dʰ', 'dʰ_1', 'dʰ_2', 'd̪', 'd̪_1', 'd̪_2', 'd̪ʰ', 'd̪ʰ_1', 'd̪ʰ_2', 'e', 'e_1', 'e_2', 'ẽ', 'ẽ_1', 'ẽ_2', 'g', 'g_1', 'g_2', 'gʰ', 'gʰ_1', 'gʰ_2', 'h', 'h_1', 'h_2', 'i', 'i_1', 'i_2', 'ĩ', 'ĩ_1', 'ĩ_2', 'i̯', 'i̯_2', 'k', 'k_1', 'k_2', 'kʰ', 'kʰ_1', 'kʰ_2', 'l', 'l_1', 'l_2', 'm', 'm_1', 'm_2', 'n', 'n_1', 'n_2', 'o', 'o_1', 'o_2', 'õ', 'õ_1', 'õ_2', 'o̯', 'o̯_1', 'o̯_2', 'p', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p_1', 'p_2', 'pʰ', 'pʰ_1', 'pʰ_2', 'r', 'r_1', 'r_2', 's', 's_1', 's_2', 't', 't_1', 't_2', 'tʰ', 'tʰ_1', 'tʰ_2', 't̪', 't̪_1', 't̪_2', 't̪ʰ', 't̪ʰ_1', 't̪ʰ_2', 'u', 'u_1', 'u_2', 'ũ', 'ũ_1', 'ũ_2', 'u̯', 'u̯_2', 'æ', 'æ_1', 'æ_2', 'æ̃', 'æ̃_2', 'ŋ', 'ŋ_2', 'ɔ', 'ɔ_1', 'ɔ_2', 'ɔ̃', 'ɔ̃_2', 'ɟ', 'ɟ_1', 'ɟ_2', 'ɟʰ', 'ɟʰ_1', 'ɟʰ_2', 'ɽ', 'ɽ_2', 'ɽʰ', 'ʃ', 'ʃ_1', 'ʃ_2', 'ʲ', 'ʲ_2', 'ʰ', 'ʷ', 'ɔ̃_1', 'ʲ_1', 'ɽʰ_1', '-'] #self._characters


    # load the model
    # num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    num_chars = len(symbols)
    # TODO: handle multi-speaker
    model = setup_model(C)
    model, _ = load_checkpoint(model, args.model_path, args.use_cuda, True)

    # data loader
    preprocessor = importlib.import_module("TTS.tts.datasets.formatters")
    preprocessor = getattr(preprocessor, args.dataset)
    meta_data = preprocessor(args.data_path, args.dataset_metafile)
    dataset = TTSDataset(
        model.decoder.r,
        C.text_cleaner,
        compute_linear_spec=False,
        ap=ap,
        meta_data=meta_data,
        characters=C.characters if "characters" in C.keys() else None,
        add_blank=C["add_blank"] if "add_blank" in C.keys() else False,
        use_phonemes=C.use_phonemes,
        phoneme_cache_path=C.phoneme_cache_path,
        phoneme_language=C.phoneme_language,
        enable_eos_bos=C.enable_eos_bos_chars,
    )

    dataset.sort_and_filter_items(C.get("sort_by_audio_len", default=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        drop_last=False,
    )

    # compute attentions
    file_paths = []
    with torch.no_grad():
        for data in tqdm(loader):
            # setup input data
            text_input = data[0]
            text_lengths = data[1]
            linear_input = data[3]
            mel_input = data[4]
            mel_lengths = data[5]
            stop_targets = data[6]
            item_idxs = data[7]

            # dispatch data to GPU
            if args.use_cuda:
                text_input = text_input.cuda()
                text_lengths = text_lengths.cuda()
                mel_input = mel_input.cuda()
                mel_lengths = mel_lengths.cuda()

            model_outputs = model.forward(text_input, text_lengths, mel_input)

            alignments = model_outputs["alignments"].detach()
            for idx, alignment in enumerate(alignments):
                item_idx = item_idxs[idx]
                # interpolate if r > 1
                alignment = (
                    torch.nn.functional.interpolate(
                        alignment.transpose(0, 1).unsqueeze(0),
                        size=None,
                        scale_factor=model.decoder.r,
                        mode="nearest",
                        align_corners=None,
                        recompute_scale_factor=None,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
                # remove paddings
                alignment = alignment[: mel_lengths[idx], : text_lengths[idx]].cpu().numpy()
                # set file paths
                wav_file_name = os.path.basename(item_idx)
                align_file_name = os.path.splitext(wav_file_name)[0] + "_attn.npy"
                file_path = item_idx.replace(wav_file_name, align_file_name)
                # save output
                wav_file_abs_path = os.path.abspath(item_idx)
                file_abs_path = os.path.abspath(file_path)
                file_paths.append([wav_file_abs_path, file_abs_path])
                np.save(file_path, alignment)

        # ourput metafile
        metafile = os.path.join(args.data_path, "metadata_attn_mask.txt")

        with open(metafile, "w", encoding="utf-8") as f:
            for p in file_paths:
                f.write(f"{p[0]}|{p[1]}\n")
        print(f" >> Metafile created: {metafile}")
