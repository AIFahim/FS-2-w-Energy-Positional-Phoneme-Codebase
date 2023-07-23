import torchaudio
from TTS.utils.synthesizer import Synthesizer
import requests
import json
import base64
import wave
import numpy as np



def infer(grapheme:str, wav_file:str):
    

    synth=Synthesizer(tts_checkpoint="/home/asif/tts_all/coqui_tts/my_exp/tts_infer_models_positional_phoneme_16khz_w_ckpt/infer_tts/align_male_infer/coqui_allign_tts_new_male/TTS/checkpoints_bn_male/bn_male_align_tts_previous_10k_dataset-February-20-2023_07+13AM-6e3f74fc/checkpoint_330000.pth",
    tts_config_path="/home/asif/tts_all/coqui_tts/my_exp/tts_infer_models_positional_phoneme_16khz_w_ckpt/infer_tts/align_male_infer/coqui_allign_tts_new_male/TTS/checkpoints_bn_male/bn_male_align_tts_previous_10k_dataset-February-20-2023_07+13AM-6e3f74fc/config.json")

    wav=synth.tts(grapheme)
    synth.save_wav(wav, wav_file)
    

if __name__ == "__main__":
    #infer("আমি বাংলায় গান গাই।", "/home/elias/testaudio/test.wav")
    #infer("প্রতিটি ভোর মানেই নতুন এক নিদর্শন-পাখিদের কোলাহলে মুখরিত চারপাশ। ", "/home/elias/testaudio/test-03.wav")
    # infer("বড়গাঁওয়া মাঠে পরিমনির নৃত্য", "/home/asif/tts_all/coqui_tts/aushik_bhai_backups/inference_audio_test/test_w_vocoder_multibandmelgan_44k.wav")
    # infer("বড়গাঁওয়া মাঠে পরিমনির নৃত্য","/home/asif/tts_all/coqui_tts/aushik_bhai_backups/inference_audio_test/test_vits_44k_weight_123000.wav")

    infer("মাওবাদীদের মুক্তির একদিনের মধ্যেই মার্কেটিং বিভাগের প্রোডাক্ট অফিসার চট্টোপাধ্যায় স্বস্তিতে নেই রবিবার সকালে বিষয়টি নিয়ে পুলিশ সুপার শ্যামল জানান জমিসংক্রান্ত বিষয় নিয়ে এই হামলা হয়েছে গুলির লড়াইয়ে খতম এক বাইক আরোহী সোসালিস্ট গাধা সন্ত্রাসীর চোট লেগেছে সকালেই বাবা  গিয়ে বলেন নিজের শখ হয় উনি আর কোথাও ছিপ নিয়ে বসে মাছ ধরবেন স্বস্তিতে অথবা  বস্তির পাশের অদূরে লেকে সকালেই বাবা  গিয়ে বলেন নিজের শখ হয় উনি বস্তির পাশের অদূরে লেকে  ছিপ নিয়ে বসে মাছ ধরবেন  কৃষকরা খুব বিরক্ত কারন তাদের ক্ষেতে আগুন লাগিয়ে দিয়েছে সন্ত্রাসীরা ","/home/asif/tts_all/coqui_tts/aushik_bhai_backups/inference_w_alignTTS_positional_phoneme_for_prince_jj.wav")