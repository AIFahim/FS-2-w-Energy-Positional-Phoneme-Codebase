import os
from TTS.tts.configs.shared_configs import BaseDatasetConfig,BaseAudioConfig,CharactersConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs
from TTS.utils.manage import ModelManager

def main():
    # BaseDatasetConfig: defines name, formatter and path of the dataset.
    # output_path = "tts_train_dir"

    # dataset_config = BaseDatasetConfig(
    #     formatter="ljspeech", meta_file_train="/home/asif/Datasets/Dataset_Bangla/MaleVoice/Meta_fr_all_bangla_context_n_noncontext/metadata.csv", path="/home/asif/Datasets/Dataset_Bangla/MaleVoice/Dataset_Bangla_ori_10152/" # os.path.join(output_path, "LJSpeech-1.1/"
    # )
    
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="/home/asif/Datasets/Dataset_Bangla/Female_Voice/new_dataset_10k/metadata_10030.txt", path="/home/asif/Datasets/Dataset_Bangla/Female_Voice/new_dataset_10k/sanzidakter091@gmail.com/" # os.path.join(output_path, "LJSpeech-1.1/"
    )

    # GlowTTSConfig: all model related values for training, validating and testing.
    # previous_my_valid_lis= ['a','a_1', 'a_2', 'ã', 'ã_1', 'ã_2', 'b', 'b_1', 'b_2', 'bʰ', 'bʰ_1', 'bʰ_2', 'c', 'c_1', 'c_2', 'cʰ', 'cʰ_1', 'cʰ_2', 'd', 'd_1', 'd_2', 'dʰ', 'dʰ_1', 'dʰ_2', 'd̪', 'd̪_1', 'd̪_2', 'd̪ʰ', 'd̪ʰ_1', 'd̪ʰ_2', 'e', 'e_1', 'e_2', 'ẽ', 'ẽ_1', 'ẽ_2', 'g', 'g_1', 'g_2', 'gʰ', 'gʰ_1', 'gʰ_2', 'h', 'h_1', 'h_2', 'i', 'i_1', 'i_2', 'ĩ', 'ĩ_1', 'ĩ_2', 'i̯', 'i̯_2', 'k', 'k_1', 'k_2', 'kʰ', 'kʰ_1', 'kʰ_2', 'l', 'l_1', 'l_2', 'm', 'm_1', 'm_2', 'n', 'n_1', 'n_2', 'o', 'o_1', 'o_2', 'õ', 'õ_1', 'õ_2', 'o̯', 'o̯_1', 'o̯_2', 'p', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p_1', 'p_2', 'pʰ', 'pʰ_1', 'pʰ_2', 'r', 'r_1', 'r_2', 's', 's_1', 's_2', 't', 't_1', 't_2', 'tʰ', 'tʰ_1', 'tʰ_2', 't̪', 't̪_1', 't̪_2', 't̪ʰ', 't̪ʰ_1', 't̪ʰ_2', 'u', 'u_1', 'u_2', 'ũ', 'ũ_1', 'ũ_2', 'u̯', 'u̯_2', 'æ', 'æ_1', 'æ_2', 'æ̃', 'æ̃_2', 'ŋ', 'ŋ_2', 'ɔ', 'ɔ_1', 'ɔ_2', 'ɔ̃', 'ɔ̃_2', 'ɟ', 'ɟ_1', 'ɟ_2', 'ɟʰ', 'ɟʰ_1', 'ɟʰ_2', 'ɽ', 'ɽ_2', 'ɽʰ', 'ʃ', 'ʃ_1', 'ʃ_2', 'ʲ', 'ʲ_2', 'ʰ', 'ʷ', 'ɔ̃_1', 'ʲ_1', 'ɽʰ_1', '-']
     
    my_valid_lis = ['a', 'a_1', 'a_2', 'ã', 'ã_1', 'ã_2', 'b', 'b_1', 'b_2', 'bʰ', 'bʰ_1', 'bʰ_2', 'bʱ', 'bʱ_1', 'bʱ_2', 'c', 'c_1', 'c_2', 'cʰ', 'cʰ_1', 'cʰ_2', 'd', 'd_1', 'd_2', 'dʰ', 'dʰ_1', 'dʰ_2', 'dʱ', 'dʱ_1', 'dʱ_2', 'd̪', 'd̪_1', 'd̪_2', 'd̪ʰ', 'd̪ʰ_1', 'd̪ʰ_2', 'd̪ʱ', 'd̪ʱ_1', 'd̪ʱ_2', 'e', 'e_1', 'e_2', 'ẽ', 'ẽ_1', 'ẽ_2', 'e̯', 'e̯_1', 'e̯_2', 'g', 'g_1', 'g_2', 'gʰ', 'gʰ_1', 'gʰ_2', 'gʱ', 'gʱ_1', 'gʱ_2', 'h', 'h_1', 'h_2', 'i', 'i_1', 'i_2', 'ĩ', 'ĩ_1', 'ĩ_2', 'i̯', 'i̯_1', 'i̯_2', 'k', 'k_1', 'k_2', 'kʰ', 'kʰ_1', 'kʰ_2', 'l', 'l_1', 'l_2', 'm', 'm_1', 'm_2', 'n', 'n_1', 'n_2', 'o', 'o_1', 'o_2', 'õ', 'õ_1', 'õ_2', 'o̯', 'o̯_1', 'o̯_2', 'p', 'p_1', 'p_2', 'pʰ', 'pʰ_1', 'pʰ_2', 'r', 'r_1', 'r_2', 's', 's_1', 's_2', 't', 't_1', 't_2', 'tʰ', 'tʰ_1', 'tʰ_2', 't̪', 't̪_1', 't̪_2', 't̪ʰ', 't̪ʰ_1', 't̪ʰ_2', 'u', 'u_1', 'u_2', 'ũ', 'ũ_1', 'ũ_2', 'u̯', 'u̯_1', 'u̯_2', 'æ', 'æ_1', 'æ_2', 'æ̃', 'æ̃_1', 'æ̃_2', 'ŋ', 'ŋ_1', 'ŋ_2', 'ɔ', 'ɔ_1', 'ɔ_2', 'ɔ̃', 'ɔ̃_1', 'ɔ̃_2', 'ɟ', 'ɟ_1', 'ɟ_2', 'ɟʰ', 'ɟʰ_1', 'ɟʰ_2', 'ɽ', 'ɽ_1', 'ɽ_2', 'ɽʰ', 'ɽʰ_1', 'ɽʰ_2', 'ɽʱ', 'ɽʱ_1', 'ɽʱ_2', 'ʃ', 'ʃ_1', 'ʃ_2', 'ʲ', 'ʲ_1', 'ʲ_2', 'ʷ', 'ʷ_1', 'ʷ_2', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14']

    # print(str(my_valid_lis))

    # assert False
    # char_list = ['a', 'a_1', 'a_2', 'ã', 'ã_1', 'ã_2', 'b', 'b_1', 'b_2', 'bʰ', 'bʰ_1', 'bʰ_2', 'c', 'c_1', 'c_2', 'cʰ', 'cʰ_1', 'cʰ_2', 'd', 'd_1', 'd_2', 'dʰ', 'dʰ_1', 'dʰ_2', 'd̪', 'd̪_1', 'd̪_2', 'd̪ʰ', 'd̪ʰ_1', 'd̪ʰ_2', 'e', 'e_1', 'e_2', 'ẽ', 'ẽ_1', 'ẽ_2', 'g', 'g_1', 'g_2', 'gʰ', 'gʰ_1', 'gʰ_2', 'h', 'h_1', 'h_2', 'i', 'i_1', 'i_2', 'ĩ', 'ĩ_1', 'ĩ_2', 'i̯', 'i̯_2', 'k', 'k_1', 'k_2', 'kʰ', 'kʰ_1', 'kʰ_2', 'l', 'l_1', 'l_2', 'm', 'm_1', 'm_2', 'n', 'n_1', 'n_2', 'o', 'o_1', 'o_2', 'õ', 'õ_1', 'õ_2', 'o̯', 'o̯_1', 'o̯_2', 'p', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p_1', 'p_2', 'pʰ', 'pʰ_1', 'pʰ_2', 'r', 'r_1', 'r_2', 's', 's_1', 's_2', 't', 't_1', 't_2', 'tʰ', 'tʰ_1', 'tʰ_2', 't̪', 't̪_1', 't̪_2', 't̪ʰ', 't̪ʰ_1', 't̪ʰ_2', 'u', 'u_1', 'u_2', 'ũ', 'ũ_1', 'ũ_2', 'u̯', 'u̯_2', 'æ', 'æ_1', 'æ_2', 'æ̃', 'æ̃_2', 'ŋ', 'ŋ_2', 'ɔ', 'ɔ_1', 'ɔ_2', 'ɔ̃', 'ɔ̃_2', 'ɟ', 'ɟ_1', 'ɟ_2', 'ɟʰ', 'ɟʰ_1', 'ɟʰ_2', 'ɽ', 'ɽ_2', 'ɽʰ', 'ʃ', 'ʃ_1', 'ʃ_2', 'ʲ', 'ʲ_2', 'ʷ']

    # traverse in the string
    # char_sen = ""
    # for x in my_valid_lis:
    #     char_sen += x
    # # print(char_sen)
    characters_config = CharactersConfig(
        pad = '',#'<PAD>',
        eos = '',#'\n', #'<EOS>', #'।',
        bos = '',#'<BOS>',# None,
        blank = '',#'<BLNK>',
        phonemes = None,
        # characters =  "তট৫ভিঐঋখঊড়ইজমএেঘঙসীঢ়হঞ‘ঈকণ৬ঁৗশঢঠ\u200c১্২৮দৃঔগও—ছউংবৈঝাযফ\u200dচরষঅৌৎথড়৪ধ০ুূ৩আঃপয়’নলো",
        characters = my_valid_lis ,#char_sen,
        punctuations = '' # "-!,|.? ",
    )


    audio_config = BaseAudioConfig(
        sample_rate = 16000,
        resample =True
    )

    output_path = "/home/asif/coqui_fastspeech_tts/TTS/fs_new_for_female" #os.path.dirname(os.path.abspath(__file__))
    
    config = Fastspeech2Config(
        batch_size=200,
        eval_batch_size=128,
        num_loader_workers=16,
        num_eval_loader_workers=16,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=5000,
        text_cleaner="collapse_whitespace",
        use_phonemes=False,
        # use_aligner=True,
        # phoneme_language="bn",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        

        compute_f0=True,
        f0_cache_path=os.path.join(output_path, "f0_cache"),
        compute_energy=True,
        energy_cache_path=os.path.join(output_path, "energy_cache"),

        print_step=25,
        print_eval=False,
        mixed_precision=False,
        output_path=output_path,
        datasets=[dataset_config],
        save_step=1000,
        audio=audio_config,
        characters=characters_config,
        cudnn_benchmark=True,
        test_sentences = [
    #         "পিপলস ইন্স্যুরেন্স অব চায়না ছেষট্টি বছর আগে ব্যবসা চালু করে।",
    #         "সোনার বাংলা আমি তোমায় ভালবাসি।"
            # "হয়,হয়ে,ওয়া,হয়েছ,হয়েছে,দিয়ে,যায়,দায়,নিশ্চয়,আয়,ভয়,নয়,আয়াত,নিয়ে,হয়েছে,দিয়েছ,রয়ে,রয়েছ,রয়েছে।",
            # "দেয়,দেওয়া,বিষয়,হয়,হওয়া,সম্প্রদায়,সময়,হয়েছি,দিয়েছি,হয়,হয়েছিল,বিষয়ে,নয়,কিয়াম,ইয়া,দেয়া,দিয়েছে,আয়াতে,দয়া।",
            # "হওয়ার,হয়েছে,নিশ্চয়ই,রায়,কিয়ামত,উভয়,দিয়েছেন,দুনিয়া,ন্যায়,অবস্থায়,যায়,ফিরিয়ে,দিয়েছিল,ভয়ে,দ্বিতীয়,দায়ক,পায়।",
            # "গিয়ে,চেয়ে,হিদায়াত,দায়ে,নিয়েছ,রয়েছে,শয়তান,কিয়ামতে,সম্প্রদায়ে,সম্প্রদায়ের,নেয়,জয়,কিয়ামতের,স্থায়ী,যাওয়া,দয়ালু।",
            # "ইয়াহুদ,নয়,ব্যয়,ইয়াহুদী,নেওয়া,উভয়ে,যায়,হয়েছিল,প্রয়োজন।"
            "ʃ_1 ɔ n n o b o t̪ i_2 ɔ_1 r t̪ t̪ʰ o_2 cʰ_1 i ʲ a n ɔ b b o i̯_2 ʃ_1 o ŋ kʰ o k_2" #ষণ্নবতি অর্থ ছিয়ানব্বই সংখ্যক।|
        ],
    )



    # compute alignments
    if config.model_args.use_aligner == False:
        manager = ModelManager()
        model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")

        print(model_path)
        # assert False
        # TODO: make compute_attention python callable
        os.system(
            # f"python TTS/bin/compute_attention_masks.py --model_path {model_path} --config_path {config_path} --dataset ljspeech --dataset_metafile /home/asif/Datasets/Dataset_Bangla/MaleVoice/Meta_fr_all_bangla_context_n_noncontext/metadata.csv --data_path /home/asif/Datasets/Dataset_Bangla/MaleVoice/Dataset_Bangla_ori_10152/  --use_cuda true"
            f"python /home/asif/coqui_fastspeech_tts/TTS/compute_attention_masks.py --model_path {model_path} --config_path {config_path} --dataset ljspeech --dataset_metafile /home/asif/Datasets/Dataset_Bangla/MaleVoice/Meta_fr_all_bangla_context_n_noncontext/metadata.csv --data_path /home/asif/Datasets/Dataset_Bangla/MaleVoice/Dataset_Bangla_ori_10152/  --use_cuda true"
    )


    # assert False

    
    ap = AudioProcessor.init_from_config(config)
  
    tokenizer, config = TTSTokenizer.init_from_config(config)


    def formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
        """Normalizes the LJSpeech meta data file to TTS format
        https://keithito.com/LJ-Speech-Dataset/"""
        txt_file = meta_file
        items = []
        speaker_name = "ljspeech"
        with open(txt_file, "r", encoding="utf-8") as ttf:
            for line in ttf:
                cols = line.split("|")
                wav_file = os.path.join(root_path, "wav", cols[0] + ".wav")
                try:
                    text = cols[2]
                except:
                    print("not found")

                items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
        return items

    
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
        formatter=formatter,
    )


    
    model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)


    
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    trainer.fit()

if __name__ == "__main__":
    main()
