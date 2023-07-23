import torchaudio
from TTS.utils.synthesizer import Synthesizer
import requests
import json
import base64
import wave
import numpy as np
import os
import io
from scipy.io.wavfile import write

# from fastapi import FastAPI 


from fastapi import FastAPI, File, UploadFile, Request
import json
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# def list_to_base64_audio(samples, sample_rate=16000):
#     # Create a wave file with the samples
#     with wave.open("audio.wav", "w") as f:
#         f.setsampwidth(2)
#         f.setnchannels(1)
#         f.setframerate(sample_rate)
#         f.writeframes(samples)
    
#     # Open the wave file and read its data
#     with open("audio.wav", "rb") as f:
#         audio_data = f.read()
    
#     # Encode the audio data as a base64 string
#     base64_audio = base64.b64encode(audio_data).decode("utf-8")
    
#     return base64_audio


# @app.post('/get_denoised_segments_from_path/')
# async def denoised_webrtcvad(body:Body):

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


url = "https://dev.revesoft.com:6790/phonemizer"
synth=Synthesizer("/home/auishik/infer_tts/coqui_fs2_infer/TTS/mode_weights_n_config/fs2_male/checkpoint_250000.pth","/home/auishik/infer_tts/coqui_fs2_infer/TTS/mode_weights_n_config/config.json", use_cuda = False)
    
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Body(BaseModel):
    text:str


@app.get('/')
async def home():
    return "Fast server is working!"


@app.post("/infer_fs2_male/")
async def infer(body:Body):
    
    text = body.dict()['text']

    payload = json.dumps({
    "text": text
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)
    phonme = response.json()['output']

    phonme = " ".join(phonme)

    wav = synth.tts(phonme)
    
    synth.save_wav(wav, "/home/auishik/infer_tts/coqui_fs2_infer/TTS/out.wav")

    with open("/home/auishik/infer_tts/coqui_fs2_infer/TTS/out.wav", "rb") as f:
        audio_data = f.read()

    base64_audio = base64.b64encode(audio_data).decode("utf-8") 
    
    response = {'output' : base64_audio}
    # print(response)
    return response 

    
if __name__ == "__main__":
    
    
    uvicorn.run("infer_fs2_male:app", host='0.0.0.0', port = 8005, reload= True, 
                ssl_certfile="/etc/letsencrypt/live/dev.revesoft.com/cert.pem",
                ssl_keyfile="/etc/letsencrypt/live/dev.revesoft.com/privkey.pem")
