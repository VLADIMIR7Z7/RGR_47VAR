"""Локальный запуск сохранённого классификатора."""
from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy import signal, fft
import torch
from torch import nn
import gradio as gr
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"models/best_model.pth"
torch.set_num_threads(2)
SR, LENGTH = 16000, 32000

def waveform(path):
    x,sr=sf.read(path,always_2d=True,dtype="float32")
    if len(x)<sr/4:
        raise ValueError("Нужно не менее 0,25 с звука.")
    x=x.mean(axis=1)
    if not np.isfinite(x).all() or np.max(np.abs(x))<1e-6:
        raise ValueError("Запись пуста или содержит тишину.")
    if sr!=SR:
        d=math.gcd(sr,SR)
        x=signal.resample_poly(x,SR//d,sr//d)
    start=max(0,(len(x)-LENGTH)//2)
    x=x[start:start+LENGTH]
    left=(LENGTH-len(x))//2
    x=np.pad(x,(left,LENGTH-len(x)-left))
    peak=np.max(np.abs(x))
    if peak<1e-6:
        raise ValueError("Центральный фрагмент содержит тишину.")
    return (x/peak).astype("float32")

hz=np.linspace(0,SR/2,257)

edges=700*(10**(np.linspace(0,2595*np.log10(1+8000/700),42)/2595)-1)

filters=np.array([np.maximum(0,np.minimum((hz-a)/(b-a),(c-hz)/(c-b))) for a,b,c in zip(edges,edges[1:],edges[2:])],dtype="float32")

def features(x):
    frames=np.lib.stride_tricks.sliding_window_view(x,400)[::160]
    spec=np.abs(fft.rfft(frames*np.hanning(400).astype("float32"),n=512))**2
    return np.log(np.maximum(filters@spec.T,1e-8)).astype("float32")

class Net(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1,8,3,stride=(1,2),padding=1),nn.BatchNorm2d(8),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,1)),nn.Flatten(),nn.Linear(64,24),nn.ReLU(),
            nn.Dropout(.25),nn.Linear(24,2))

def load_model():
    ck=torch.load(MODEL,map_location="cpu",weights_only=True)
    net=Net()
    net.load_state_dict(ck["weights"])
    return net.eval(),ck

net,ck=load_model()

def predict(path):
    if not path or Path(path).suffix.lower() not in {".flac",".wav",".ogg"}:
        raise ValueError("Загрузите WAV, FLAC или OGG.")
    try:
        x=waveform(path)
    except (RuntimeError,OSError) as err:
        raise ValueError("Не удалось прочитать аудиофайл.") from err
    z=torch.from_numpy((features(x)-ck["mu"])/ck["sd"])[None,None]
    with torch.inference_mode():
        prob=net(z).softmax(1)[0].tolist()
    return dict(zip(["Женский голос (F)","Мужской голос (M)"],prob))

def analyze(path):
    try:
        probs=predict(path)
        return max(probs,key=probs.get),probs
    except ValueError as err:
        return str(err),None

def build_gui():
    with gr.Blocks(title="Анализ записи · РГР 47") as gui:
        gr.Markdown("# Анализ речевой записи\nСравнение с метками M/F корпуса LibriSpeech. Используются центральные 2 секунды речи.")
        uploaded=gr.File(label="Аудиозапись · WAV, FLAC или OGG",type="filepath")
        action=gr.Button("Анализировать",variant="primary")
        result=gr.Textbox(label="Результат",interactive=False)
        probability=gr.Label(label="Вероятности модели",num_top_classes=2)
        action.click(analyze,uploaded,[result,probability])
    return gui

if __name__ == "__main__":
    build_gui().launch(server_name="127.0.0.1",share=False)
