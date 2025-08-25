#!/usr/bin/env python3
import os
import queue
import time
import wave
import sys
import subprocess
import shutil
import platform
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

import webrtcvad
import whisper

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30  
VAD_AGGRESSIVENESS = 2
SILENCE_TIMEOUT = 1.2  
MAX_UTTERANCE_SECONDS = 30
TEMP_AUDIO_FILE = "_tmp_question.wav"
os.environ["GOOGLE_API_KEY"] = "API_KEY"


WHISPER_MODEL = "small"


GOOGLE_API_KEY = "API_KEY"
if not GOOGLE_API_KEY:
    print("Please Provide API Key")


def int16_from_float32(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16)

def write_wav(filename: str, data: np.ndarray, samplerate: int = SAMPLE_RATE):
    data_i16 = int16_from_float32(data)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(data_i16.tobytes())

def play_wav(filename: str):
    data, sr = sf.read(filename, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    sd.play(data, sr)
    sd.wait()

class Recorder:
    def __init__(self, samplerate=SAMPLE_RATE, channels=CHANNELS, frame_duration_ms=FRAME_DURATION_MS, vad_mode=VAD_AGGRESSIVENESS):
        self.samplerate = samplerate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(samplerate * (frame_duration_ms / 1000.0))
        self.vad = webrtcvad.Vad(vad_mode)
        self.q = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Stream status: {status}", file=sys.stderr)
        self.q.put(indata.copy())

    def record_utterance(self, silence_timeout=SILENCE_TIMEOUT, max_seconds=MAX_UTTERANCE_SECONDS):
        print("Listening... speak now (will stop after a short silence).")
        frames = []
        started = False
        silent_duration = 0.0
        start_time = time.time()

        with sd.InputStream(samplerate=self.samplerate, channels=self.channels, callback=self._audio_callback, blocksize=self.frame_size):
            while True:
                try:
                    data = self.q.get(timeout=1.0)
                except queue.Empty:
                    continue

                if data.ndim > 1:
                    data_mono = data.mean(axis=1)
                else:
                    data_mono = data[:, 0] if data.ndim > 1 else data

                pcm16 = int16_from_float32(data_mono)
                is_speech = False
                try:
                    is_speech = self.vad.is_speech(pcm16.tobytes(), sample_rate=self.samplerate)
                except Exception:
                    energy = np.mean(np.abs(data_mono))
                    is_speech = energy > 0.01

                frames.append(data_mono)

                if is_speech:
                    started = True
                    silent_duration = 0.0
                else:
                    if started:
                        silent_duration += (self.frame_duration_ms / 1000.0)

                if started and silent_duration >= silence_timeout:
                    break

                if time.time() - start_time > max_seconds:
                    break

        audio = np.concatenate(frames, axis=0).astype(np.float32)
        maxval = np.max(np.abs(audio)) + 1e-9
        audio = audio / maxval
        write_wav(TEMP_AUDIO_FILE, audio, samplerate=self.samplerate)
        return TEMP_AUDIO_FILE

class STT:
    def __init__(self, model_name=WHISPER_MODEL):
        print(f"Loading Whisper model '{model_name}' (may take time)...")
        self.model = whisper.load_model(model_name)

    def transcribe(self, wav_path: str) -> str:
        audio_data, samplerate = sf.read(wav_path, dtype='float32')

        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        if samplerate != SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=SAMPLE_RATE)

        res = self.model.transcribe(audio_data, fp16=False)
        return res.get("text", "").strip()


class CLI_TTS:
    def __init__(self):
        self.platform = platform.system().lower()
        self.has_say = shutil.which("say") is not None
        self.has_espeak = shutil.which("espeak") is not None
        self.has_spd = shutil.which("spd-say") is not None

    def say(self, text: str):
        text = text.strip()
        if not text:
            return

        try:
            if self.has_say:
                subprocess.run(["say", text])
                return
            if self.has_espeak:
                subprocess.run(["espeak", text])
                return
            if self.has_spd:
                subprocess.run(["spd-say", text])
                return
            if self.platform.startswith("windows"):
                safe_text = text.replace("'", "''")
                ps_cmd = f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('{safe_text}');"
                subprocess.run(["powershell", "-Command", ps_cmd], check=False)
                return
        except Exception as e:
            print(f"TTS subprocess failed: {e}", file=sys.stderr)

        print("\n--- TTS fallback (no native TTS found). Showing text: ---")
        print(text)
        print("--- end ---\n")

def build_chains():
    loader = PyPDFLoader("data.pdf")
    pdf_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(pdf_docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    check_template = """OUTPUT Responses: ["1", "0"]
Analyse the question and return only 1 or 0.
If the question is related to a person (His name is Rishik Reddy), return 1.
If not related to any person and can be answered from the internet or is generalized, return 0.

Question: {question}
"""
    check_prompt = ChatPromptTemplate.from_template(check_template)
    check_chain = (
        {"question": RunnablePassthrough()}
        | check_prompt
        | llm
        | StrOutputParser()
    )

    context_template = """Answer the question based ONLY on the following context.
Treat the user like your BOSS.

Context: {context}

Question: {question}
"""
    prompt1 = ChatPromptTemplate.from_template(context_template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt1
        | llm
        | StrOutputParser()
    )

    general_template = """Answer the question. Treat the user like your BOSS. Call him boss when ever necessary

Question: {question}
"""
    prompt2 = ChatPromptTemplate.from_template(general_template)
    general_chain = (
        {"question": RunnablePassthrough()}
        | prompt2
        | llm
        | StrOutputParser()
    )

    return check_chain, rag_chain, general_chain

def main():
    print("Initializing models and chains (this may take a while)...")
    check_chain, rag_chain, general_chain = build_chains()

    recorder = Recorder()
    stt = STT(model_name=WHISPER_MODEL)
    tts = CLI_TTS()

    print("Ready. Press Ctrl+C to quit.\n")

    try:
        while True:
            wav = recorder.record_utterance()
            print("Transcribing...")
            question = stt.transcribe(wav)
            print(f"You said: {question!r}")
            if not question.strip():
                print("No speech detected; listening again.\n")
                continue

            flow = check_chain.invoke(question)
            if flow.strip() == "1":
                answer = rag_chain.invoke(question)
            else:
                answer = general_chain.invoke(question)

            print("\nAnswer:\n", answer, "\n")
            tts.say(answer)
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()