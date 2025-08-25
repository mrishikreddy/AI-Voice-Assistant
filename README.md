# Voice AI Assistant

A simple voice-based AI assistant that listens to your questions, transcribes them, and provides answers using AI models. It can handle questions related to a specific person (Rishik Reddy) by retrieving information from a provided PDF file, or answer general questions using a generative AI model. The assistant uses speech-to-text (STT), retrieval-augmented generation (RAG), and text-to-speech (TTS) for a hands-free experience.

---

## Description

This project is a Python script that turns your computer into a voice-activated AI assistant. It records audio input until it detects silence, transcribes the speech using OpenAI's **Whisper** model, and then processes the question:

* If the question is about **"Rishik Reddy,"** it uses a RAG setup to pull relevant context from `data.pdf` and generates an answer.
* For other questions, it uses a general AI model to respond.

Answers are spoken back using available TTS tools on your system (like `say` on macOS, `espeak` on Linux, or PowerShell on Windows) or printed to the console as a fallback.

The script is designed to run in a loop, listening for new questions until interrupted.

---

## Features

* **Voice Recording**: Detects speech and stops on silence or after a maximum duration.
* **Speech-to-Text**: Uses Whisper to transcribe audio.
* **Question Routing**: Checks if the question is person-specific and routes to RAG or general response accordingly.
* **RAG Integration**: Loads and queries a PDF file (`data.pdf`) for context-aware answers.
* **Text-to-Speech**: Speaks responses using system TTS tools.
* **Configurable**: Adjustable parameters like silence timeout, max utterance length, and VAD aggressiveness.

---

## Requirements

* Python 3.12 or compatible.
* **Libraries (install via `pip`):**
    * `numpy`
    * `sounddevice`
    * `soundfile`
    * `webrtcvad`
    * `whisper` (from OpenAI)
    * `langchain`
    * `langchain-community`
    * `langchain-google-genai`
    * `chromadb` (for vector store)
    * `pypdf` (for PDF loading)
    * `tiktoken` (for text splitting)
    * `librosa` (optional, for audio resampling if needed)

> **Note:** Some libraries like `pygame`, `chess`, etc., are mentioned in the environment but are not used in this script.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Create a `requirements.txt` file with the libraries listed above if not already present.)

3.  **Provide a Google API Key:**
    Set `os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"` in the script or via an environment variable. The script uses Google Generative AI for embeddings and the LLM.

4.  **Add your PDF file:**
    Place `data.pdf` in the root directory. This file contains context for person-specific questions.

---

## Usage

1.  **Run the script:**
    ```bash
    python main.py
    ```
2.  The script will initialize models and start listening.
3.  Speak your question clearly.
4.  It will transcribe, process, and respond (via TTS or console).
5.  Press `Ctrl+C` to exit.

### Example flow:

* **Say:** "Tell me about Rishik Reddy."
    * _Routes to RAG using `data.pdf`._
* **Say:** "What's the weather like?"
    * _Uses general AI response._

---

## Configuration

* **API Keys**: Replace `"API_KEY"` with your actual Google API key.
* **Whisper Model**: Defaults to `"small"`. Change `WHISPER_MODEL` for other sizes (e.g., `"base"`, `"medium"`).
* **Audio Settings**: Adjust `SILENCE_TIMEOUT`, `MAX_UTTERANCE_SECONDS`, `VAD_AGGRESSIVENESS` as needed.
* **PDF File**: The script loads `data.pdf`. Update the path in `build_chains()` if using a different file.

---

## Limitations

* Requires a quiet environment for accurate voice activity detection.
* TTS depends on system tools; falls back to console print if none are available.
* No internet access for package installation in the code environment—install dependencies manually.
* Hardcoded to treat the user as "BOSS" in responses.
