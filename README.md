# Voice Assistant for Mac
This repository includes code to host a full, local speech-to-speech pipeline on a Mac Mini with Apple Silicon, but you have to bring-your-own-LLM backend.

🔥 **Low latency**: Less than *4 seconds* from the end of your speech input to the end of the final word generation

💻 **Local**: All models can be hosted locally on consumer grade hardware

🟰 **Parallel streaming**: Workflow is optimized to run models in parallel

🫙 **Storage efficiency**: No more than 10GB storage space required for all the models

🔌 **Plug'n'Play**: Easily modify the server code to integrate your own LLM endpoints


## Design
<img width="996" height="473" alt="image" src="https://github.com/user-attachments/assets/b8724e6f-10f0-4fa8-8e52-4994bdff2b3c" />

There are 4 components to the fully functioning pipeline
- **Voice Activity Detection (VAD)**: Detects any speech activity. This lightweight model runs on the client device, and uses the Silero-VAD model ([Github](https://github.com/snakers4/silero-vad))
- **Speech to Text (STT)**: This uses whisper v3 large turbo, optimized for Apple silicon using MLX. The Whisper MLX Lightning library was cloned and updated to enable this ([Github](https://github.com/mustafaaljadery/lightning-whisper-mlx)) ([Huggingface](https://huggingface.co/mlx-community/whisper-large-v3-turbo))
- **Large Language Model (LLM)**: Generates a natural language response to the user prompt. Gemma3n 4B was used for its lightweight and conversationally natural properties. *Note: This implementation was not included in this repository*
- **Text to Speech (TTS)**: Generates speech from the text. The lightweight 82M parameter Kokoro engine was used ([Huggingface](https://huggingface.co/hexgrad/Kokoro-82M))

## Performance
For a generic prompt and conversational exchange, the end-to-end latency takes less than 3 seconds
- Q: Hello there, how are you?
- A: I am doing well, thanks for asking. How about you?
- STT: 2.1 seconds | LLM: 1.5 seconds | TTS: 0.2 seconds

<img width="538" height="322" alt="image" src="https://github.com/user-attachments/assets/229ca33e-0554-4cd7-8561-5ef72f2fac22" />

System specifications: Mac Mini (M4), 32GB RAM, 256GB SSD

## Future Roadmap
- Incorporate voice recognition
- Parallelise STT engine

## Get Started
### Server Setup
- Requirement: Mac device with Apple Silicon
- Clone this repository
```
git clone https://github.com/KhorEugene/VoiceAssistant-Mac.git
```
- Install dependencies
```
pip install -r requirements.txt
brew install espeak-ng
```
- Run server
```
gunicorn -k uvicorn.workers.UvicornWorker -w 1 server:app --bind=0.0.0.0  --timeout 60
```

### Client Setup
- Install dependencies
```
pip install -r client_requirements.txt
```
- Update the Websocket URI in the client.py to your server IP and port
- Run client
```
python client.py
```
- Speak to your client microphone. The server should echo what you've just said
- To integrate your LLM into the pipeline, add your streaming code into the server.py accordingly
