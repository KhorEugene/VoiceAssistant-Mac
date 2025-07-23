import pyaudio
import asyncio
import websockets
from websockets.exceptions import ConnectionClosed
import torch
import sounddevice as sd
import numpy as np
import json

WEBSOCKET_URI = "<SERVER_URL_HERE>/communicate"

# ===============================
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 512

audio = pyaudio.PyAudio()

num_samples = 512
vad_threshold = 0.2
max_consec_silence = 3

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK
)

model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True
)


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound
# ===============================


async def receive_messages(websocket):
    try:
        while True:
            response = await websocket.recv()
            if isinstance(response, bytes):
                audio_data = np.frombuffer(response, dtype=np.int16)
                sd.play(audio_data, samplerate=24000)
                sd.wait()
            elif isinstance(response, str):
                try:
                    metadata_dict = json.loads(response)
                    metadata_type = metadata_dict.get("type", None)
                    if metadata_type:
                        if metadata_type == "metadata":
                            print(
                                f"STT: {metadata_dict.get('stt')} | "
                                f"LLM: {metadata_dict.get('llm')} | "
                                f"TTS: {metadata_dict.get('tts')}"
                            )
                        elif metadata_type == "prompt":
                            print(f"Prompt: {metadata_dict.get('content')}")
                        elif metadata_type == "response":
                            print(f"Response: {metadata_dict.get('content')}")
                except Exception:
                    print("Unexpected response received.")
    except ConnectionClosed:
        print("\nConnection to server closed.")


async def send_messages(websocket):
    chunk_buffer = []
    consec_silence = 0
    while True:
        # ============================================
        audio_chunk = await asyncio.to_thread(
            stream.read,
            num_frames=CHUNK,
            exception_on_overflow=False
        )

        audio_int16 = np.frombuffer(audio_chunk, np.int16)

        audio_float32 = int2float(audio_int16)

        voice_confidence = model(torch.from_numpy(audio_float32), 16000).item()
        if voice_confidence > vad_threshold:
            consec_silence = 0
            chunk_buffer.append(audio_chunk)
        else:
            if len(chunk_buffer) > 0:
                if consec_silence > max_consec_silence:
                    combined_chunk = b''.join(chunk_buffer)
                    await websocket.send(combined_chunk)
                    chunk_buffer = []
                    print("Voice chunk sent")
                else:
                    chunk_buffer.append(audio_chunk)
            consec_silence += 1
        # ============================================


async def client_app():
    try:
        async with websockets.connect(WEBSOCKET_URI) as websocket:
            print("Connected to server.")
            receiver_task = asyncio.create_task(receive_messages(websocket))
            sender_task = asyncio.create_task(send_messages(websocket))

            _, pending = await asyncio.wait(
                [receiver_task, sender_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
    except (ConnectionRefusedError, ConnectionClosed):
        print("Connection failed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(client_app())
    except KeyboardInterrupt:
        print("\nClient stopped by user.")
