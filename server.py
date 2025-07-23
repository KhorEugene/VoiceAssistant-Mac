import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from lightning_whisper_mlx import LightningWhisperMLX
import numpy as np
from kokoro import KPipeline
import json
import time


# =========================
whisper = LightningWhisperMLX(
    model="whisper-large-v3-turbo",
    batch_size=6,
    quant=None
)
pipeline = KPipeline(lang_code='a')


async def tts_pipeline(text: str):
    tts_start_i = time.time()
    generator = pipeline(text, voice='af_heart')
    audio_bytes = None
    for _, (_, _, audio) in enumerate(generator):
        audio_numpy = audio.cpu().numpy()
        audio_scaled = (audio_numpy * 32767).astype(np.int16)
        audio_bytes = audio_scaled.tobytes()
    tts_time_i = time.time() - tts_start_i
    return audio_bytes, tts_time_i
# =========================


async def receiver(websocket: WebSocket):
    async for data in websocket.iter_bytes():
        try:
            # Speech to text pipeline
            stt_start = time.time()
            audio_int16 = np.frombuffer(
                data,
                np.int16
            )
            prompt = whisper.transcribe(audio_int16)['text']
            stt_time = time.time() - stt_start

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "prompt",
                        "content": prompt,
                    }
                )
            )

            # Response generation pipeline
            llm_start = time.time()
            # TODO - Insert your own LLM integration here
            response = prompt
            llm_time = time.time() - llm_start

            # Text to speech pipeline
            tts_time = 0
            (
                audio_bytes,
                tts_time_part
            ) = await tts_pipeline(
                response
            )
            tts_time += tts_time_part
            if audio_bytes:
                await websocket.send_bytes(
                    audio_bytes
                )

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "metadata",
                        "stt": stt_time,
                        "llm": llm_time,
                        "tts": tts_time,
                    }
                )
            )

        except Exception as e:
            print(e)


app = FastAPI()


@app.websocket("/communicate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")

    receiver_task = asyncio.create_task(
        receiver(websocket)
    )

    try:
        await asyncio.wait([receiver_task])
    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        receiver_task.cancel()
        print("Connection closed and tasks cancelled.")
