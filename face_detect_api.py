import cv2
import os
import warnings
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi import HTTPException
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from scipy.io import wavfile
from pydub import AudioSegment
import random

app = FastAPI()

@app.get("/")
async def read_root():
    return "Server is live !!!"

@app.post("/detect_face/")
async def face_detect(input_image: UploadFile = File(...)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    try:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error decoding image: " + str(e))
    
    if image is None:
        raise HTTPException(status_code=400, detail="Image file could not be loaded")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    num_faces = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    resize_img = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)

    if num_faces == 1:
        return {"Result": True}
    else:
        return {"Result": False}

@app.post("/detect_audio/")
async def speech_detect(audio_file: UploadFile = File(...)):
    audio_file_path = f"./{audio_file.filename}"
    # audio_file_name = audio_file_path.split(".")
    # converted_audio_file = f"./{audio_file_name[0]}.wav"
    if audio_file_path.endswith(".mp3") or audio_file_path.endswith(".m4a"):
        print("audio file will get convert into .wav format")
        res = await convert_to_wav(audio_file_path)
        print(f"Audio file converted to {res}")

        base_options = python.BaseOptions(model_asset_path='classifier.tflite')
        options = audio.AudioClassifierOptions(base_options=base_options, max_results=4)

        with audio.AudioClassifier.create_from_options(options) as classifier:
            sample_rate, wav_data = wavfile.read(res)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(wav_data)
            audio_clip = containers.AudioData.create_from_array(
                wav_data.astype(float) / np.iinfo(np.int16).max, sample_rate)
            classification_result_list = classifier.classify(audio_clip)
        print("=====>>>>")
        music_category_count = 0
        duration = await audio_length(res)
        num_values = 6
        duration_list = []
        for _ in range(num_values):
            random_duration = random.randint(100, duration)
            duration_list.append(random_duration)
        # print(duration, "===============>")
        # for idx, timestamp in enumerate([0, 975, 1950, 2925]):
        for idx, timestamp in enumerate(duration_list):
            classification_result = classification_result_list[idx]
            top_category = classification_result.classifications[0].categories[0]
            # print(f'Timestamp {timestamp}: {top_category.category_name} ({top_category.score:.2f})')

            if top_category.category_name.lower() == "music":
                music_category_count += 1

    else:
        base_options = python.BaseOptions(model_asset_path='classifier.tflite')
        options = audio.AudioClassifierOptions(base_options=base_options, max_results=4)

        with audio.AudioClassifier.create_from_options(options) as classifier:
            sample_rate, wav_data = wavfile.read(audio_file_path)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(wav_data)
            audio_clip = containers.AudioData.create_from_array(wav_data.astype(float) / np.iinfo(np.int16).max, sample_rate)
            classification_result_list = classifier.classify(audio_clip)

        music_category_count = 0
        duration = await audio_length(audio_file_path)
        num_values = 6
        duration_list = []
        for _ in range(num_values):
            random_duration = random.randint(100, duration)
            duration_list.append(random_duration)
        # print(duration, "===============>")
        # for idx, timestamp in enumerate([0, 975, 1950, 2925]):
        for idx, timestamp in enumerate(duration_list):

            classification_result = classification_result_list[idx]
            top_category = classification_result.classifications[0].categories[0]
            # print(f'Timestamp {timestamp}: {top_category.category_name} ({top_category.score:.2f})')

            if top_category.category_name.lower() == "music":
                music_category_count += 1

    if music_category_count >= 1:
        return {"Result": False}
    else:
        return {"Result": True}

async def convert_to_wav(audio_file_path):
    sound = AudioSegment.from_file(audio_file_path)
    output_format = "wav"

    audio_file_name = audio_file_path.split(".")
    converted_audio_file = f"./{audio_file_name[0]}.wav"

    sound.export(converted_audio_file, format=output_format)
    print("Done")
    return converted_audio_file

async def audio_length(audio_file_path):
    sound = AudioSegment.from_file(audio_file_path)
    return len(sound)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port = "4400")



