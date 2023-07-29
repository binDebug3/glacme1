import time

from pocketsphinx import LiveSpeech

import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

from display import *
from process import *
import complete_radar


def main():
    # Create an argument parser to handle command line arguments.----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=1,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=2,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()



    # initialize microphone objects ---------------------------------------------------------------
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the
    # SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)




    # Load / Download model -----------------------------------------------------------------------
    model = args.model
    # TODO reinsert not english
    if args.model != "large" and args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)




    # Listening ----------------------------------------------------------------------------------
    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")




    # Main loop ----------------------------------------------------------------------------------
    while True:
        start = time.perf_counter()
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())




                """  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  INSERTED CODE START  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

                speech_window = 'rolling_audio.wav'
                print(get_audio_duration(temp_file))
                # visualize_audio(temp_file)
                # start_dur, end_dur = remove_silence(temp_file, 0.02)
                # print("Start duration: ", start_dur, "-- End duration: ", end_dur)
                # visualize_audio(temp_file)

                append_to_wav(speech_window, temp_file)
                # play_audio(speech_window)
                cut_wav(speech_window, 3)
                # play_audio(speech_window)

                # location = complete_radar.radar()


                """  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  INSERTED CODE FINISH  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """


                # Read the transcription.
                time_transcribe = time.perf_counter()
                result = audio_model.transcribe(speech_window, fp16=torch.cuda.is_available())
                print(f"Transcribed in {round(time.perf_counter() - time_transcribe, 2)} seconds")
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise, edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                    print(text)
                else:
                    transcription[-1] = text

                # # Clear the console to reprint the updated transcription.
                # os.system('cls' if os.name == 'nt' else 'clear')
                # for line in transcription:
                #     print(line)
                # print(f"[{round(time.perf_counter() - start, 2)} seconds behind]")
                # # Flush stdout.
                # print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)





# first attempt testing ---------------------------------------------------------------------------
def whisper_test():
    model = whisper.load_model("base")
    try:
        result = model.transcribe("glacme1/data/archive/harvard.wav")
    except RuntimeError:
        result = model.transcribe(r"C:\Users\dalli\source\repos\glacme1\data\archive\harvard.wav")

    print(result["text"])

def whisper_testing():
    for phrase in LiveSpeech():
        print(phrase)


def playground():
    start = time.perf_counter()





if __name__ == "__main__":
    main()
    # whisper_testing()
