import sounddevice as sd
import pyaudio
import numpy as np
import webrtcvad
import soundfile as sf
import time
from collections import deque
from faster_whisper import WhisperModel
import io
import noisereduce as nr

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000  # Required for webrtcvad
CHUNK = int(RATE / 50)  # 20ms chunks for VAD
THRESHOLD_SECONDS = 1.5  # Stop recording after this much silence
RECORD_SECONDS_LIMIT = 60  # Optional maximum recording time
BEAM_SIZE = 1
VAD_MODE = 3

class VoiceActivationDetector:
    def __init__(self):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(VAD_MODE)  # 0 = least aggressive, 3 = most aggressive
        self.buffer = deque(maxlen=int(RATE / 50))  # 20ms buffer

    def is_speech(self, frame, sample_rate):
        return self.vad.is_speech(frame, sample_rate)
    
    def reduce_noise(self,audio_data):
        """Apply noise reduction to the audio data."""
        return nr.reduce_noise(y=audio_data, sr=RATE)

class WhisperProcessor:
    def __init__(self, model_type, processor_type, compute_format):
        self.model = WhisperModel(model_type, device=processor_type, compute_type=compute_format)

    def transcribe_audio(self, audio_data, beam):

        audio_float32 = audio_data.astype(np.float32) / 32768.0
        buffer = io.BytesIO()
        sf.write(buffer, audio_float32, RATE, format="WAV", subtype="PCM_16")  # Write audio as WAV format
        buffer.seek(0)

        segments, info = self.model.transcribe(
            buffer,
            beam_size=beam,
            no_speech_threshold=0.4,  # More aggressive silence detection
            log_prob_threshold=-0.5, # Avoid low-confidence transcriptions
            vad_filter=True,         # Use VAD to remove non-speech
            vad_parameters={"min_silence_duration_ms": 200},  # Shorter silence for quicker filtering
        )

        transcribed_text = " ".join([segment.text for segment in segments])  # Join all texts with space
        return transcribed_text

class SpeechToText:
    def __init__(self):
        self.vad = VoiceActivationDetector()
        self.model = WhisperProcessor(model_type="tiny", processor_type="cpu", compute_format="float32")
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    def listen(self):
        recording = False
        frames = []
        silent_chunks = 0
        start_time = time.time()

        while True:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_data = self.vad.reduce_noise(audio_data)
            if self.vad.is_speech(audio_data.tobytes(), RATE):
                if not recording:
                    recording = True
                    print("Speech detected. Recording...")
                silent_chunks = 0
                frames.append(data)
            elif recording:
                    silent_chunks += 1
                    frames.append(data)

            if recording and silent_chunks > (THRESHOLD_SECONDS * 50):
                print("Silence detected. Stopping recording...")
                break
            # Stop recording after a maximum duration (safety net)
            if recording and (time.time() - start_time) > RECORD_SECONDS_LIMIT:
                print("Maximum recording time reached. Stopping...")
                break

        if frames:
            print("Recording complete.")
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            return self.model.transcribe_audio(audio_data=audio_data, beam=BEAM_SIZE)
        return None 
    
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
