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

# Initialize PyAudio and VAD
audio = pyaudio.PyAudio()
vad = webrtcvad.Vad()
vad.set_mode(3)  # 0 = least aggressive, 3 = most aggressive

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

        # for segment in segments:
        #     # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        #     print("%s" % (segment.text))
        transcribed_text = " ".join([segment.text for segment in segments])  # Join all texts with space
        print(transcribed_text)

class VoiceActivationDetector:
    def __init__(self):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # 0 = least aggressive, 3 = most aggressive
        self.buffer = deque(maxlen=int(RATE / 50))  # 20ms buffer

    def is_speech(self, frame, sample_rate):
        return self.vad.is_speech(frame, sample_rate)


model = WhisperProcessor(model_type="tiny", processor_type="cpu", compute_format="float32")

# Open stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Listening for speech...")

def is_speech(frame, sample_rate):
    """Check if the audio frame contains speech using VAD."""
    return vad.is_speech(frame, sample_rate)

def reduce_noise(audio_data):
    """Apply noise reduction to the audio data."""
    return nr.reduce_noise(y=audio_data, sr=RATE)

try:
    while True:
        print("Waiting for speech...")
        recording = False
        frames = []
        silent_chunks = 0
        start_time = time.time()

        while True:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)

            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_data = reduce_noise(audio_data)
            # Check if current chunk contains speech
            if is_speech(audio_data.tobytes(), RATE):
                if not recording:
                    print("Speech detected. Recording...")
                    recording = True
                silent_chunks = 0  # Reset silence counter
                frames.append(data)
            else:
                if recording:
                    silent_chunks += 1
                    frames.append(data)

            # Check if silence exceeds threshold
            if recording and silent_chunks > (THRESHOLD_SECONDS * 50):
                print("Silence detected. Stopping recording...")
                break

            # Stop recording after a maximum duration (safety net)
            if recording and (time.time() - start_time) > RECORD_SECONDS_LIMIT:
                print("Maximum recording time reached. Stopping...")
                break

        # Process and save the recorded audio if anything was captured
        if frames:
            print("Recording complete.")

            # Convert recorded data to NumPy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

            # Optionally save to a file
            # filename = f"recorded_audio_temp.mp3"
            # sf.write(filename, audio_data, RATE)

            model.transcribe_audio(audio_data=audio_data, beam=BEAM_SIZE)
            # model.transcribe_audio(audio_data=filename, beam=BEAM_SIZE)
            # Play back the recorded audio

        time.sleep(0.1)  # Short break before listening again

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Clean up
    stream.stop_stream()
    stream.close()
    audio.terminate()
