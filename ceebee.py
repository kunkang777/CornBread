#!/usr/bin/env python3

import sys
import wave
import tempfile
import os
import logging
import argparse
import time
import sys
import speech_to_text as stt
# LLM
from ollama import chat

# Text-to-Speech
sys.path.append('piper/src/python_run')
from piper import PiperVoice

# Audio Playback
import simpleaudio as sa

def sanitize_text(text: str) -> str:
    """Remove non-ASCII characters (including emojis)."""
    return text.encode("ascii", "ignore").decode("ascii")

class PersonalAssistant:
    def __init__(
        self, 
        llm_model="gemma2:2b",
        name="CeeBee",
        piper_model="voice_models/en_GB-alan-medium.onnx",  # Hard-coded default
        piper_config=None,
        use_cuda=False,
        length_scale=0.7,
        noise_scale=0.3,
        noise_w=0.5
    ):
        self.llm_model = llm_model
        self.name = name
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w

        if not piper_config:
            piper_config = f"{piper_model}.json"

        # Load Piper model
        self.piper_voice = PiperVoice.load(
            model_path=piper_model,
            config_path=piper_config,
            use_cuda=use_cuda
        )

    def speak_text(self, text: str):
        """Synthesize text with Piper and play it using simpleaudio."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        with wave.open(tmp_path, "wb") as wav_file:
            self.piper_voice.synthesize(
                text,
                wav_file,
                length_scale=self.length_scale,
                noise_scale=self.noise_scale,
                noise_w=self.noise_w,
            )
        print("DEBUG: length_scale=", self.length_scale,
              "noise_scale=", self.noise_scale,
              "noise_w=", self.noise_w)

        try:
            wave_obj = sa.WaveObject.from_wave_file(tmp_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        finally:
            os.remove(tmp_path)

    def response(self, user_message: str):
        """Generate response with Ollama, replace prefixes to remove the period, then speak."""
        system_prompt = (
            f"You are {self.name}, an advanced AI assistant modeled after Jarvis "
            "from Iron Man. Response with the wit and mildly sarcastic tone. "
            "Keep your response simple but answer the question "
            "completely. Always refer to the user as 'Mr Kang' or 'Sir'. "
            "Constraint: do not use onomatopoeia and no beeping noises."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            text_response = []
            stream = chat(model=self.llm_model, messages=messages, stream=True)
            for chunk in stream:
                chunk_text = chunk["message"]["content"]
                text_response.append(chunk_text)
                # print(chunk_text, end="", flush=True)

            # print()  # newline
            final_response = "".join(text_response)
            for prefix in ["Mr.", "Dr.", "Ms.", "Mrs.", ",", "*"]:
                final_response = final_response.replace(prefix, prefix[:-1])

            final_response = final_response.replace("...", ".")
            sanitized_text = sanitize_text(final_response)
            print(sanitized_text)
            self.speak_text(sanitized_text)

        except Exception as e:
            error_msg = f"\nApologies, Sir. I encountered an issue: {str(e)}"
            print(error_msg)
            self.speak_text("Apologies, Sir. I encountered an issue.")

def main():
    # logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma2:2b", help="Ollama model name")
    parser.add_argument("--name", default="CeeBee", help="AI Assistant name")
    parser.add_argument("--cuda", action="store_true", help="Use GPU (onnxruntime-gpu)")
    parser.add_argument("--length-scale", type=float, default=0.7, help="Speech speed")
    parser.add_argument("--noise-scale", type=float, default=1.2, help="Voice variation")
    parser.add_argument("--noise-w", type=float, default=1.5, help="Pitch/energy variability")
    args = parser.parse_args()

    ceebee = PersonalAssistant(
        llm_model=args.model,
        name=args.name,
        use_cuda=args.cuda,
        length_scale=args.length_scale,
        noise_scale=args.noise_scale,
        noise_w=args.noise_w,
    )

    ceebee.response("say: welcome back sir. all systems are online. But vary the response slightly to add variety")
    speech_listener = stt.SpeechToText()
    try:
        while True:
            user_input = speech_listener.listen()
            print(user_input)
            if user_input.lower() in ["shutdown", "quit", "exit", "shut down"]:
                print("Goodbye Sir")
                ceebee.response("Say: Shutting down, Goodbye. But vary the response slightly to add variety")
                break
            ceebee.response(user_input)
    except KeyboardInterrupt:
        ceebee.response("Say: Shutting down, Goodbye. But vary the response slightly to add variety")
    finally:
        speech_listener.close()

if __name__ == "__main__":
    main()
