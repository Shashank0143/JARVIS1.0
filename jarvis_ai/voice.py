from __future__ import annotations

import os


class VoiceInterface:
    def __init__(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError as exc:
            raise RuntimeError("speech_recognition is not installed.") from exc

        self.sr = sr
        self.recognizer = sr.Recognizer()
        self.speaker = None
        try:
            import pyttsx3

            self.speaker = pyttsx3.init()
        except Exception:
            self.speaker = None

    def listen_once(self, *, timeout: int | None = None, phrase_time_limit: int = 8) -> str:
        with self.sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
            )
        try:
            return self.recognizer.recognize_sphinx(audio).strip()
        except Exception as exc:
            if os.environ.get("JARVIS_ALLOW_CLOUD_STT") == "1":
                return self.recognizer.recognize_google(audio).strip()
            raise RuntimeError(
                "Local speech recognition is unavailable. Install pocketsphinx for offline voice, "
                "or set JARVIS_ALLOW_CLOUD_STT=1 to allow online speech recognition."
            ) from exc

    def speak(self, text: str) -> None:
        if not self.speaker:
            return
        self.speaker.say(text)
        self.speaker.runAndWait()
