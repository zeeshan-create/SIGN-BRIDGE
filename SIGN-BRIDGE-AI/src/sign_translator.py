import os
import speech_recognition as sr
import requests
from bs4 import BeautifulSoup
from moviepy.editor import VideoFileClip, concatenate_videoclips
import pysrt
import re

class SignTranslator:
    def __init__(self, signs_dir='data/signs_database'):
        self.signs_dir = signs_dir
        self.recognizer = sr.Recognizer()
        self._load_available_signs()

    def _load_available_signs(self):
        """Build database of available sign clips."""
        if not os.path.exists(self.signs_dir):
            os.makedirs(self.signs_dir)
        self.available_signs = {
            f.split('.')[0].lower(): os.path.join(self.signs_dir, f)
            for f in os.listdir(self.signs_dir) if f.endswith(('.mp4', '.mov', '.avi'))
        }
        print(f"[SignTranslator] Loaded {len(self.available_signs)} signs.")

    def text_to_glosses(self, text):
        """Simplistic English to ASL Gloss translation (normalization)."""
        # Basic normalization for mapping: lowercase, remove punctuation
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        # In professional sign translators, this would involve NLP for grammar (OVS/OSV etc)
        # For now, we perform a direct keyword mapping to our available clips.
        return [w for w in words if w in self.available_signs]

    def generate_sign_video(self, words, output_filename="translated_asl.mp4"):
        """Concatenate video clips for the given gloss sequence."""
        clips = []
        for word in words:
            if word in self.available_signs:
                clip_path = self.available_signs[word]
                clips.append(VideoFileClip(clip_path))
            else:
                print(f"[Warning] Sign for '{word}' not found.")

        if not clips:
            print("[Error] No valid signs found to build a video.")
            return None

        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(output_filename, fps=24, codec='libx264', audio=False)
        print(f"[Success] Sign video generated: {output_filename}")
        return output_filename

    # --- INPUT FEATURES ---

    def from_text(self, text, output_file="text_translation.mp4"):
        """Feature: Text -> Sign Language"""
        print(f"Processing Text: {text}")
        glosses = self.text_to_glosses(text)
        return self.generate_sign_video(glosses, output_file)

    def from_speech(self, audio_source=None, output_file="speech_translation.mp4"):
        """Feature: Speech / Audio -> Sign Language"""
        print("Listening for speech...")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"Recognized Speech: {text}")
                return self.from_text(text, output_file)
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Speech service error: {e}")
        return None

    def from_url(self, url, output_file="url_translation.mp4"):
        """Feature: URLs -> Sign Language"""
        print(f"Scraping content from: {url}")
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract main text content (headers and paragraphs)
            text = " ".join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
            # Limit length for demo performance
            text = text[:500] 
            return self.from_text(text, output_file)
        except Exception as e:
            print(f"URL processing error: {e}")
        return None

    def from_captions(self, srt_path, output_file="caption_translation.mp4"):
        """Feature: Captions / Subtitles -> Sign Language"""
        print(f"Parsing SRT: {srt_path}")
        try:
            subs = pysrt.open(srt_path)
            full_text = " ".join([sub.text for sub in subs])
            # Remove HTML/Formatting tags if present
            full_text = re.sub(r'<[^>]*>', '', full_text)
            return self.from_text(full_text, output_file)
        except Exception as e:
            print(f"Caption processing error: {e}")
        return None

    def from_video(self, video_path, output_file="video_translation.mp4"):
        """Feature: Video Input -> Sign Translation (Extract audio and transcribe)"""
        print(f"Processing video: {video_path}")
        try:
            temp_audio = "temp_transcription.wav"
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(temp_audio, codec='pcm_s16le')
            
            with sr.AudioFile(temp_audio) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                print(f"Transcribed Video: {text}")
                os.remove(temp_audio)
                return self.from_text(text, output_file)
        except Exception as e:
            print(f"Video input processing error: {e}")
        return None

if __name__ == "__main__":
    # Internal component testing
    translator = SignTranslator()
    # translator.from_text("hello thanks goodbye") # Example test
