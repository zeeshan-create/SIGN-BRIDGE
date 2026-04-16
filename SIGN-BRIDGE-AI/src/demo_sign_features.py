import sys
from sign_translator import SignTranslator
import os

def print_menu():
    print("\n" + "="*40)
    print("      ASL INPUT FEATURES DEMO")
    print("="*40)
    print("1. Text -> ASL Sign Video")
    print("2. Speech -> ASL Sign Video (Use Mic)")
    print("3. URL -> ASL Sign Video (Scrape & Translate)")
    print("4. Captions (SRT) -> ASL Sign Video")
    print("5. Video File -> ASL Sign Translation")
    print("q. Quit")
    print("="*40)

def main():
    # Make sure we have a directory for results
    if not os.path.exists('results'):
        os.makedirs('results')

    translator = SignTranslator(signs_dir='data/signs_database')
    
    while True:
        print_menu()
        choice = input("Select a feature to test: ").lower()

        if choice == '1':
            text = input("Enter text to translate: ")
            if not text: text = "hello thanks goodbye"
            translator.from_text(text, output_file="results/text_demo.mp4")
            print("Check results/text_demo.mp4")

        elif choice == '2':
            print("Listening (Speak into your mic)...")
            translator.from_speech(output_file="results/speech_demo.mp4")
            print("Check results/speech_demo.mp4")

        elif choice == '3':
            url = input("Enter URL (default: https://en.wikipedia.org/wiki/American_Sign_Language): ")
            if not url: url = "https://en.wikipedia.org/wiki/American_Sign_Language"
            translator.from_url(url, output_file="results/url_demo.mp4")
            print("Check results/url_demo.mp4")

        elif choice == '4':
            # Create a mock SRT if it doesn't exist
            srt_path = 'sample.srt'
            if not os.path.exists(srt_path):
                with open(srt_path, 'w') as f:
                    f.write("1\n00:00:00,000 --> 00:00:05,000\nHello and thanks for coming.\n")
            
            translator.from_captions(srt_path, output_file="results/caption_demo.mp4")
            print("Check results/caption_demo.mp4")

        elif choice == '5':
            path = input("Enter path to video file: ")
            if os.path.exists(path):
                translator.from_video(path, output_file="results/video_trans_demo.mp4")
                print("Check results/video_trans_demo.mp4")
            else:
                print("Skipping: Video file not found.")

        elif choice == 'q':
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
