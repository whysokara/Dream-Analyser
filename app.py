import speech_recognition as sr
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the speech recognizer
recognizer = sr.Recognizer()

def capture_voice():
    """Capture voice input and convert to text using SpeechRecognition with free models"""
    with sr.Microphone() as source:
        print("Please speak now. I'm listening...")
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        # Listen for input
        audio = recognizer.listen(source)
        
    try:
        # Use the free Sphinx model for offline recognition
        # Alternatively, use Google's service with a daily limit: 
        text = recognizer.recognize_google(audio)
        # text = recognizer.recognize_sphinx(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
        return None
    except sr.RequestError as e:
        print(f"Recognition service error: {e}")
        return None

def clean_text(text):
    """Basic text cleaning"""
    if not text:
        return None
    
    # Remove common filler words - expand as needed
    fillers = ["um", "uh", "like", "you know"]
    for filler in fillers:
        text = text.replace(f" {filler} ", " ")
    
    # Trim extra spaces
    text = " ".join(text.split())
    return text

def main():
    print("Voice to Text - Speak your prompt")
    print("--------------------------------")
    
    # Capture voice and convert to text
    text = capture_voice()
    
    # Clean the text
    if text:
        cleaned_text = clean_text(text)
        print(f"Processed text: {cleaned_text}")
    else:
        print("No text to process. Please try again.")

if __name__ == "__main__":
    main()