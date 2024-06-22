import argparse
import base64
from threading import Lock, Thread
import os
import time
import logging
import cv2
import numpy as np
import openai
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError
from collections import deque
import pyautogui
import sounddevice as sd
import queue
import sys

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a file handler to save logs
file_handler = logging.FileHandler('assistant_log.txt')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

class OnScreenLogger:
    def __init__(self, max_lines=10):
        self.logs = deque(maxlen=max_lines)

    def log(self, message, level=logging.INFO):
        self.logs.append(message)
        logger.log(level, message)

    def get_logs(self):
        return list(self.logs)

class WebcamStream:
    def __init__(self, camera_index=0):
        self.stream = cv2.VideoCapture(camera_index)
        self.frame = None
        self.running = False
        self.lock = Lock()
        self.frame_logged = False

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            if not self.stream.isOpened():
                if not self.frame_logged:
                    logger.warning("Webcam is not opened")
                    self.frame_logged = True
                self.stream = cv2.VideoCapture(0)
                time.sleep(1)
                continue
            ret, frame = self.stream.read()
            if ret:
                with self.lock:
                    self.frame = frame
                if not self.frame_logged:
                    logger.info("Frame available")
                    self.frame_logged = True
            else:
                if not self.frame_logged:
                    logger.warning("Failed to read frame from webcam")
                    self.frame_logged = True
                time.sleep(0.1)

    def read(self, encode=False):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()
        if encode:
            # Resize the frame before encoding to reduce the token count
            frame = self.resize_image(frame)
            _, buffer = cv2.imencode(".jpeg", frame)
            return base64.b64encode(buffer).decode()
        return frame

    def resize_image(self, frame, max_size=512):
        height, width = frame.shape[:2]
        if height > max_size or width > max_size:
            scaling_factor = max_size / float(max(height, width))
            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

class DesktopStream:
    def __init__(self):
        self.frame = None
        self.running = False
        self.lock = Lock()
        self.frame_logged = False

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            try:
                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                with self.lock:
                    self.frame = frame
                if not self.frame_logged:
                    logger.info("Frame available")
                    self.frame_logged = True
            except Exception as e:
                if not self.frame_logged:
                    logger.error(f"Error capturing desktop: {str(e)}")
                    self.frame_logged = True
            time.sleep(0.1)

    def read(self, encode=False):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()
        if encode:
            # Resize the frame before encoding to reduce the token count
            frame = self.resize_image(frame)
            _, buffer = cv2.imencode(".jpeg", frame)
            return base64.b64encode(buffer).decode()
        return frame

    def resize_image(self, frame, max_size=512):
        height, width = frame.shape[:2]
        if height > max_size or width > max_size:
            scaling_factor = max_size / float(max(height, width))
            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

class Assistant:
    def __init__(self, model, on_screen_logger):
        self.chain = self._create_inference_chain(model)
        self.logger = on_screen_logger

    def answer(self, prompt, image):
        if not prompt:
            return

        self.logger.log(f"Prompt: {prompt}")

        try:
            response = self.chain.invoke(
                {"prompt": prompt, "image_base64": image},
                config={"configurable": {"session_id": "unused"}},
            ).strip()

            self.logger.log(f"Response: {response}")

            if response:
                self._tts(response)
        except Exception as e:
            self.logger.log(f"Error generating response: {str(e)}")

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        try:
            with openai.Audio.create(
                model="tts-1",
                voice="alloy",
                response_format="pcm",
                input=response,
            ) as stream:
                for chunk in stream.iter_bytes(chunk_size=1024):
                    player.write(chunk)
        except Exception as e:
            self.logger.log(f"Error in TTS: {str(e)}")
        finally:
            player.close()

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {"type": "text", "text": "{image_base64}"},
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

def get_model(api_choice):
    try:
        if api_choice == 'azure':
            return AzureChatOpenAI(
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            ), 'azure'
        elif api_choice == 'openai':
            openai.api_type = 'openai'
            return ChatOpenAI(
                model="gpt-4",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            ), 'openai'
        elif api_choice == 'google':
            return ChatGoogleGenerativeAI(
                model="gemini-pro-vision",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            ), 'google'
        else:
            raise ValueError("Invalid API choice")
    except Exception as e:
        logger.error(f"Error initializing API client: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="AI Assistant with multiple API and capture options")
    parser.add_argument("--api", choices=['azure', 'openai', 'google'], required=True, help="Choose the API to use")
    parser.add_argument("--capture", choices=['webcam', 'desktop'], required=True, help="Choose capture method")
    args = parser.parse_args()

    on_screen_logger = OnScreenLogger()

    try:
        if args.capture == 'webcam':
            capture_stream = WebcamStream().start()
        else:
            capture_stream = DesktopStream().start()
    except Exception as e:
        on_screen_logger.log(f"Error initializing capture stream: {str(e)}", level=logging.ERROR)
        sys.exit(1)

    try:
        api_client, api_type = get_model(args.api)
    except Exception as e:
        on_screen_logger.log(f"Error initializing API client: {str(e)}", level=logging.ERROR)
        sys.exit(1)

    assistant = Assistant(api_client, on_screen_logger)

    recognizer = Recognizer()
    microphone = Microphone()

    def audio_callback(recognizer, audio):
        try:
            on_screen_logger.log("Processing audio...")
            prompt = recognizer.recognize_whisper(audio, model="base", language="english")
            on_screen_logger.log(f"Recognized: {prompt}")
            frame = capture_stream.read(encode=True)
            if frame is not None:
                on_screen_logger.log("Passing audio and frame to the model")
                assistant.answer(prompt, frame)
            else:
                on_screen_logger.log("No frame available for processing")
        except UnknownValueError:
            on_screen_logger.log("Could not understand audio")
        except Exception as e:
            on_screen_logger.log(f"Unexpected error in audio processing: {str(e)}", level=logging.ERROR)

    stop_listening = recognizer.listen_in_background(microphone, audio_callback)

    on_screen_logger.log("Assistant is ready. Press 'q' to quit.")

    try:
        while True:
            frame = capture_stream.read()
            if frame is not None:
                if not hasattr(capture_stream, 'frame_logged') or not capture_stream.frame_logged:
                    on_screen_logger.log("Frame available for display")
                    capture_stream.frame_logged = True
                height, width = frame.shape[:2]
                max_height = 800
                if height > max_height:
                    scale = max_height / height
                    frame = cv2.resize(frame, (int(width * scale), max_height))

                logs = on_screen_logger.get_logs()
                for i, log in enumerate(logs):
                    cv2.putText(frame, log, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Capture Preview", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                on_screen_logger.log("No frame available", level=logging.WARNING)
            time.sleep(0.1)
    except KeyboardInterrupt:
        on_screen_logger.log("Interrupted by user. Stopping...")
    except Exception as e:
        on_screen_logger.log(f"Unexpected error in main loop: {str(e)}", level=logging.ERROR)
    finally:
        capture_stream.stop()
        cv2.destroyAllWindows()
        stop_listening(wait_for_stop=False)
        on_screen_logger.log("Assistant stopped.")

if __name__ == "__main__":
    main()
