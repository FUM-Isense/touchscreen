import queue
import threading
import espeakng

class TextToSpeechPlayer:
    def __init__(self, language='en'):
        self.language = language
        # pygame.mixer.init()
        self.speaker = espeakng.Speaker(language=language,wpm=150)
        # self.text_queue = queue.Queue()
        # self.stop_flag = threading.Event()
        # self.speech_th = threading.Thread(target=self.__speak)
        # self.speech_th.start()

    # def __speak(self):
    #     stop_called = False
        
    #     while True and not stop_called:
    #         # pygame.time.Clock().tick(10)
    #         if not self.text_queue.empty():
    #             next_text = self.text_queue.get()
    #             self.speaker.say(next_text)
    #             self.speaker.wait()
                    
    #         else:
    #             if self.stop_flag.is_set():
    #                 stop_called = True

    def say(self, text):
        self.speaker.say(text)
        
    def stop(self):
        pass
        # self.stop_flag.set()
        # self.speech_th.join()