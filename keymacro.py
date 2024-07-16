import tkinter as tk
from pynput.keyboard import Key, Controller
import threading
import time

class TabPresser:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tab Key Presser")
        
        self.is_pressing = False
        self.press_thread = None
        self.keyboard = Controller()

        self.start_button = tk.Button(self.root, text="Start Pressing Tab", command=self.start_pressing)
        self.start_button.pack(pady=10)
        
        self.stop_button = tk.Button(self.root, text="Stop Pressing Tab", command=self.stop_pressing)
        self.stop_button.pack(pady=10)

        self.status_label = tk.Label(self.root, text="Status: Stopped")
        self.status_label.pack(pady=10)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def start_pressing(self):
        if not self.is_pressing:
            self.is_pressing = True
            self.status_label.config(text="Status: Pressing")
            self.press_thread = threading.Thread(target=self.press_tab)
            self.press_thread.start()

    def stop_pressing(self):
        if self.is_pressing:
            self.is_pressing = False
            self.status_label.config(text="Status: Stopped")
            if self.press_thread is not None:
                self.press_thread.join()
                self.press_thread = None

    def press_tab(self):
        while self.is_pressing:
            self.keyboard.press(Key.tab)
            time.sleep(0.1)
            self.keyboard.release(Key.tab)  # Ensure key is released

    def on_closing(self):
        self.stop_pressing()
        self.root.quit()  # Exit the main loop

if __name__ == "__main__":
    TabPresser()
