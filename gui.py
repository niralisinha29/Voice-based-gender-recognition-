
import numpy as np
import librosa
import pyaudio
import wave
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    return WAVE_OUTPUT_FILENAME

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained with accuracy on test set:", accuracy_score(y_test, model.predict(X_test)))
    return model

class GenderPredictorApp(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Voice based gender prediction')
        self.setGeometry(100, 100, 400, 200)
        self.label = QLabel('Click record to predict gender from voice', self)
        self.btn = QPushButton('Record', self)
        self.btn.clicked.connect(self.perform_prediction)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.btn)
    
        

    def perform_prediction(self):
        filename = record_audio()
        features = extract_features(filename)
        prediction = self.model.predict([features])
        predicted_gender = "Male" if prediction[0] == 0 else "Female"
        self.label.setText(f"Predicted Gender: {predicted_gender}")

def main():
    app = QApplication(sys.argv)

    # Simulate loading your dataset here
    # For demo purposes, you would need a dataset with corresponding labels
    # X, y = load_dataset()

    # model = train_model(X, y)  # Normally you'd load your trained model

    # Dummy model and data for demonstration
    X_dummy = np.random.rand(100, 40)  # 100 samples, 40 features each
    y_dummy = np.random.randint(0, 2, 100)
    model = train_model(X_dummy, y_dummy)

    ex = GenderPredictorApp(model)
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

