# Real Time American Sign Language (ASL) Translator

This is a train AI model that uses Google's MediaPipe library to track hand movments in real-time.  The movements are translated into english alphabet letters.

## Requirements

**Python Version:**
You must use Python 3.11. Mediapipe is not supported on Python 3.13. If you have multiple Python versions installed (e.g., via Homebrew), use `python3.11` for all commands below.

## Installation

1. **Clone the repository:**
	```bash
	git clone https://github.com/nejohnson2/real-time-asl-translator.git
	cd real-time-asl-translator
	```

2. **Install required Python packages:**
	```bash
	python3.11 -m pip install --upgrade pip
	python3.11 -m pip install -r requirements.txt
	```

## Running the Real-Time ASL Translator

To start the real-time ASL translator, run:

```bash
python3.11 asl-translator/realtime_asl.py
```

This will open your webcam and begin recognizing ASL hand signs, displaying the predicted letter in real time.

## Troubleshooting

- If you encounter issues with Mediapipe or OpenCV, ensure you are using Python 3.11.
- The model file (`models/asl_model.pt`) must exist. If missing, retrain using `asl-translator/train_model.py`.

## Project Structure

- `asl-translator/`: Main Python scripts
- `models/`: Trained model file
- `data/`: Collected landmark data
- `documents/`: Reference images

## License

MIT License