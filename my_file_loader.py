import os
import sys

def load_shape_predictor_dat_file():
    # Get the directory where the executable or script is located
    if getattr(sys, 'frozen', False):  # PyInstaller sets this attribute
        # Running in a bundle (executable)
        exe_dir = sys._MEIPASS  # PyInstaller stores the extracted data files here
    else:
        # Running in a normal Python environment
        exe_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the shape_predictor_68_face_landmarks.dat file
    dat_file_path = os.path.join(exe_dir, 'shape_predictor_68_face_landmarks.dat')

    # Check if the file exists
    if os.path.exists(dat_file_path):
        # Load the file (example: using openCV)
        # Example: predictor = dlib.shape_predictor(dat_file_path)
        print("Successfully loaded shape_predictor_68_face_landmarks.dat file.")
        return dat_file_path
    else:
        print("Error: shape_predictor_68_face_landmarks.dat file not found at:", dat_file_path)


def load_example_video():
    # Get the directory where the executable or script is located
    if getattr(sys, 'frozen', False):  # PyInstaller sets this attribute
        # Running in a bundle (executable)
        exe_dir = sys._MEIPASS  # PyInstaller stores the extracted data files here
    else:
        # Running in a normal Python environment
        exe_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the shape_predictor_68_face_landmarks.dat file
    dat_file_path = os.path.join(exe_dir, 'example_video.mp4')

    # Check if the file exists
    if os.path.exists(dat_file_path):
        # Load the file (example: using openCV)
        # Example: predictor = dlib.shape_predictor(dat_file_path)
        print("Successfully loaded example video file.")
        return dat_file_path
    else:
        print("Error: example video file not found at:", dat_file_path)


def load_model_file():
    # Get the directory where the executable or script is located
    if getattr(sys, 'frozen', False):  # PyInstaller sets this attribute
        # Running in a bundle (executable)
        exe_dir = sys._MEIPASS  # PyInstaller stores the extracted data files here
    else:
        # Running in a normal Python environment
        exe_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the shape_predictor_68_face_landmarks.dat file
    dat_file_path = os.path.join(exe_dir, 'model4.pt')

    # Check if the file exists
    if os.path.exists(dat_file_path):
        # Load the file (example: using openCV)
        # Example: predictor = dlib.shape_predictor(dat_file_path)
        print("Successfully loaded model file.")
        return dat_file_path
    else:
        print("Error: model file not found at:", dat_file_path)

