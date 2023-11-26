
if __name__ == "__main__":
    # System imports
    from venv import create
    from os.path import join, expanduser, abspath
    from subprocess import run

    # # Create virtual environment
    # try:
    #     dir = join(expanduser("."), "venv")
    #     create(dir, with_pip=True)
    #     print("Virtual environment created on: ", dir)
    # except Exception as e:
    #     raise Exception("Failed to create the virtual environment: " + str(e))

    # # Install packages in 'requirements.txt'.
    # try:
    #     run(["python3", "-m", "pip3", "install", "--upgrade", "pip3"])
    #     run(["bin/pip3", "install", "-r", abspath("requirements.txt")], cwd=dir)
    # except:
    #     run(["python", "-m", "pip", "install", "--upgrade", "pip"])
    #     run(["bin/pip", "install", "-r", abspath("requirements.txt")], cwd=dir)
    # finally:
    #     print("Completed installation of requirements.")

    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import idx2numpy
    from tqdm.autonotebook import tqdm
    from IPython import display
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.metrics import Precision, Recall

    from src.leNet import leNet
    from src.Gan import GAN, gan_loss, Gan
    from src.auxiliary import partitioning
    from src.plotting import show_images, show_performance_curve, plot_reconstruction

    leNet(plotting = True)
    Gan(plotting = True)
