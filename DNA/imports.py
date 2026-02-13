import os, warnings

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hides INFO, shows WARNING+ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # optional for reproducible CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # force CPU, hide CUDA warnings

# Hide Python warnings (optional)
warnings.filterwarnings("ignore")

# Standard library
import os
import math
import csv

# Scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# BioPython
from Bio import SeqIO

# TensorFlow / Keras (use ONE style: tensorflow.keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, Concatenate, Flatten,
    Dropout, Dense, Reshape, LSTM
)
from tensorflow.keras.optimizers import Adam

# Callbacks
from livelossplot import PlotLossesKerasTF as PlotLossesCallback

# Scikit-learn
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

# Gradio
import gradio as gr
