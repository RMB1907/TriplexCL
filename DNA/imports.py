import os,warnings

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hides INFO, shows WARNING+ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # optional for reproducible CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # force CPU, hide CUDA warnings

# Hide Python warnings (optional)
warnings.filterwarnings("ignore")

#import absl.logging
#absl.logging.set_verbosity(absl.logging.ERROR)

