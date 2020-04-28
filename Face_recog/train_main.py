# Face Recognition
from config import get_config
from Learner import face_learner
import argparse

# Person Detector
from Yolo_v2_pytorch.train_yolo import get_args as get_fd_args
from Yolo_v2_pytorch.train_yolo import train as fd_train

# TRACKING(FACE RECOGNITION) - not completed
# ========================================================
# learner = face_learner(conf)
# learner.train(conf, args.epochs)
# ========================================================

# PERSON DETECTOR
# ========================================================
fd_args = get_fd_args()
fd_train(fd_args)
# ========================================================

# MAKE YOUR LOADER IN YOUR TRAIN CLASS
# ========================================================

# ========================================================