import sys
from pathlib import Path
from utils.argutils import print_args
from utils.modelutils import check_model_paths
import argparse
import os
import sklearn.utils._typedefs
import sklearn.neighbors._partition_nodes
import librosa
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtGui import QPixmap
from toolbox import Toolbox

# pyinstaller -D -w -i ./toolbox/assets/diana.ico ./demo_toolbox.py --additional-hooks=extra-hooks --additional-hooks-dir "E
# :\MockingBird-main\MockingBird-main/extra-hooks"

class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(200, 100)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.CustomizeWindowHint | Qt.Window)
        self.image_label = QLabel(self)

        self.pixmap_list = []
        self.pixmap_list.append(QPixmap('./toolbox/assets/diana.webp'))

        # self.image_label.resize(256, 256)
        self.image_label.setAlignment(Qt.AlignCenter)
        # label.setPixmap(pixmap)
        self.image_label.setPixmap(self.pixmap_list[0].scaled(self.image_label.size(), Qt.KeepAspectRatio))
        self.show()

    def close_loading_view(self):
        self.close()

if __name__ == '__main__':

    # import pdb
    #
    # app = QApplication(sys.argv)
    # # window = QMainWindow()
    # demo = LoadingScreen()

    # demo.mainUI(window)
    # window.show()
    # app.exec_()
    # pdb.set_trace()
    # loading = LoadingScreen()

    parser = argparse.ArgumentParser(
        description="Runs the toolbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-d", "--datasets_root", type=Path, help= \
        "Path to the directory containing your datasets. See toolbox/__init__.py for a list of "
        "supported datasets.", default=None)
    parser.add_argument("-e", "--enc_models_dir", type=Path, default="encoder/saved_models",
                        help="Directory containing saved encoder models")
    parser.add_argument("-s", "--syn_models_dir", type=Path, default="synthesizer/saved_models",
                        help="Directory containing saved synthesizer models")
    parser.add_argument("-v", "--voc_models_dir", type=Path, default="vocoder/saved_models",
                        help="Directory containing saved vocoder models")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("--no_mp3_support", action="store_true", help=\
        "If True, no mp3 files are allowed.")
    args = parser.parse_args()
    print_args(args, parser)

    if args.cpu:
        # Hide GPUs from Pytorch to force CPU processing
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    del args.cpu
    ## Remind the user to download pretrained models if needed
    check_model_paths(encoder_path=args.enc_models_dir, synthesizer_path=args.syn_models_dir,
                      vocoder_path=args.voc_models_dir)

    # Launch the toolbox
    Toolbox(**vars(args))
