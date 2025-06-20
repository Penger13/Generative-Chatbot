from enum import Enum
import os

DATA_FOLDER = os.path.join("data")
UBUNTU_CORPUS_FOLDER = os.path.join("..", "Ubuntu-dialogue-corpus")
DATASET_FILE_PREFIX = "ubuntu_dialogue_corpus"
PYTORCH_MODELS_FOLDER = "models"

class DatasetSize(Enum):
    """ Ubuntu Dialogue Corpus size variants """
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    def __str__(self):
        return self.value

# Ubuntu Dialogue Dataset variants
# https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus
# "Each file contains dialogues from one folder" - this is incorrect, medium and large contain dialogues from multiple folders
UBUNTU_CORPUS_DATASET_FILE_SUFFIXES = {
    DatasetSize.SMALL: "",
    DatasetSize.MEDIUM: "_196",
    DatasetSize.LARGE: "_301"     # 930,000 dialogues and over 100,000,000 words
}

# Character to use when joining folder and dialogueID fields
FOLDER_DIALOGUE_ID_SEPARATOR = "/"

# Filter the dataset to this number of rows - set to None for all rows
#MAX_ROWS = 100000
MAX_ROWS = None
# Maximum number of concurrent messages from user to join into one question or answer
MAX_CONCURRENT_MESSAGES = 3
DATA_ANALYSIS = False
SHOW_PLOTS = False
PROFILE_DATASET = False
# The label used by the question vs. statement classifier to indicated a question
QUESTION_STATEMENT_CLASSIFIER_QUESTION_LABEL = "LABEL_1"

# Training parameters
NUM_TRAIN_EPOCHS = 3
NUM_EPOCHS = 10
DEFAULT_MAX_LEN = 100
MAX_LENGTH = 384
DEFAULT_BATCH_SIZE = 256
DEFAULT_EARLY_STOP_COUNT = 4
PAD = 3

# Some sample questions
TEST_QUESTIONS = [
    "Tell me all that you know about Ubuntu.",
    "How does java relate to Ubuntu?",
    "How do I solve an issue with Ubuntu?",
    "Can anyone tell me how I get around the root password problem?",
    "Does Ubuntu come with a firewall by default?",
    "Can someone tell me how to get rid of Google Chrome?",
    "Is there a way to see if a hard disk has bad blocks on Ubuntu?",
    "What's the best way for a bash script to pick up variables from /etc/environment?",
    "Is there a CLI command to roll back any updates/upgrades I made recently?",
    "Is there a way to adjust gamma settings in totem?",
    "Is there a graphical way to search for an NFS server on Gutsy?",
    "How do I move a file from one place to another using the console?",
    "How do I fix bad disk sectors?",
    "How do I upgrade to the latest Ubuntu version?",
]

# Some sample question and answer pairs
TEST_QUESTION_ANSWERS = [
    ("How do I move a file from one place to another using the console?", "Use mv"),
    ("How do I fix bad disk sectors?", "Use fsck"),
    ("How do I uninstall Chrome?", "sudo apt-get remove chrome"),
    ("Where can I find a log of the latest updates that Ubuntu has done?", "In /var/log/dpkg"),
]
