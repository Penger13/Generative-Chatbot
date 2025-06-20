import os
import pandas as pd
import swifter
from cleantext import clean
import matplotlib.pyplot as plt
import seaborn as sns
# Hugging Face transformers
from transformers import pipeline
# Group assignment Python modules
import data_analysis
from chatbot_common import *

# Pre-processesed datasets are available here: https://drive.google.com/drive/u/0/folders/1jhZD4BJ2sT-Weq3YCsodItVm8JRWoNmJ


# Set defaults when parallel processing over DataFrame rows is required
swifter.set_defaults(
    npartitions=None,
    dask_threshold=1,
    scheduler="processes",
    progress_bar=True,
    progress_bar_desc=None,
    allow_dask_on_strings=True,
    force_parallel=True,
)


def preprocess(text: str) -> str:
    """ Preprocess text using the Python cleantext library

    Args:
        text (str): Input text to be cleansed

    Returns:
        str: Cleansed text
    """
    return clean(text, fix_unicode=True, to_ascii=True, lower=True, no_urls=False, no_emails=False, no_phone_numbers=True, no_line_breaks=True, no_emoji=True).strip()


# Trained model for determining questions vs. statements
question_or_statement_pipeline = pipeline("text-classification", model="shahrukhx01/question-vs-statement-classifier")


def is_question(text: str) -> bool:
    """ Determine if text is a question or a statement.

    Args:
        text (str): Text to evaluate

    Returns:
        bool: True if text is a question
    """
    return question_or_statement_pipeline(text)[0]["label"] == QUESTION_STATEMENT_CLASSIFIER_QUESTION_LABEL
    # Much quicker yet extremely naive alternative
    #return "?" in text


def load_preprocessed_dataset(dataset_size: DatasetSize) -> pd.DataFrame:
    """ Load the preprocessed dataset (ZIP compressed CSV).
    Regenerate from the Ubuntu Corpus dataset if the preprocessed file does not exist.

    Args:
        dataset_size (DatasetSize): Which size dataset to load

    Returns:
        pd.DataFrame: Pandas DataFrame with columns ["folder", "dialogueID", "date", "from", "to", "text", "id", "is_question"]
    """
    preprocessed_csv_file = os.path.join(DATA_FOLDER, f"{DATASET_FILE_PREFIX}_preprocessed_{dataset_size}_csv.zip")

    if os.path.exists(preprocessed_csv_file):
        return pd.read_csv(preprocessed_csv_file, parse_dates=["date"], compression="zip")

    ubuntu_corpus_dataset_file = os.path.join(UBUNTU_CORPUS_FOLDER, f"dialogueText{UBUNTU_CORPUS_DATASET_FILE_SUFFIXES[dataset_size]}.csv")
    print(f"Processing {dataset_size} Ubuntu Corpus dataset from file '{ubuntu_corpus_dataset_file}'...")

    # Load the dataset making sure the date column is parsed as a date
    df = pd.read_csv(ubuntu_corpus_dataset_file, parse_dates=["date"])

    # Display the min/max folder ids
    print(f"{dataset_size}: Dataset folder range: {df['folder'].min()} .. {df['folder'].max()}")

    # Drop rows where the text is missing
    num_records = len(df)
    print(f"{dataset_size}: Number of records before removing blank text values: {num_records:,}")
    print(f"{dataset_size}: Number of records with null text values: {df['text'].isna().sum():,}")
    df.dropna(subset=["text"], inplace=True)
    # Do some basic preprocessing on the text to remove emails, URLs, convert to ASCII, etc.
    print(f"{dataset_size}: Preprocessing text...")
    df["text"] = df["text"].swifter.allow_dask_on_strings(enable=True).force_parallel(enable=True).apply(preprocess)
    # Remove rows where the text is still blank
    df = df.loc[df["text"].map(lambda s: len(s.strip())) > 0]
    print(f"{dataset_size}: Removed {num_records - len(df):,} blank text values, remaining count: {len(df):,}")
    # Remove rows with single word dialogue
    print(f"Single word text values: {df['text'].loc[df['text'].map(lambda s: ' ' not in s)].head()}")
    df = df.loc[df["text"].map(lambda s: " " in s)]
    print(f"{dataset_size}: Removed {num_records - len(df):,} single word text values, remaining count: {len(df):,}")

    # Dialogue Ids are reused across folders so merge to create a unique id
    print(f"{dataset_size}: Adding unique id column...")
    df["id"] = df[["folder", "dialogueID"]].astype(str).apply(FOLDER_DIALOGUE_ID_SEPARATOR.join, axis=1)

    # Determine if the text is a question or a statement
    print(f"{dataset_size}: Determining questions vs. statements...")
    df["is_question"] = df["text"].swifter.allow_dask_on_strings(enable=True).force_parallel(enable=True).apply(is_question)

    # Make sure dialogues are sorted by date?
    #df.sort_values(["id", "date"], inplace=True)

    # Save the processed data frame to CSV
    df.to_csv(preprocessed_csv_file, index=False, compression="zip")

    return df


def get_dialogue(row: pd.Series) -> tuple[str|None, str|None, bool, str]:
    """ Generate a dialogue tuple containing the following values: from, to, is_question and text

    Args:
        row (pd.Series): Row from the preprocessed data frame

    Returns:
        tuple[str|None, str|None, bool, str]: Tuple of from, to, is_question and text values
    """
    return (None if pd.isna(row["from"]) else row["from"], None if pd.isna(row["to"]) else row["to"], row["is_question"], row["text"].strip())


def merge_sets(s: pd.Series) -> set[str]:
    """ Merge the sets in the first two columns of this row.
    Used to merge the sets of from and to values in the consolidated dialogues dataset.

    Args:
        s (pd.Series): Pandas Series containing two columns containing sets of from and to values

    Returns:
        set[str]: Union of the two sets
    """
    # Return a union of the two sets
    return None if pd.isna(s.iloc[0]) else s.iloc[0] | None if pd.isna(s.iloc[1]) else s.iloc[1]


def get_dialogue_dataset(dataset_size: DatasetSize) -> pd.DataFrame:
    """ Get the consolidated dialogue dataset from the preprocessed Ubuntu Corpus dataset.
    Use persistence to avoid repeating the same processing.

    Args:
        dataset_size (DatasetSize): Ubuntu Corpus dataset size

    Returns:
        pd.DataFrame: Pandas DataFrame containing id, set of from values, set of to values, dialogue start date, and list of dialogue values as generated by get_dialogue()
    """
    dialogue_dataset_pickle_file = os.path.join(DATA_FOLDER, f"{DATASET_FILE_PREFIX}_dialogues_{dataset_size}_pkl.zip")
    
    if os.path.exists(dialogue_dataset_pickle_file):
        return pd.read_pickle(dialogue_dataset_pickle_file, compression="zip")

    # Get the basic preprocessed dataset organised by dialogueID
    df = load_preprocessed_dataset(dataset_size)

    # Combine from, to, is_question and text into a tuple and store in a new dialogue column
    print(f"{dataset_size}: Combining from, to, is_question and text into dialogue column...")
    df["dialogue"] = df.apply(get_dialogue, axis=1)
    print(df.head())

    # Group by the unique id column and aggregate into one combined dialogue value
    dialogues = df.groupby("id").agg({
        "from": set,        # Set of unique values in the 'from' column
        "to": set,          # Set of unique values in the 'to' column
        "date": "min",      # Dialogue start date
        "dialogue": list,   # List of from/to/is_question/text dialogue tuples
    }).reset_index()

    # Rename columns
    dialogues.rename(columns={"from": "unique_from", "to": "unique_to", "date": "start_date"}, inplace=True)
    
    # Populate the set of all users participating in this dialogue
    print(f"{dataset_size}: Populating dialogue unique participants...")
    dialogues["participants"] = dialogues[["unique_from", "unique_to"]].apply(merge_sets, axis=1)
    dialogues["num_participants"] = dialogues["participants"].map(len)
    print(dialogues.head())

    # Store the counts of unique from/to values
    dialogues["unique_from_count"] = dialogues["unique_from"].map(len)
    dialogues["unique_to_count"] = dialogues["unique_to"].map(len)
    # Print count of unique values in the "from" column
    print(f"{dataset_size}: Unique from counts: {dialogues['unique_from_count'].min()} .. {dialogues['unique_from_count'].max()}")
    # Print count of unique values in the "to" column
    print(f"{dataset_size}: Unique to counts: {dialogues['unique_to_count'].min()} .. {dialogues['unique_to_count'].max()}")

    # Remove dialogues with just one particpant - a monologue
    print(f"{dataset_size}: There are {len(dialogues['num_participants'].where(lambda n: n <= 1)):,} monologues")
    num_records = len(dialogues)
    dialogues.drop(dialogues[dialogues["num_participants"] <= 1].index, inplace=True)
    print(f"{dataset_size}: Removed {num_records-len(dialogues):,} monologues, remaining count: {len(dialogues):,}")

    # Calculate and store dialogue length
    # From Kaggle: "The conversations have an average of 8 turns each, with a minimum of 3 turns."
    dialogues["dialogue_len"] = dialogues["dialogue"].apply(len)

    # Drop dialogues of length 1
    num_records = len(dialogues)
    dialogues.drop(dialogues[dialogues["dialogue_len"] <= 1].index, inplace=True)
    print(f"{dataset_size}: Removed {num_records-len(dialogues):,} dialogues with one message, remaining count: {len(dialogues):,}")

    # Sort dialogue threads by start_date and id
    dialogues.sort_values(by=["start_date", "id"], inplace=True)

    dialogues.info()
    print(dialogues.head())
    
    # Now save the dataset so we don't have to do the above every time
    dialogues.to_csv(os.path.join(DATA_FOLDER, f"{DATASET_FILE_PREFIX}_dialogues_{dataset_size}_csv.zip"), index=False, compression="zip")
    dialogues.to_pickle(dialogue_dataset_pickle_file)

    return dialogues


def extract_source_target_pairs(dialogues: pd.DataFrame) -> list[tuple[str, str, str, int, int]]:
    """ Extract merged source-target pairs from the consolidated dialogues dataset.
    Logic is as follows:
    For each conversation thread (sharing the same folder/dialogueID):
    1. For each message in the consolidated dialogue (list of tuples as returned by get_dialogue())
        a. If first message: start new "source" message and set appending_to_source = True
        b. If appending_to_source
            1. If from == previous from
                a. Append message to source
            2. Else
                a. Append message to target and set appending_to_source = False
        c. Else
            1. If end of list or this is a new question or from == thread originating from
                a. Add source-target pair to source_target_pairs and start new source-target pair (set appending_to_source = True)
            2. Else
                a. Append message to target

    Args:
        dialogues (pd.DataFrame): Consolidated dialogues dataset

    Returns:
        list[tuple[str, str, str, int, int]]: List of tuples containing (id, source, target, and numbers of merged source and target messages)
    """
    source_target_pairs = []
    
    for _, row in dialogues.iterrows():
        source = []
        target = []
        # Unpack row data
        id: str = row["id"]
        sub_dialogue_index = 0
        start_date = row["start_date"]
        dialogue: tuple[str|None, str|None, bool, str] = row["dialogue"]
        dialogue_len = row["dialogue_len"]
        last_d_from = None
        dialogue_originator = None
        appending_to_source = True
        for idx, message in enumerate(dialogue):
            d_from, d_to, is_question, text = message
            if not d_from:
                print(f"d_from is None, {id}, {text}")
                d_from = "anon"
            if not dialogue_originator:
                dialogue_originator = d_from
            if appending_to_source:
                if not last_d_from:
                    source.append(text)
                    last_d_from = d_from
                else:
                    # Has message sender changed?
                    if d_from == last_d_from:
                        source.append(text)
                    else:
                        appending_to_source = False
                        target.append(text)
                        last_d_from = d_from
            else:
                # Is this the end of the dialogue
                if idx+1 == len(dialogue):
                    if source and target:
                        source_target_pairs.append((
                            f"{id}/{sub_dialogue_index}",
                            " ".join(source),
                            " ".join(target),
                            len(source),
                            len(target),
                        ))
                else:
                    # Is this the start of a new question or back to the dialogue originator?
                    if is_question or d_from == dialogue_originator:
                        source_target_pairs.append((
                            f"{id}/{sub_dialogue_index}",
                            " ".join(source),
                            " ".join(target),
                            len(source),
                            len(target),
                        ))
                        sub_dialogue_index += 1
                        source = [text]
                        target = []
                        appending_to_source = True
                    else:
                        target.append(text)

    return source_target_pairs


def load_source_target_pair_dataset(dataset_size: DatasetSize, max_concurrent_messages: int = MAX_CONCURRENT_MESSAGES, regenerate: bool = False) -> pd.DataFrame:
    """ Load the source-target pair dataset. Use persistence to avoid repeat processing.

    Args:
        dataset_size (DatasetSize): Ubuntu Corpus dataset size
        max_concurrent_messages (int): Maximum number of question or response messages
        regenerate (bool): Regenerate the dataset rather than load from Pickle

    Returns:
        pd.DataFrame: Pandas DataFrame containing id, merged source and target messages, and numbers of source and target messages that were merged
    """
    source_target_pairs_pickle_file = os.path.join(DATA_FOLDER, f"{DATASET_FILE_PREFIX}_source_target_pairs_{dataset_size}_pkl.zip")
    if not regenerate and os.path.exists(source_target_pairs_pickle_file):
        source_target_pairs = pd.read_pickle(source_target_pairs_pickle_file, compression="zip")
    else:
        dialogues_df = get_dialogue_dataset(dataset_size)

        # Extract "source-target" pairs from the dialogue
        source_target_pairs = pd.DataFrame(extract_source_target_pairs(dialogues_df),
                                           columns=["id", "source", "target", "num_question_messages", "num_response_messages"]).reset_index(drop=True)

        source_target_pairs.to_pickle(source_target_pairs_pickle_file, compression="zip")

    # Limit the number of concurrent messages in a dialogue
    if max_concurrent_messages is not None:
        source_target_pairs = source_target_pairs.loc[(source_target_pairs["num_question_messages"] <= max_concurrent_messages) & (source_target_pairs["num_response_messages"] <= max_concurrent_messages)]

    return source_target_pairs


def main():
    os.makedirs(DATA_FOLDER, exist_ok=True)

    for dataset_size in DatasetSize:
        if DATA_ANALYSIS:
            preprocessed_df = load_preprocessed_dataset(dataset_size)
            preprocessed_df.info()
            print(preprocessed_df.head())
            if PROFILE_DATASET:
                data_analysis.profile_dataset(preprocessed_df)
            # Demonstrate that dialogueID is not unique
            data_analysis.analyse_dialogue_id(preprocessed_df, dataset_size)
            
            dialogues = get_dialogue_dataset(dataset_size)
            # Analyse dialogue lengths
            data_analysis.analyse_dialogue_lengths(dataset_size, dialogues, preprocessed_df)
            #data_analysis.analyse_dialogue_flow(dialogues)
        
        source_target_pairs = load_source_target_pair_dataset(dataset_size)
        print(f"{dataset_size}: Loaded {len(source_target_pairs):,} source-target pairs")
        print(source_target_pairs.head())
        
        print(source_target_pairs["num_question_messages"].describe().T)
        print(source_target_pairs["num_response_messages"].describe().T)

        if MAX_ROWS:
            source_target_pairs = source_target_pairs.iloc[:MAX_ROWS]

        train_data = source_target_pairs
        val_data = source_target_pairs.iloc[:30000].reset_index(drop=True)
        
        # https://www.kaggle.com/code/debanga/auto-hugging-face-to-the-rescue
        #tokeniser = AutoTokenizer.from_pretrained("bert-base-uncased")
        #print(f"{type(tokeniser)}, {inspect.getmro(type(tokeniser))}")
        #dataloaders = DialogueDataLoaders(train_data, val_data, tokeniser, DEFAULT_BATCH_SIZE, DEFAULT_MAX_LEN)
        #print(f"{dataset_size}: {len(dataloaders.train_loader)=:,}, {len(dataloaders.valid_loader)=:,}")


if __name__ == "__main__":
    main()
