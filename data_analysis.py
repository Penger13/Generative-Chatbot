import string
import os
import pandas as pd
# Graphing
import matplotlib.pyplot as plt
import seaborn as sns
# Text processing
#from ydata_profiling import ProfileReport

# Group assignment Python modules
from chatbot_common import *


def analyse_dialogue_id(df: pd.DataFrame, dataset_size: DatasetSize) -> None:
    # Demonstrate that dialogueID values are reused across folders in the medium and large datasets
    analysis_df = df.groupby("dialogueID").agg({
        "folder": set,
    }).reset_index()

    reused_dialogue_id_df = analysis_df.loc[analysis_df["folder"].apply(len) > 1]
    if reused_dialogue_id_df.empty:
        print(f"{dataset_size}: There are NO dialogueID values reused across folders")
    else:
        print(f"{dataset_size}: There are {len(reused_dialogue_id_df):,d} dialogueID values reused across folders")
        reused_dialogue_id_df.info()
        print(f"{dataset_size}: dialogueID values that are reused across folders:")
        print(reused_dialogue_id_df.head())


def analyse_dialogue_lengths(dataset_size: str, dialogues: pd.DataFrame, preprocessed_df: pd.DataFrame) -> None:
    print(f"{dataset_size}: Dialogue length statistics:")
    print(dialogues["dialogue_len"].describe().T)
    for i in range(5):
        len_df = dialogues.loc[dialogues["dialogue_len"] == i]
        print(f"{dataset_size}: Number of dialogues of length {i}: {len(len_df):,}")
        if not len_df.empty:
            print(len_df)
    dialogue_len_1_df = dialogues.loc[dialogues["dialogue_len"] == 1]
    if not dialogue_len_1_df.empty:
        print(f"{dataset_size}: First dialogue of length 1: {dialogue_len_1_df["id"].iloc[0]}")
        dialogue_id = dialogue_len_1_df["id"].iloc[0].split(FOLDER_DIALOGUE_ID_SEPARATOR)[1]
        print(preprocessed_df.loc[preprocessed_df["dialogueID"] == dialogue_id])


def profile_dataset(dataset_size: DatasetSize, df: pd.DataFrame) -> None:
    pass
    # Profile the corpus dataset
    #profile = ProfileReport(df, title=f"YData Profiling Report for Ubuntu Dialogue Corpus Dataset ({dataset_size})")
    #profile.to_file(os.path.join(DATA_FOLDER, f"{DATASET_FILE_PREFIX}_report_{dataset_size}.html"))


def get_dialogue_flow(dialogue: list[tuple[str|None, str|None, bool, str]]) -> str:
    # Mapping from participant name to participant code
    participants = {}
    dialogue_flow = []
    codes = list(string.ascii_uppercase)

    for message in dialogue:
        message_from = message[0]
        message_to = message[1]

        if not message_from:
            code_from = "Anon"
        elif message_from in participants:
            code_from = participants[message_from]
        else:
            code_from = codes.pop(0)
            participants.setdefault(message_from, code_from)

        if not message_to:
            code_to = "Anon"
        elif message_to in participants:
            code_to = participants[message_to]
        else:
            code_to = codes.pop(0)
            participants.setdefault(message_to, code_to)

        dialogue_flow.append(f"{code_from}-{code_to}")

    return ",".join(dialogue_flow)


def analyse_dialogue_flow(dataset_size: DatasetSize, dialogues: pd.DataFrame) -> None:
    # Extract unique dialogue flow patterns
    # FIXME Update this so that it doesn't modify the input data frame
    dialogues["dialogue_flow"] = dialogues["dialogue"].apply(get_dialogue_flow)
    print(dialogues.head(20))
    print(dialogues["dialogue_flow"].unique())
    
    print(f"{dataset_size}: Sorted by dialogue length:")
    print(dialogues.sort_values(by="dialogue_len", ascending=False).head(10))
    
    dialogue_len_counts = dialogues["dialogue_len"].value_counts()
    print(dialogue_len_counts)
    if SHOW_PLOTS:
        ax = sns.barplot(dialogue_len_counts)
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Dialogue length")
        ax.set_title(f"Frequency distribution of dialogue lengths ({dataset_size} dataset)")
        plt.show()

    dialogue_flow_value_counts = dialogues["dialogue_flow"].value_counts()
    print(dialogue_flow_value_counts)
    if SHOW_PLOTS:
        ax = sns.barplot(dialogue_flow_value_counts)
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Dialogue flow")
        # tick_params method doesn't support horizontalalignment parameter
        # which is required when rotating as default is centered
        #ax.tick_params(axis='x', labelrotation=30)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_title(f"Frequency distribution of dialogue flow ({dataset_size} dataset)")
        plt.subplots_adjust(bottom=0.32, left=0.19)
        plt.show()
    
    # For the smallest dataset
    explore_dialogue_flow = [
        "A-Anon,A-Anon,B-A",    # 44%    Normal  - request-request-response; join first 2
        "A-Anon,B-A,B-A",       # 24%    Normal  - request-response-response; join last 2
        "A-Anon,B-A,A-B",       # 22%    Normal  - request-response-ack; ignore last one
        "A-B,B-A,A-B",          #  4%    Unsure  - message-response-message - direct conversation between A & B
        "A-B,A-B,A-B",          #  1.6%  Ignore? - message-message-message - one-way conversation
        "A-B,A-B,B-A",          #  1.2%  Check   - message-message-response
        "A-B,B-A,B-A",          #  1.2%  Check   - message-response-response
        "A-B,A-Anon,B-A",       #  0.6%  Check   - message-request-response
        "A-B,B-C,A-B",          #  0.3%  Check   - message-onward_message-response
        "A-A,A-A,A-A",          #  0.03% Ignore? - monologue
    ]

    # Autodetect terminal width
    # , 'display.max_colwidth', 200
    with pd.option_context('display.width', 200):
        for dialogue_flow in explore_dialogue_flow:
            print()
            print(dialogue_flow)
            #print(dialogues.loc[dialogues["dialogue_flow"] == dialogue_flow, "dialogue"].apply(lambda l: ", ".join([x[3] for x in l])).head(10).to_string(index=False))
            for _, row in dialogues.loc[dialogues["dialogue_flow"] == dialogue_flow][:10].iterrows():
                for message in row["dialogue"]:
                    print(f"{row['id']}: {message[0]}-{message[1]}: {message[3]}")
