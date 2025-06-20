import pandas as pd
from chatbot_common import *
from data_analysis import *
from data_preparation import load_preprocessed_dataset, get_dialogue_dataset, load_source_target_pair_dataset

def main():
    """ Perform low-level statistical data analysis """
    for dataset_size in DatasetSize:
        print()
        print("========================")
        print(f"Dataset size: {dataset_size}")
        print("========================")
        preprocessed_df = load_preprocessed_dataset(dataset_size)
        analyse_dialogue_id(preprocessed_df, dataset_size)
        print("Folder value counts: {}".format(preprocessed_df["folder"].value_counts().describe().T))
        preprocessed_df["to"].dropna().value_counts()

        dialogue_df = get_dialogue_dataset(dataset_size)
        print(dialogue_df.head())
        print(f"Number of dialogues: {len(dialogue_df)}")
        #analyse_dialogue_lengths(dataset_size, dialogue_df, preprocessed_df)
        #analyse_dialogue_flow(dataset_size, dialogue_df)
        print(f"Dialogue length statistics: {dialogue_df.dialogue_len.describe().T}")
        dialogue_len_counts = dialogue_df.dialogue_len.value_counts()
        #print(f"Dialogue length value count statistics: {dialogue_len_counts.describe().T}")
        print(f"Number participants statistics: {dialogue_df.num_participants.describe().T}")
        if SHOW_PLOTS:
            ax = sns.barplot(dialogue_len_counts)
            ax.set_ylabel("Frequency")
            ax.set_xlabel("Dialogue length")
            ax.set_title(f"Frequency distribution of dialogue lengths ({dataset_size} dataset)")
            plt.show()

        print()
        print("+++++++++++++++++++++++++++++")
        print("Unbounded source-target pairs:")
        print("+++++++++++++++++++++++++++++")
        print()
        source_target_pairs_df = load_source_target_pair_dataset(dataset_size, None, False)
        print(source_target_pairs_df.head())
        print(len(source_target_pairs_df))

        # Analysis where there are large numbers of merged source / target messages indicates there a frequent monologues
        print(f"num_question_messages stats: {source_target_pairs_df.num_question_messages.describe().T}")
        print(f"num_response_messages stats: {source_target_pairs_df.num_response_messages.describe().T}")
        if SHOW_PLOTS:
            sns.barplot(source_target_pairs_df["num_question_messages"].value_counts())
            plt.show()
            sns.barplot(source_target_pairs_df["num_response_messages"].value_counts())
            plt.show()
        
        source_target_pairs_df.sort_values(by="num_question_messages", ascending=False, inplace=True)
        df = source_target_pairs_df.loc[source_target_pairs_df.num_question_messages > 1]
        print(f"{dataset_size}: Questions with multiple messages:")
        print(df.head())
        
        source_target_pairs_df.sort_values(by="num_response_messages", ascending=False, inplace=True)
        df = source_target_pairs_df.loc[source_target_pairs_df.num_response_messages > 1]
        print(f"{dataset_size}: Responses with multiple messages:")
        print(df.head())

        print()
        print("++++++++++++++++++++++++++++++++++++++++++")
        print("Limiting source-target pairs to 3 message:")
        print("++++++++++++++++++++++++++++++++++++++++++")
        print()
        source_target_pairs_df = load_source_target_pair_dataset(dataset_size, 3, False)
        print(source_target_pairs_df.head())
        print(f"{len(source_target_pairs_df)=:,}")

        print(f"num_question_messages stats: {source_target_pairs_df.num_question_messages.describe().T}")
        print(f"num_response_messages stats: {source_target_pairs_df.num_response_messages.describe().T}")
        print(f"num_question_messages value count stats: {source_target_pairs_df.num_question_messages.value_counts().describe().T}")
        print(f"num_response_messages value count stats: {source_target_pairs_df.num_response_messages.value_counts().describe().T}")
        if SHOW_PLOTS:
            sns.barplot(source_target_pairs_df["num_question_messages"].value_counts())
            plt.show()
            sns.barplot(source_target_pairs_df["num_response_messages"].value_counts())
            plt.show()
        
        source_target_pairs_df.sort_values(by="num_question_messages", ascending=False, inplace=True)
        df = source_target_pairs_df.loc[source_target_pairs_df.num_question_messages > 1]
        print(f"{dataset_size}: Questions with multiple messages:")
        print(df.head())
        
        source_target_pairs_df.sort_values(by="num_response_messages", ascending=False, inplace=True)
        df = source_target_pairs_df.loc[source_target_pairs_df.num_response_messages > 1]
        print(f"{dataset_size}: Responses with multiple messages:")
        print(df.head())

if __name__ == "__main__":
    main()
