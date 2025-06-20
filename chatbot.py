from transformers import pipeline
from chatbot_common import *

# Pretrained models are available here: https://drive.google.com/drive/u/0/folders/1-vnMaiuIMTjABXTmIDhfHW8yGE_3e8Mh

def main():
    """ Simple command line chatbot. """

    #trained_model_name = ubuntu_dialogue_qa_model
    trained_model_name = "dialogues_seq2seq_model_small"
    # Note the T5 model requires specific prompts to prevent it assuming translation from English to German
    #trained_model_name = "dialogues_t5_qa_model_small"

    # Load the pre-trained model    
    question_answerer = pipeline("translation", model=trained_model_name)

    print("Welcome to the Ubuntu Dialogue Corpus chatbot, how can I help? (type 'exit' or CTRL-C to exit)")
    while True:
        print("> ", end="", flush=True)
        question = input().strip()
        if question == "quit" or question == "exit":
            break
        print(f"Response: {question_answerer(question)[0]['translation_text']}")

if __name__ == "__main__":
    main()
