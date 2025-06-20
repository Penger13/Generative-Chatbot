from typing import Dict, List, Optional
import time
import math
import numpy as np
import os
# Hugging Face transformers
from datasets import load_dataset, Dataset, VerificationMode
from transformers import pipeline, is_torch_xla_available
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers.trainer_utils import speed_metrics
from transformers import PreTrainedTokenizerBase, BatchEncoding
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
import evaluate
import sacrebleu
# Group assignment Python modules
from chatbot_common import *
from data_preparation import load_source_target_pair_dataset


# Pretrained models are available here: https://drive.google.com/drive/u/0/folders/1-vnMaiuIMTjABXTmIDhfHW8yGE_3e8Mh


TRAIN_SOURCE_TARGET_PAIRS = False
UBUNTU_DIALOGUE_QA_MODEL_NAME = "ubuntu_dialogue_qa_model"
max_train_samples = None
max_eval_samples = 5000
EVALUATE = True
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
RETRAIN = False
RESUME_TRAINING_FROM_CHECKPOINT = False

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


class QuestionAnsweringSeq2SeqTrainer(Seq2SeqTrainer):
    # Taken from https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/trainer_seq2seq_qa.py#L34
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    # def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if gen_kwargs.get("max_length") is None and self.args.generation_max_length is not None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        self._gen_kwargs = gen_kwargs

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        if self.post_process_function is not None and self.compute_metrics is not None and self.args.should_save:
            # Only the main node write the results by default
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            metrics.update(output.metrics)
        else:
            metrics = output.metrics

        if self.args.should_log:
            # Only the main node log the results by default
            self.log(metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


def compute_metrics(tokeniser: PreTrainedTokenizerBase, eval_preds) -> dict:
    """ Compute performance metrics (BLEU) """
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokeniser.pad_token_id)
    decoded_preds = tokeniser.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokeniser.pad_token_id)
    decoded_labels = tokeniser.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


def preprocess_dialogue(dialogues: Dataset, source_column: str, target_column: str, tokeniser: PreTrainedTokenizerBase) -> BatchEncoding:
    """ Pre-process dialogue text for training

    Args:
        dialogues (Dataset): The dialogues dataset containing source-target pairs
        source_column (str): The name of the source column in the dataset
        target_column (str): The name of the target column in the dataset
        tokeniser (PreTrainedTokenizerBase): A pre-trained tokeniser for encoding source and target values

    Returns:
        (BatchEncoding) Encoded model inputs
    """
    inputs = [q.strip() for q in dialogues[source_column]]
    targets = [r.strip() for r in dialogues[target_column]]
    return tokeniser(inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True)


def train_ubuntu_dialogue_qa_seq2seq(dialogues_ds: Dataset, source_column: str, target_column: str,
                             base_model_name: str, trained_model_name: str, max_eval_samples: int, max_train_samples: int) -> None:
    """ Train a Seq2Seq model on the provided dialogue dataset.
    Can be used for both our data as well as the publicly available INSTRUCTION/RESPONSE dataset available here: https://huggingface.co/datasets/sedthh/ubuntu_dialogue_qa
    Based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_seq2seq_qa.py

    Args:
        dialogues_ds (Dataset): The dialogues dataset containing source-target pairs
        source_column (str): The name of the source column in the dataset
        target_column (str): The name of the target column in the dataset
        base_model_name (str): Name of the base pre-trained model that we will be extending
        trained_model_name (str): Name of the model we will be training
        max_eval_samples (int): Maximum number of samples for evaluation (set to None for all "test" records)
        max_train_samples (int): Maximum number of samples for training (set to None for all "train" records)
    """
    print(f"train_ubuntu_dialogue_qa_seq2seq: {trained_model_name=}, {max_eval_samples=}, {max_train_samples=}")

    # Load the pretrained tokeniser and base model
    tokeniser = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    
    print(f"{trained_model_name}: {dialogues_ds['train'][0]=}")
    
    # Encode (tokenise) the source and target values using the preprocess_dialogue method
    tokenised_dialogues = dialogues_ds.map(lambda x: preprocess_dialogue(x, source_column, target_column, tokeniser),
                                           batched=True, remove_columns=dialogues_ds["train"].column_names)
    print(f"{trained_model_name}: {tokenised_dialogues=}")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokeniser)

    # Specify Seq2Seq training parameters
    training_args = Seq2SeqTrainingArguments(
        output_dir=trained_model_name,
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=3,
        predict_with_generate=True,
        fp16=True, # Set fp16=True modern GPUs, bf16=True for XPU
        push_to_hub=False,
    )
    
    # Prepare the model trainer
    eval_dataset = tokenised_dialogues["test"]
    print(f"{trained_model_name}: {len(eval_dataset)=}")
    trainer = QuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dialogues["train"],
        eval_dataset=eval_dataset,
        processing_class=tokeniser,
        data_collator=data_collator,
        compute_metrics=lambda preds: compute_metrics(tokeniser, preds),
    )

    # Optionally evaluate
    if EVALUATE:
        print(f"{trained_model_name}: Evaluating...")
        metrics = trainer.evaluate(max_length=MAX_LENGTH)
        if max_eval_samples is None:
            max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Train the model
    print(f"{trained_model_name}: Training...")
    train_result = trainer.train(resume_from_checkpoint=RESUME_TRAINING_FROM_CHECKPOINT)
    print(f"{trained_model_name}: Saving...")
    trainer.save_model(trained_model_name)
    print(f"{trained_model_name}: Done.")

    # Generate and save metrics
    metrics = train_result.metrics
    if max_train_samples is None:
        max_train_samples = len(dialogues_ds)
    metrics["train_samples"] = min(max_train_samples, len(dialogues_ds))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def train_source_target_pairs_t5(dialogues_ds: Dataset, source_column: str, target_column: str,
                                 t5_model_name: str, trained_model_name: str, max_eval_samples: int, max_train_samples: int) -> None:
    """ Train a T5 model on the provided dialogue dataset.
    This is included for testing purposes only - Seq2Seq is more appropriate for this exercse.
    Can be used for both our data as well as the publicly available INSTRUCTION/RESPONSE dataset available here: https://huggingface.co/datasets/sedthh/ubuntu_dialogue_qa
    Based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_seq2seq_qa.py

    Args:
        dialogues_ds (Dataset): The dialogues dataset containing source-target pairs
        source_column (str): The name of the source column in the dataset
        target_column (str): The name of the target column in the dataset
        base_model_name (str): Name of the base pre-trained model that we will be extending
        trained_model_name (str): Name of the model we will be training
        max_eval_samples (int): Maximum number of samples for evaluation (set to None for all "test" records)
        max_train_samples (int): Maximum number of samples for training (set to None for all "train" records)
    """
    print(f"train_source_target_pairs_t5: {trained_model_name=}, {max_eval_samples=}, {max_train_samples=}")

    # Load the pretrained tokeniser and base model
    tokeniser = AutoTokenizer.from_pretrained(t5_model_name)
    # FIXME Not sure why this didn't work...
    #model = T5ForQuestionAnswering.from_pretrained(t5_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)

    print(f"{trained_model_name}: {dialogues_ds['train'][0]=}")
    
    # Encode (tokenise) the source and target values using the preprocess_dialogue method
    tokenised_dialogues = dialogues_ds.map(lambda x: preprocess_dialogue(x, source_column, target_column, tokeniser),
                                           batched=True, remove_columns=dialogues_ds["train"].column_names)
    print(f"{trained_model_name}: {tokenised_dialogues=}")

    #data_collator = DataCollatorForSeq2Seq(tokenizer=tokeniser, model=model)
    data_collator = DataCollatorForSeq2Seq(tokeniser)
    
    # Specify training parameters
    training_args = Seq2SeqTrainingArguments(
        output_dir=trained_model_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=3,
        predict_with_generate=True,
        fp16=True, # Set fp16=True modern GPUs, bf16=True for XPU
        push_to_hub=False,
    )
    
    # Prepare the model trainer
    eval_dataset = tokenised_dialogues["test"]
    print(f"{trained_model_name}: {len(eval_dataset)=}")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dialogues["train"],
        eval_dataset=eval_dataset,
        processing_class=tokeniser,
        data_collator=data_collator,
        compute_metrics=lambda preds: compute_metrics(tokeniser, preds),
    )

    # Optionally evaluate
    if EVALUATE:
        print(f"{trained_model_name}: Evaluating...")
        metrics = trainer.evaluate(max_length=MAX_LENGTH)
        if max_eval_samples is None:
            max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Train the model
    print(f"{trained_model_name}: Training...")
    train_result = trainer.train(resume_from_checkpoint=RESUME_TRAINING_FROM_CHECKPOINT)
    print(f"{trained_model_name}: Saving...")
    trainer.save_model(trained_model_name)
    print(f"{trained_model_name}: Done.")

    # Generate and save metrics
    metrics = train_result.metrics
    if max_train_samples is None:
        max_train_samples = len(dialogues_ds)
    metrics["train_samples"] = min(max_train_samples, len(dialogues_ds))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def generate_answer(tokeniser: PreTrainedTokenizerBase, model, question: str) -> str:
    """ Use the pre-trained model to generate an answer to the provided question.

    Args:
        tokeniser (PreTrainedTokenizerBase): Tokeniser
        model (_type_): The pre-trained model
        question (str): Question to ask

    Returns:
        str: Response generated by the model
    """
    tokenised_question = tokeniser(question, return_tensors="pt")
    outputs = model.generate(**tokenised_question)
    return tokeniser.decode(outputs[0], skip_special_tokens=True)


def generate_metrics(test_data: Dataset, source_column: str, target_column: str, tokeniser: PreTrainedTokenizerBase, model) -> None:
    """ Display BLEU / ROUGE metrics for the test slice of the dataset

    Args:
        test_data (Dataset): Test slice of the dataset
        source_column (str): Name of the source column, e.g. "INSTRUCTION" or "question"
        target_column (str): Name of the target column, e.g. "RESPONSE" or "answer"
        tokeniser (PreTrainedTokeniser): Pre-trained tokeniser
        model (_type_): Pre-trained model
    """
    # Prepare question and reference data suitable for metrics APIs
    questions = [instruction for instruction in test_data[source_column]]
    references = [[response] for response in test_data[target_column]]
    #print(f"{questions=}")
    #print(f"{references=}")

    # Generate predicted responses for all quesionts
    print(f"Generating predictions for {len(questions):,} questions...")
    predictions = [generate_answer(tokeniser, model, question) for question in questions]
    #print(f"{predictions=}")

    print(f"BLEU score: {bleu.compute(predictions=predictions, references=references)}")
    print(f"ROUGE score: {rouge.compute(predictions=predictions, references=references)}")

    # Alternative approach using sacrebleu APIs directly...
    sacrebleu_references = [ref[0] for ref in references]
    sacrebleu_score = sacrebleu.corpus_bleu(predictions, [sacrebleu_references])
    print(f"BLEU Score (sacrebleu direct): {sacrebleu_score.score:.2f}")


def main():
    num_train_records_for_metrics = 1000

    # Load and convert publicly available processed Ubuntu Dialogue Corpus Question-Answer pairs
    ubuntu_dialogue_qa_ds = load_dataset("sedthh/ubuntu_dialogue_qa", verification_mode=VerificationMode.NO_CHECKS, split="train").train_test_split(test_size=0.2)
    # Load and convert raw Ubuntu Dialogue Corpus data into a Hugging Face Dataset
    dataset_size = DatasetSize.SMALL
    source_target_pairs = load_source_target_pair_dataset(dataset_size)
    print(f"{len(source_target_pairs)=}")
    print(source_target_pairs.head())
    dialogues_ds = Dataset.from_pandas(source_target_pairs)
    print(f"{dialogues_ds[0]=}")
    dialogues_ds = dialogues_ds.train_test_split(test_size=0.2)
    print(f"{dialogues_ds['train'][0]=}")

    training_sets = [
        ("facebook/bart-base", UBUNTU_DIALOGUE_QA_MODEL_NAME, ubuntu_dialogue_qa_ds, "INSTRUCTION", "RESPONSE", train_ubuntu_dialogue_qa_seq2seq),
        ("facebook/bart-base", f"dialogues_seq2seq_model_{dataset_size}", dialogues_ds, "source", "target", train_ubuntu_dialogue_qa_seq2seq),
        ("google-t5/t5-small", f"dialogues_t5_model_{dataset_size}", dialogues_ds, "source", "target", train_source_target_pairs_t5),
    ]
    for base_model_name, trained_model_name, ds, source_column, target_column, train_func in training_sets:
        print(f"Using model {trained_model_name}")
        if RETRAIN or not os.path.exists(os.path.join(trained_model_name, "config.json")):
            train_func(ds, source_column, target_column, base_model_name,
                       trained_model_name, max_eval_samples, max_train_samples)
        model = AutoModelForSeq2SeqLM.from_pretrained(trained_model_name)
        print(f"Loaded model '{trained_model_name}'")
        tokeniser = AutoTokenizer.from_pretrained(trained_model_name)
        question_answerer = pipeline("translation", model=trained_model_name)
        for question in TEST_QUESTIONS:
            print(f"{trained_model_name}: Question: {question=} ... Response: {question_answerer(question)[0]['translation_text']}")
        predictions = [generate_answer(tokeniser, model, qa[0]) for qa in TEST_QUESTION_ANSWERS]
        references = [qa[1] for qa in TEST_QUESTION_ANSWERS]
        sacrebleu_score = sacrebleu.corpus_bleu(predictions, [references])
        print(f"{trained_model_name}: BLEU Score for our TEST_QUESTION_ANSWERS (sacrebleu): {sacrebleu_score.score:.2f}")
        print(f"{trained_model_name}: BLEU Score for our TEST_QUESTION_ANSWERS (bleu): {bleu.compute(predictions=predictions, references=references)}")

        print(f"{trained_model_name}: Generating metrics for {num_train_records_for_metrics:,} records in train dataset")
        generate_metrics(ds["train"][:num_train_records_for_metrics], source_column, target_column, tokeniser, model)
        print(f"{trained_model_name}: Generating metrics for test dataset - {len(ds['test']):,} records")
        generate_metrics(ds["test"], source_column, target_column, tokeniser, model)
    

if __name__ == "__main__":
    main()
