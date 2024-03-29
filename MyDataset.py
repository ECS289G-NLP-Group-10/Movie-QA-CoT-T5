from collections import Counter
from typing import List, Tuple
import datasets
import transformers
import torch
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import re


class DatasetMap():
    @staticmethod
    def duorc(example):
        nr_answer = len(example["answers"])
        return [example["plot"]]*nr_answer, [example["question"]]*nr_answer, [answer if len(answer) > 0 else "" for answer in example["answers"]]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset: datasets.arrow_dataset.Dataset, tokenizer, parser=None):
        """Constructor for Dataset class
        Args:
            hf_dataset (datasets.arrow_dataset.Dataset): HuggingFace Dataset
            tokenizer: HuggingFace Tokenizer

        Raises:
            Exception: if two between questions, answers and contexts have different length it will raise an exception
        """
        self.tokenizer = tokenizer
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.contexts: List[str] = []

        for row in tqdm(hf_dataset):
            _contexts, _questions, _answers = parser(row)

            self.contexts += _contexts
            self.questions += _questions
            self.answers += _answers

        if len(self.questions) != len(self.answers) or len(self.questions) != len(self.contexts):
            raise Exception(
                "something wrong while building the dataset: questions, contexts and answers result in different dimensions")

        self.item_count: int = len(self.questions)

    def __len__(self):
        """Magic method over-ride for class lenght evaluation

        Returns:
            int: lenght of the object 
        """
        return self.item_count

    def __getitem__(self, index: int):
        """Magic method over-ride for class getitem method

        Args:
            index (int): index for identify question-context and answer example

        Returns:
            Tuple(str,str,str): (Context, Question, Answer)
        """
        return self.contexts[index], self.questions[index], self.answers[index]

    def pack_minibatch(self, data: List[Tuple[str, str]]):
        """Pack mini-batch function

        Args:
            data (Tuple[List[str],List[str],List[str]]): (Contexts, Questions, Answers)

        Returns:
            Tuple[List[str],List[str],List[str]]: (Contexts, Questions, Answers)
        """
        return zip(*data)

    def __exact_match_score(self, prediction, ground_truth):
        """_summary_

        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(ground_truth) == len(prediction):
            if all(token1 == token2 for token1, token2 in zip(ground_truth, prediction)):
                return 1
        return 0

    def __f1_score(self, prediction_tokens, ground_truth_tokens):
        """_summary_

        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_

        Returns:
            _type_: _description_
        """
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def __rouge_score(self, hyps, refs):
        """_summary_

        Args:
            prediction (_type_): _description_
            ground_truth (_type_): _description_

        Returns:
            _type_: _description_
        """
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)
        return scores

    def __bleu_score(self, references, predictions):
        """
        Compute BLEU score.

        Args:
            references (List[str]): List of reference sentences.
            predictions (List[str]): List of predicted sentences.

        Returns:
            bleu_score (float): BLEU score.
        """
        references_split = [reference.split() for reference in references]
        list_of_references = [[references]for references in references_split]
        predictions_split = [prediction.split() for prediction in predictions]
        return corpus_bleu(list_of_references, predictions_split, weights=(0.5, 0.5))

    def evaluate(self, predictions, gold_answers):
        """_summary_

        Args:
            predictions (_type_): _description_
            gold_answers (_type_): _description_

        Returns:
            _type_: _description_
        """
        f1 = exact_match = 0
        hypotheses = []
        references = []
        for ground_truths, prediction in tqdm(zip(gold_answers, predictions)):
            # Remove pad token
            tokens_to_remove = {
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.mask_token_id
            }
            prediction = list(
                filter(lambda token: token not in tokens_to_remove, prediction))
            ground_truths = list(
                filter(lambda token: token not in tokens_to_remove, ground_truths))
            # Convert token IDs to tokens
            hypothesis = self.tokenizer.decode(
                prediction, skip_special_tokens=True)
            reference = self.tokenizer.decode(
                ground_truths, skip_special_tokens=True)

            pattern = r'-->(.*?)END'
            h_extracted_text = ""
            r_extracted_text = ""
            h_matches = re.findall(pattern, hypothesis, re.DOTALL)
            r_matches = re.findall(pattern, reference, re.DOTALL)
            if h_matches:
                h_extracted_text = h_matches[0].strip()
                print(h_extracted_text)
            else:
                h_extracted_text = hypothesis
                print("No h match found.")
            if r_matches:
                r_extracted_text = r_matches[0].strip()
                print(r_extracted_text)
            else:
                r_extracted_text = reference
                print("No r match found.")
            hypotheses.append(h_extracted_text)
            references.append(r_extracted_text)
            # Convert tokens to IDs
            extracted_hypothesis_ids = self.tokenizer.encode(
                h_extracted_text, add_special_tokens=False)
            extracted__reference_ids = self.tokenizer.encode(
                r_extracted_text, add_special_tokens=False)
            f1 += self.__f1_score(extracted_hypothesis_ids,
                                  extracted__reference_ids)
            exact_match += self.__exact_match_score(
                extracted_hypothesis_ids, extracted__reference_ids)
            # hypotheses.append(hypothesis)
            # references.append(reference)
            print(f"hypthesis: {hypothesis}")
            print("\n")
            print(f"reference: {reference}")
            print("\n")
            # f1 += self.__f1_score(prediction, ground_truths)
            # exact_match += self.__exact_match_score(prediction, ground_truths)

        # Compute BLEU score
        bleu = self.__bleu_score(references, hypotheses)
        # Compute ROUGE
        rouge_scores = self.__rouge_score(hypotheses, references)
        return 100*f1/len(predictions), 100*exact_match/len(predictions), bleu, rouge_scores
