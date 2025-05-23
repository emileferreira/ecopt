from functools import partial
from math import ceil

from tqdm import tqdm
from ecopt.model import Model
from ecopt.hyperparameter import Hyperparameter, Fixed
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset


def dataset_generator(dataset_name, num_inferences, max_prompt_length):
    dataset = load_dataset(dataset_name, trust_remote_code=True,
                           streaming=True, split="train")
    for row in dataset.take(num_inferences):
        words = row['text'].split()
        row['text'] = ' '.join(words[:max_prompt_length])
        yield row


def batch(iterable, n=1):
    """Return batches of an iterable."""
    length = len(iterable)
    for i in range(0, length, n):
        yield iterable[i:min(i + n, length)]


class TextGenerationModel(Model):

    def __init__(self,
                 model_name: Hyperparameter = Fixed("google/gemma-3-1b-it"),
                 dataset_name: Hyperparameter = Fixed("bookcorpus"),
                 max_new_tokens: Hyperparameter = Fixed(10),
                 batch_size: Hyperparameter = Fixed(1),
                 num_inferences: Hyperparameter = Fixed(1000),
                 max_prompt_length: Hyperparameter = Fixed(20),
                 do_sample: Hyperparameter = Fixed(False)):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.num_inferences = num_inferences
        self.max_prompt_length = max_prompt_length
        self.do_sample = do_sample

    def define(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name.value, torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name.value)
        if "gpt2" in self.model_name.value:
            self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.dataset = Dataset.from_generator(
            partial(dataset_generator,
                    self.dataset_name.value,
                    self.num_inferences.value,
                    self.max_prompt_length.value)
        )
        self.text_gen = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            device_map="auto",  # use GPU(s) if available
        )

    def evaluate(self):
        num_tokens = 0

        def count_tokens(text: str) -> int:
            return len(self.tokenizer.encode(text, add_special_tokens=False))

        num_batches = ceil(
            self.num_inferences.value / self.batch_size.value)
        for chunk in tqdm(batch(self.dataset, self.batch_size.value),
                          unit="batch", desc="Generate", total=num_batches):
            prompts = chunk["text"]
            outputs = self.text_gen(prompts,
                                    batch_size=self.batch_size.value,
                                    max_new_tokens=self.max_new_tokens.value,
                                    do_sample=self.do_sample.value)
            for prompt, output in zip(prompts, outputs):
                generated = output[0]["generated_text"]
                num_tokens += count_tokens(generated) - count_tokens(prompt)
        return 1, num_tokens
