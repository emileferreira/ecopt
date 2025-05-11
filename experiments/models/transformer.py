from ecopt.model import Model
from ecopt.hyperparameter import Hyperparameter, Fixed
from transformers import pipeline, AutoTokenizer


class TextGenerationModel(Model):

    def __init__(self,
                 model_name: Hyperparameter = Fixed("google/gemma-3-1b-it"),
                 prompt: Hyperparameter = Fixed(
                     """Write a story about the meaning of life."""),
                 max_new_tokens: Hyperparameter = Fixed(1000),
                 do_sample: Hyperparameter = Fixed(False)):
        self.model_name = model_name
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

    def define(self):
        self.generator = pipeline(
            "text-generation",
            model=self.model_name.value,
            device_map="auto",  # use GPU if available
            model_kwargs={"torch_dtype": "auto"}  # use FP16 if GPU supports it
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name.value)

    def evaluate(self):
        prompt = self.prompt.value
        output = self.generator(prompt,
                                max_new_tokens=self.max_new_tokens.value,
                                do_sample=self.do_sample.value)
        generated_text = output[0]["generated_text"]

        def count_tokens(text: str) -> int:
            return len(self.tokenizer.encode(text, add_special_tokens=False))

        num_tokens = count_tokens(generated_text) - count_tokens(prompt)
        return 0, num_tokens
