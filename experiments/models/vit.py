from functools import partial

from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from evaluate import load

from ecopt.model import Model
from ecopt.hyperparameter import Hyperparameter, Fixed


def compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric.compute(predictions=preds, references=labels,
                          average="weighted")


class ViTMNISTModel(Model):

    def __init__(self,
                 model_name: Hyperparameter =
                 Fixed("WinKawaks/vit-tiny-patch16-224"),
                 epochs: Hyperparameter = Fixed(3),
                 batch_size: Hyperparameter = Fixed(64)):
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        dataset = load_dataset("mnist")
        image_processor = ViTImageProcessor.from_pretrained(model_name.value)
        # transform MNIST (28x28) to 224x224 RGB
        normalize = Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std)
        transform = Compose([
            Resize((224, 224)),
            lambda x: x.convert("RGB"),
            ToTensor(),
            normalize
        ])

        def transform_example(example):
            example["pixel_values"] = transform(example["image"])
            return example

        # prepare dataset
        self.dataset = dataset.map(transform_example, batched=False)
        self.dataset = self.dataset.rename_column("label", "labels")
        self.dataset.set_format(type="torch", columns=[
            "pixel_values", "labels"])
        self.model = ViTForImageClassification.from_pretrained(
            model_name.value,
            num_labels=10,
            ignore_mismatched_sizes=True
        )

    def train(self):
        training_args = TrainingArguments(
            output_dir="./vit-mnist",
            per_device_train_batch_size=self.batch_size.value,
            per_device_eval_batch_size=self.batch_size.value,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.epochs.value,
            logging_dir="./logs",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",  # disable MLflow and CodeCarbon integrations
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            compute_metrics=partial(compute_metrics, load("f1"))
        )
        self.trainer.train()

    def evaluate(self) -> (float, int):
        metrics = self.trainer.evaluate()
        return metrics["eval_f1"], len(self.dataset["test"])
