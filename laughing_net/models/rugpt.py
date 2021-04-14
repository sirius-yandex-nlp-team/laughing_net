import click

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    pipeline, 
)
from datasets import load_dataset

from laughing_net.context import ctx
from laughing_net.config import params

@click.group()
def cli():
    pass

@cli.command()
@click.option("--train-name", type=str)
@click.option("--test-name", type=str)
@click.option("--train-type", type=str, default="text")
@click.option("--test-type", type=str, default="text")
def train(train_name, test_name, train_type, test_type):
    rugpt_params = params.models.rugpt
    train_params = rugpt_params.stages.train
    tokenizer = AutoTokenizer.from_pretrained(rugpt_params.name)
    tokenizer.pad_token = "<pad>"
    model = AutoModelForCausalLM.from_pretrained(rugpt_params.name)
    dataset_dict = load_dataset(
        train_type, 
        data_files={
            "train": str(ctx.data_dir / "processed" / train_name),
            "test": str(ctx.data_dir / "processed" / test_name),
        }
    )
    train_dataset = dataset_dict["train"].map(lambda examples: tokenizer(examples['text']), batched=True)
    test_dataset = dataset_dict["test"].map(lambda examples: tokenizer(examples['text']), batched=True)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    training_args = TrainingArguments(
        output_dir=ctx.root_dir / "checkpoints" / rugpt_params.checkpoint_name,
        overwrite_output_dir=True,
        **train_params,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    trainer.save_model(ctx.root_dir / "artifacts" / rugpt_params.output_name)

@cli.command()
def generate():
    pass

if __name__ == "__main__":
    cli()
