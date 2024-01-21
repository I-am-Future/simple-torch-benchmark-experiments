from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def train_bert():
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

    # Load dataset
    dataset = load_dataset("imdb", split='train').shuffle().select(range(1000))  # Small subset for benchmarking

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Load model
    config = BertConfig.from_pretrained("bert-large-uncased")
    model = BertForMaskedLM(config=config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./bert_large_uncased_output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_strategy="no",  # Disable checkpoint saving
        logging_steps=10,  # Log every 10 steps
        prediction_loss_only=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    train_bert()
