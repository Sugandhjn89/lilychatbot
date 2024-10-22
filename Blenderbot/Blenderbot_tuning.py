import torch
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Step 1: Load the conversational data from the text file
def load_conversational_data(file_path):
    conversations = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):  # Assuming each pair of lines is a question and answer
            question = lines[i].strip()
            if i + 1 < len(lines):
                answer = lines[i + 1].strip()
                conversations.append((question, answer))
    return conversations

# Step 2: Write inputs and targets to a file for manual inspection
def write_to_file(conversations, output_path):
    with open(output_path, "w") as f:
        for idx, (input_text, target_text) in enumerate(conversations):
            f.write(f"Index {idx}: Input: {input_text}, Target: {target_text}\n")
    print(f"Written conversations to {output_path} for manual inspection.")

# Step 3: Tokenize data one by one with error handling and max_length
def tokenize_data_one_by_one(tokenizer, conversations):
    max_length = 128  # Define maximum token length to avoid truncation errors
    inputs = []
    targets = []
    for idx, (input_text, target_text) in enumerate(conversations):
        # Skip invalid data
        if not input_text or not target_text:
            print(f"Skipping invalid data at index {idx}: Input: '{input_text}', Target: '{target_text}'")
            continue

        # Tokenize
        try:
            print(f"Tokenizing entry {idx}:")
            print(f"Input: {input_text}\nTarget: {target_text}")
            tokenized_input = tokenizer([input_text], text_target=[target_text], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            inputs.append(tokenized_input["input_ids"].squeeze(0))
            targets.append(tokenized_input["labels"].squeeze(0))
            print(f"Tokenized entry {idx} successfully!")
        except Exception as e:
            print(f"Error tokenizing entry {idx}: {e}")
            break  # Stop after the first error for easier debugging

    return {"input_ids": inputs, "labels": targets}

# Step 4: Prepare the dataset for training
class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

# Step 5: Main function for tokenization and fine-tuning
def main():
    # Load the tokenizer and model
    model_name = "facebook/blenderbot-90M"
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)
    model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_name)

    # Load the conversational data from the text file
    file_path = "C:/Users/MY HP/Desktop/GitHub_Requirements/chatbot_env/Blenderbot/conversational_faq.txt"  # Adjust the file path for your system
    conversations = load_conversational_data(file_path)

    # Write data to a file for manual inspection
    output_path = "C:/Users/MY HP/Desktop/GitHub_Requirements/chatbot_env/Blenderbot/debug_conversations.txt"  # Adjust the file path for your system
    write_to_file(conversations, output_path)

    # Tokenize the data
    model_inputs = tokenize_data_one_by_one(tokenizer, conversations)

    # Step 6: Create a dataset for training
    dataset = ConversationDataset(model_inputs)

    # Step 7: Set up data collator to handle padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Step 8: Set up training arguments and trainer
    training_args = TrainingArguments(
        output_dir='./results',           # output directory for saving models
        num_train_epochs=3,               # number of training epochs
        per_device_train_batch_size=4,    # batch size for each device
        save_steps=500,                   # save model every 500 steps
        save_total_limit=2,               # limit number of saved checkpoints
        logging_dir='./logs',             # directory for logging
    )

    # Step 9: Fine-tuning the model using the Trainer API
    trainer = Trainer(
        model=model,                      # the model to fine-tune
        args=training_args,               # training arguments
        train_dataset=dataset,            # dataset for training
        data_collator=data_collator       # data collator for padding
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_blenderbot')
    tokenizer.save_pretrained('./fine_tuned_blenderbot')

    # --- ADD THIS CODE HERE ---

    # Load the fine-tuned model and tokenizer
    model = BlenderbotSmallForConditionalGeneration.from_pretrained('./fine_tuned_blenderbot')
    tokenizer = BlenderbotSmallTokenizer.from_pretrained('./fine_tuned_blenderbot')

    # Test the model with a new input
    input_text = "You: What is this website about?"
    inputs = tokenizer(input_text, return_tensors='pt')

    # Generate a response from the fine-tuned model
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    print("Bot:", response)

if __name__ == "__main__":
    main()