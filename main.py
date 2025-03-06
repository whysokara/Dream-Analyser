from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def fine_tune_model(text, model_name="distilbert-base-uncased"):
    """Fine-tune a model using the extracted text"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Create a dataset
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

    dataset = TextDataset(inputs)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Create a Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Fine-tune the model
    trainer.train()

# Extract text from PDF and fine-tune the model
pdf_path = "/Users/kara/Desktop/Dream/ExploringDreamIJIP.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
fine_tune_model(pdf_text)

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.4)

chat_history = [
    SystemMessage(content="You are a highly advanced dream interpretation and analysis assistant. Your goal is to listen to the user's voice-recorded descriptions of their dreams, extract key elements, emotions, symbols, and themes, and provide a meaningful and insightful interpretation. Use established dream interpretation theories, psychological insights, and symbolism databases to help the user understand what their dreams might indicate about their emotions, subconscious thoughts, or life experiences.\n\nOver time, you will also identify recurring patterns, symbols, or emotions across multiple dreams and provide a deeper analysis of how these patterns might be connected to the user's mental state, life events, or personal growth journey.\n\nBe supportive, thoughtful, and open-minded — acknowledge that dream interpretation is subjective, and offer possible meanings instead of definitive conclusions. If there are cultural, spiritual, or personal contexts that might influence the meaning, be sure to highlight those.\n\nYour analysis should be clear, easy to understand, and sensitive to the personal nature of dreams. Where possible, offer practical suggestions or reflection questions to help the user better understand themselves through their dreams.")
]
print("Welcome to bot K")

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print(chat_history)