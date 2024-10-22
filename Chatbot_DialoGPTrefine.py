import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Manually set pad_token_id to eos_token_id
model.config.pad_token_id = model.config.eos_token_id

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Streamlit app interface
st.title("Ask from chatbot")
st.write("Please ask any question to the chatbot !")

# User input text box
user_input = st.text_input("You: ", "")

# If there is user input
if user_input:
    # Append user input to chat history
    st.session_state['chat_history'].append(f"You: {user_input}")

    # Prepare the input text for the model: only user input, no full chat history
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Generate response based only on the latest user input
    with torch.no_grad():
        response_ids = model.generate(
            input_ids,
            max_length=150,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

    # Decode the response
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Append bot response to chat history
    st.session_state['chat_history'].append(f"Bot: {response}")

    # Display the chat history
    for chat in st.session_state['chat_history']:
        st.text(chat)

    # Reset user input
    st.text_input("You: ", "", key="new_input")

# Clear the conversation button
if st.button("Clear Conversation"):
    st.session_state['chat_history'] = []
