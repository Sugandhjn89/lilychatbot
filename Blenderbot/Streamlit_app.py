import streamlit as st
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration

# Load the fine-tuned model and tokenizer
model = BlenderbotSmallForConditionalGeneration.from_pretrained('./fine_tuned_blenderbot')
tokenizer = BlenderbotSmallTokenizer.from_pretrained('./fine_tuned_blenderbot')

# Custom CSS for background color and other style adjustments
st.markdown(
    """
    <style>
    /* Set the background color for the main content */
    .stApp {
        background-color: #f0f8ff;
    }
    /* Optional: Style the input and text elements */
    .stTextInput, .stButton {
        background-color: #d1e7ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title and logo
st.title(" Hey Hi! My name is Lily ")
st.image("my_picture.jpg", caption="I am here to help you !", width=150)

st.write("Ask me anything 👉 ")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# User input
user_input = st.text_input("You: ", "")

# Check for "Clear Chat" condition
if user_input.lower() == "clear chat" or user_input.lower() == "reset" or st.button("Clear Chat"):
    st.session_state['chat_history'] = []  # Reset chat history
    st.write("Chat has been cleared. You can ask a new question.")
    st.stop()  # Stop further execution to prevent input from being processed

# If there is user input
if user_input:
    inputs = tokenizer(user_input, return_tensors='pt')
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    # Remove any occurrence of "bot:" at the beginning of the response
    if response.lower().startswith("bot:"):
        response = response[4:].strip()

    # Append the user input and bot response to chat history
    st.session_state['chat_history'].insert(0, f"Bot: {response}")
    st.session_state['chat_history'].insert(0, f"You: {user_input}")

    # Clear the user input field after the message is sent
    st.text_input("You: ", "", key="new_input")

# Display the chat history in reverse order so the latest response is at the top
if st.session_state['chat_history']:
    for chat in st.session_state['chat_history']:
        st.write(chat)
