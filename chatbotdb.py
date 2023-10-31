import streamlit as st
import os
import pinecone
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory, ConversationKGMemory, ConversationSummaryBufferMemory, ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dataclasses import dataclass
from langchain.chains import ConversationalRetrievalChain
from typing import Literal
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import sqlite3

def load_css():
    """Load the CSS to allow for styles.css to affect the look and feel of the Streamlit interface."""
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_vector_store():
    """Initialize a Pinecone vector store for similarity search."""
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
    index = pinecone.Index(os.getenv('PINECONE_INDEX_NAME'))
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Pinecone(index, embed_model, "text")
    return vectorstore

def initialize_session_state():
    """Initialize the session state variables for Streamlit."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo-16k",
        )

        # Initialize the SQLite database for chat message history
        db_connection = sqlite3.connect("chat_history.db")
        cursor = db_connection.cursor()
        # Create a table to store chat history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin TEXT,
                message TEXT
            )
        ''')
        db_connection.commit()

        st.session_state.db_connection = db_connection

        memory = ConversationSummaryBufferMemory(llm=llm, return_messages=True)
        st.session_state.conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: str
    message: str

def on_click_callback():
    """Function to handle the submit button click event."""
    with get_openai_callback() as cb:
        # Get the human prompt in session state (read from the text field)
        human_prompt = st.session_state.human_prompt
        
        # Conduct a similarity search on our vector database
        vectorstore = initialize_vector_store()
        similar_docs = vectorstore.similarity_search(
            human_prompt,  # our search query
            k=7  # return relevant docs
        )
        
        # Create a prompt with the human prompt and the context from the most similar documents
        prompt = f"""
            Your are a University of South Florida chatbot, when someone greets you greets them back. Then based on the question give the appropriate detailed response based on the knowledge base. At the end of questions, "only for those questions related to MS BAIS program always mention that for further inquiries please drop a mail to muma-msbais@usf.edu along with your U-number."
            When someone asks a question other than MS BAIS program do not ask them to drop mail to muma-msbais@usf.edu since this mail id is not for other things like housing, employment, immigration related information.
            Always structure your answers in point-wise with appropriate details. Also, when someone asks question about other universities or things unrelated to University of South Florida please tell them that you do not have information about it and this is very important! \n\n
            
            Query:\n
            "{human_prompt}" \n\n                        
            
            Context:" \n
            "{' '.join([doc.page_content for doc in similar_docs])}" \n
            """
        
        # Get the llm response from the prompt
        llm_response = st.session_state.conversation.run(prompt)

        # Create a new SQLite connection and cursor
        db_connection = sqlite3.connect("chat_history.db")
        cursor = db_connection.cursor()
        
        # Store the human prompt and llm response in the SQLite database
        cursor.execute("INSERT INTO chat_history (origin, message) VALUES (?, ?)", ("human", human_prompt))
        cursor.execute("INSERT INTO chat_history (origin, message) VALUES (?, ?)", ("ai", llm_response))

        # Commit the changes and close the cursor and connection
        db_connection.commit()
        cursor.close()
        db_connection.close()
        
        # Store messages in the session history
        st.session_state.history.append(Message("human", human_prompt))
        st.session_state.history.append(Message("ai", llm_response))
        
        # Keep track of the number of tokens used
        st.session_state.token_count += cb.total_tokens

# MAIN PROGRAM
load_dotenv()  # Load environment variables from .env file
load_css()  # Load CSS for styles

initialize_session_state()  # Initialize session state

# Create the Streamlit UI
st.markdown("<img src='https://raw.githubusercontent.com/AkshayRamesh23/Chatbot/main/usf_muma_logo.png' width=250 height=60>", unsafe_allow_html=True)
st.markdown("<strong style='font-size: 30px;'>LangChain based ChatBot 🦜🔗</strong>", unsafe_allow_html=True)

chat_placeholder = st.container()  # Container for chat history
prompt_placeholder = st.form("chat-form")  # Chat form

with chat_placeholder:  # Container for chat history
    for chat in st.session_state.history:
        div = f"""
            <div class="chat-row {'row-reverse' if chat.origin == 'human' else ''}">
                <img class="chat-icon" src="{'https://raw.githubusercontent.com/AkshayRamesh23/Chatbot/main/user_logo.png' if chat.origin == 'human' else 'https://raw.githubusercontent.com/AkshayRamesh23/Chatbot/main/rocky_the_bull.png'}" width=40 height=55>
                <div class="chat-bubble {'human-bubble' if chat.origin == 'human' else 'ai-bubble'}">
                    &#8203;{chat.message}
                </div>
            </div>
        """
        st.markdown(div, unsafe_allow_html=True)
    for _ in range(3):  # Add blank lines between chat history and input field
        st.markdown("")

with prompt_placeholder:  # Container for chat input field
    col1, col2 = st.columns((6, 1))  # col1 is 6 wide, and col2 is 1 wide
    col1.text_input(
        "Chat",
        value="",
        placeholder="Please enter your question here",
        label_visibility="collapsed",
        key="human_prompt",
    )
    col2.form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,
    )
