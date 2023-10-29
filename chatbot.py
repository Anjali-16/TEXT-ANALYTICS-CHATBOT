import streamlit as st
import os
import pinecone
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory, ConversationKGMemory, ConversationSummaryBufferMemory, ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

def load_css():
    """load the css to allow for styles.css to affect look and feel of the streamlit interface"""
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_vector_store():
    """Initialize a Pinecone vector store for similarity search."""
    
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'),environment=os.getenv('PINECONE_ENVIRONMENT') )
    index = pinecone.Index(os.getenv('PINECONE_INDEX_NAME')) # you need to have pinecone index name already created (ingest.py should be run first)
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Pinecone(index, embed_model, "text") # 'text' is the field name in pinecone index where original text is stored
                           
    return vectorstore



def initialize_session_state():
    """Initialize the session state variables for streamlit."""
    
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        # create a connection to OpenAI text-generation API
        llm = ChatOpenAI(
            temperature=0.2,
            openai_api_key=os.environ["OPENAI_API_KEY"],
#            max_tokens=500,
            model_name="gpt-4",
            #model_name="gpt-4", 16k model for better strategy.
        )

        memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000000)

        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True,         
        )


@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal['human', 'ai']
    message: str
    
# when submit button is clicked, this function is called    
def on_click_callback(): #called on click of submit button
    """Function to handle the submit button click event."""
    
    with get_openai_callback() as cb:
        
        # get the human prompt in session state (read from text field)
        human_prompt = st.session_state.human_prompt
        #print(human_prompt)
        # conduct a similarity search on our vector database
        vectorstore = initialize_vector_store()
        similar_docs = vectorstore.similarity_search(
            human_prompt,  # our search query
            k=7  # return relevant docs choosing k doesnot effect that.
        )
        #exit()
        
        # create a prompt with the human prompt and the context from the most similar documents
        prompt = f"""
            Your are a chatbot, when someone greets you greets them back. Then based on the question give the appropriate repsonse that is given in knowledge base at the end of questions related to MS BAIS program always mention that "for further inquiries please drop a mail to muma-msbais@usf.edu along with your U-number.".\n\n
            
            Query:\n
            "{human_prompt}" \n\n                        
            
            Context:" \n
            "{' '.join([doc.page_content for doc in similar_docs])}" \n
            """
        print(prompt) # for debugging purposes
        
        # get the llm response from prompt
        llm_response = st.session_state.conversation.run(
            prompt
        )
        
        #store the human prompt and llm response in the history buffer
        st.session_state.history.append(
            Message("human", human_prompt)
        )
        st.session_state.history.append(
            Message("ai", llm_response) 
        )
        
        # keep track of the number of tokens used
        st.session_state.token_count += cb.total_tokens

#############################
# MAIN PROGRAM

# initializations
load_dotenv() # load environment variables from .env file
load_css() # need to load the css to allow for styles.css to affect look and feel
initialize_session_state() # initialize the history buffer that is stored in UI  

# create the Streamlit UI
st.markdown("<img src='https://raw.githubusercontent.com/AkshayRamesh23/Chatbot/main/usf_muma_logo.png' width=250 height=60>", unsafe_allow_html=True)
st.markdown("<strong style='font-size: 30px;'>LangChain based ChatBot 🦜🔗</strong>", unsafe_allow_html=True)
#st.markdown("<strong style='font-size: 45px;'>Pinnacle ChatBot</strong> <img src='https://raw.githubusercontent.com/AkshayRamesh23/Chatbot/main/usf_muma_logo.png' width=250 height=60>", unsafe_allow_html=True)
chat_placeholder = st.container() # container for chat history
prompt_placeholder = st.form("chat-form") # chat-form is the key for this form. This is used to reference this form in other parts of the code
debug_placeholder = st.empty() # container for debugging information

# below is the code that describes how each of the three containers are displayed

with chat_placeholder: # this is the container for the chat history
    for chat in st.session_state.history:
        div = f"""
            <div class="chat-row {'' if chat.origin == 'ai' else 'row-reverse'}">
                <img class="chat-icon" src="{'https://raw.githubusercontent.com/AkshayRamesh23/Chatbot/main/rocky_the_bull.png' if chat.origin == 'ai' else 'https://raw.githubusercontent.com/AkshayRamesh23/Chatbot/main/user_logo.png'}" width=40 height=55>
                <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                    &#8203;{chat.message}
                </div>
            </div>
        """
        st.markdown(div, unsafe_allow_html=True)
    for _ in range(3): # add a few blank lines between chat history and input field
        st.markdown("")

with prompt_placeholder: # this is the container for the chat input field
    col1, col2 = st.columns((6,1)) # col1 is 6 wide, and col2 is 1 wide
    col1.text_input(
        "Chat",
        #value="Please enter your question here",
        value="",  
        placeholder="Please enter your question here",
        label_visibility="collapsed",
        key="human_prompt",  # this is the key, which allows us to read the value of this text field later in the callback function
    )
    col2.form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,  # important! this set's the callback function for the submit button
    )

debug_placeholder.caption(  # display debugging information
    f"""
    Used {st.session_state.token_count} tokens \n
    Debug Langchain.coversation:
    {st.session_state.conversation.memory.buffer}
    """)


