# Step 1: Setup your project structure:
# ├── .env                   <-- Your environment file containing OPENAI_API_KEY=YOUR_API_KEY
# ├── forensics.db           <-- Your SQLite database file
# ├── app.py                 <-- Your Streamlit application (this file)
# └── requirements.txt       <-- List of Python dependencies (if needed)

# Step 2: Import Libraries
import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
import streamlit as st

# Step 3: Load Environment Variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")
os.environ["OPENAI_API_KEY"] = api_key

# Step 4: Initialize the SQLite Database
db_path = "forensics.db"  # Path to your SQLite database
engine = create_engine(f"sqlite:///{db_path}")
db = SQLDatabase(engine=engine)

# Step 5: Set Up the LLM Agent
#    - Replace "gpt-4o" with your chosen model if needed.
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Step 6: Create a Data Dictionary for Context
data_dictionary = """
### Database Structure and Tables

The `forensics.db` database contains three tables (sheets):

1. **investigation**
    - Latitude
    - Longitude
    - Date
    - Location
    - Drug Type

2. **frequency**
    - Drug Type
    - 2022 Q1
    - 2022 Q2
    - 2022 Q3
    - 2022 Q4
    - 2023 Q1
    - 2023 Q2
    - 2023 Q3
    - 2023 Q4
    - 2024 Q1
    - 2024 Q2
    - 2024 Q3
    - 2024 Q4

3. **timeline**
    - Drug Type
    - 2022 Q1
    - 2022 Q2
    - 2022 Q3
    - 2022 Q4
    - 2023 Q1
    - 2023 Q2
    - 2023 Q3
    - 2023 Q4
    - 2024 Q1
    - 2024 Q2
    - 2024 Q3
    - 2024 Q4

There is a common column **Drug Type** that could be used to join or compare data among these tables if needed. 

### Example Queries You Can Ask:
- "Show me all drug types and their coordinates from the investigation table."
- "Which quarter has the highest frequency for a specific drug type?"
- "How does the frequency change over the timeline for a particular drug type?"
"""

# Step 7: Build the Streamlit UI
st.title("Forensic Digital Assistant")
st.write("Ask me anything about the forensics data")

# Initialize conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Input field for user query
user_input = st.text_input("You:", key="user_input")

if user_input:
    # Combine data dictionary with user query for better context
    query = f"{data_dictionary}\n\n{user_input}"

    try:
        # Use the LLM agent to handle the query
        result = agent_executor.invoke({"input": query})["output"]
        # Store both user query and assistant response
        st.session_state.conversation.append(("User", user_input))
        st.session_state.conversation.append(("Assistant", result))
    except Exception as e:
        st.session_state.conversation.append(("Assistant", f"Error: {str(e)}"))

    # Clear the input field after submission
    user_input = ""

# -----------------------------------------------------
# Display only the most recent exchange (Q&A) on screen
# -----------------------------------------------------
if len(st.session_state.conversation) >= 2:
    # The second-to-last message is from the user, and the last one is from the assistant
    last_user = st.session_state.conversation[-2][1]
    last_assistant = st.session_state.conversation[-1][1]
    st.markdown(f"**You:** {last_user}")
    st.markdown(f"**Assistant:** {last_assistant}")
elif len(st.session_state.conversation) == 1:
    # If there's only one message in the list, it must be from the user
    st.markdown(f"**You:** {st.session_state.conversation[-1][1]}")
