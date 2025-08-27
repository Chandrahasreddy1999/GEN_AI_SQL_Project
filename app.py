import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent, initialize_agent, AgentType
from langchain.sql_database import SQLDatabase
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from urllib.parse import quote_plus
import os
from langchain_core.prompts import SystemMessagePromptTemplate
import pandas as pd
from io import BytesIO, StringIO
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import re
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv()


# Streamlit setup

st.set_page_config(page_title="LangChain: Chat with SQL DB")
st.title("LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use the SQLite 3 Database", "Connect MySQL Database"]
select_opt = st.sidebar.radio(label="Choose the DB which you want", options=radio_opt)


# Database inputs

if radio_opt.index(select_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL user")
    mysql_password = st.sidebar.text_input("MySQL password", type="password")
    mysql_db = st.sidebar.text_input("MySQL database")
else:
    db_uri = LOCALDB


##api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
api_key = os.getenv("NVIDIA_KEY")


if not db_uri:
    st.info("Please enter the database information and uri")
##if not api_key:
  ##  st.warning("Please add the NVIDIA API key")
    ##st.stop()
if not api_key:
    st.warning("Please add the NVIDIA API key in your .env file")
    st.stop()



llm = ChatNVIDIA(
    model="qwen/qwen2.5-coder-32b-instruct",
    api_key=api_key,
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)


# Configure Database

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "company_employees.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MYSQL connection details")
            st.stop()
        mysql_password_encoded = quote_plus(mysql_password)
        return SQLDatabase(
            create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password_encoded}@{mysql_host}/{mysql_db}")
        )

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)


# Save to Excel Tool

class SaveToExcelInput(BaseModel):
    data: str = Field(..., description="Data in CSV or JSON string format to be saved into Excel.")

class SaveToExcelTool(BaseTool):
    name: str = "save_to_excel"
    description: str = "Use this to save query results into an Excel file. Input must be tabular data (CSV/JSON string)."
    args_schema: Optional[Type[BaseModel]] = SaveToExcelInput

    def _run(self, data: str):
        try:
            try:
                df = pd.read_csv(BytesIO(data.encode("utf-8")))
            except Exception:
                df = pd.read_json(data)
            st.session_state["last_query_df"] = df
            return "Data saved. Use the download button below to get the Excel file."
        except Exception as e:
            return f"Failed to save Excel: {e}"

    async def _arun(self, data: str):
        raise NotImplementedError("Async not supported")


# Initialize Agent

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
excel_tool = SaveToExcelTool()
tools = toolkit.get_tools() + [excel_tool]

system_message = SystemMessagePromptTemplate.from_template("""
You are a precise SQL assistant.
Rules:
- Always use SQL tools to query the database (never guess values).
- When showing query results, return ONLY a markdown table (| col1 | col2 | ... |) or CSV format.
- Do not include explanations, text, or commentary outside the table.
- If no rows exist, still return a valid empty table with headers.
- Columns must be exactly: NAME, DEPARTMENT, TEAM, PERFORMANCE_SCORE (if query matches those fields).
""")

sql_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": system_message}
)


# Helper functions

def markdown_to_csv(md_text: str) -> str:
    lines = md_text.strip().split('\n')
    table_lines = [
        line for line in lines
        if re.match(r"^\|.*\|$", line.strip())
        and not re.match(r"^\|?[-\s|]+\|?$", line.strip())
    ]
    csv_lines = []
    for line in table_lines:
        cleaned = ','.join(cell.strip() for cell in line.strip().strip('|').split('|'))
        csv_lines.append(cleaned)
    return '\n'.join(csv_lines)

def parse_table_from_response(response: str) -> pd.DataFrame | None:
    try:
        csv_data = markdown_to_csv(response)
        if csv_data.strip():
            return pd.read_csv(StringIO(csv_data))
    except:
        pass
    try:
        return pd.read_csv(StringIO(response))
    except:
        return None

def send_email_with_excel(recipient_emails: list, excel_buffer: BytesIO, sender_email: str, app_password: str, filename="query_results.xlsx"):
    msg = EmailMessage()
    msg["Subject"] = "Query Results Excel File"
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipient_emails)  # now works with a list
    msg.set_content("Please find the attached Excel file with the query results.")

    excel_buffer.seek(0)
    msg.add_attachment(
        excel_buffer.read(),
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename
    )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False



# Chat UI

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.pop("last_query_df", None)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask any question from the database")
if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = sql_agent.run(
            f"{user_query}\n\nReturn the results ONLY as a markdown table or CSV, no explanations.",
            callbacks=[streamlit_callback]
        )

        df = parse_table_from_response(response)
        if df is not None and not df.empty:
            st.session_state["last_query_df"] = df
            st.write("### Query Results (Table)")
            st.dataframe(df)
        else:
            st.warning("⚠️ Could not parse tabular data. Raw response:")
            st.code(response)


# Export / Email Sidebar

st.sidebar.markdown("### 📊 Export Data")
df = st.session_state.get("last_query_df")

if df is not None and not df.empty:
    buffer = BytesIO()
    df.to_excel(buffer, index=False, sheet_name="Results", engine="xlsxwriter")
    buffer.seek(0)
    
    st.sidebar.download_button(
        label="📥 Download Last Query as Excel",
        data=buffer,
        file_name="query_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📧 Send Excel by Email")
    recipient = st.sidebar.text_input("Enter recipient email address")
    sender_email = st.sidebar.text_input("Your Gmail address")
    app_password = st.sidebar.text_input("Your Gmail App Password", type="password")
    if st.sidebar.button("Send Email"):
        if recipient and sender_email and app_password:
            recipient_list = [email.strip() for email in recipient.split(",") if email.strip()]
            if recipient_list:
                all_sent = True
                for rec in recipient_list:
                    sent = send_email_with_excel([rec], buffer, sender_email, app_password)  # wrap single email in list
                    if not sent:
                        all_sent = False
                        
                if all_sent:
                    st.sidebar.success(f"Email sent successfully to: {', '.join(recipient_list)}")
                else:
                    st.sidebar.error("Failed to send email to one or more recipients.")

