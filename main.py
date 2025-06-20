from typing import Annotated, Sequence, TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, START
from langchain_core.tools import tool
from dotenv import load_dotenv
import streamlit as st
import PyPDF2 as pdf
import os

st.set_page_config(layout='wide', page_title="Job Relevancy AI Agent", initial_sidebar_state='expanded', page_icon='🤖')

os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']

load_dotenv()

import logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s')


class Resume(BaseModel):
    name: str = Field(..., description='extract the name of the person')
    skills: List[str] = Field(..., description='extract list of the skills of the person')
    experience: float = Field(..., description='extract the total work experience of the person')
    email: str = Field(..., description='extract the email of the person')
    phone: str = Field(..., description='extract the phone number of the person')
    summary: str = Field(description='extract the profile summary of the person')

class JD(BaseModel):
    job_role: str = Field(..., description='extract the name of the Job role')
    required_skills: List[str] = Field(..., description='extract the required skills for the job role')
    experience: float = Field(..., description='extract the total work experience required for the job role')
    summary: str = Field(description='extract the summary and resposibilities of the role')



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    agent_status: Annotated[list[str], add_messages]
    user_data: Resume
    jd: JD

def llm_with_structured_output(schema):
    llm_with_structured_output = ChatGoogleGenerativeAI(model='gemini-2.0-flash').with_structured_output(schema)
    return llm_with_structured_output

@tool
def resume_data_extractor(resume: str)->dict:
    '''
    Extracts the relevant information from a resume and returns a dictionary containing the following keys:\n
    - name: str\n
    - skills: List[str]\n
    - experience: float\n
    - email: str\n
    - phone: str\n
    - summary: str\n
    '''
    if isinstance(resume, str):
        content = resume
        # print(content)
        res = llm_with_structured_output(Resume).invoke(content)
        logging.info('Resume tool called')
        print('Resume tool called')
        return {'user_data': res}

    else:
        return {'user_data': None}

@tool
def JD_data_extractor(jd: str)-> dict:
    '''
    Extracts the relevant information from Job description and returns a dictionary containing the following keys:\n
    - job_role: str\n
    - required_skills: List[str]\n
    - experience: float\n
    - summary: str
    '''
    model = llm_with_structured_output(JD)
    res = model.invoke(jd)
    logging.info('JD tool called')
    print('JD tool called')
    return {'jd': res}



tools = [resume_data_extractor, JD_data_extractor]
llm_orchastrator = ChatGoogleGenerativeAI(model='gemini-2.0-flash').bind_tools(tools)

def Job_relevancy_agent(state:AgentState) -> dict:
    system_msg = SystemMessage(content=f'''
You are a resume evaluator.

You are given structured data for a resume and a job description. Your job is to evaluate how well the resume matches the job and present your output in **clean Markdown**.

Use the following structure in your response:

---

### Verdict

Start with a **clear verdict** — is the resume suitable for the job? (e.g., "✅ Yes, the resume is a good match" or "❌ No, the resume does not match well")

---

### Comparison 

Compare the resume to the job description and provide a **detailed comparison** of skills, experience, and summary.
---

### Explanation

Briefly explain the reasoning behind your verdict. Mention how well the skills, experience, and summary align with the job description.

---

### Suggestions to Improve Resume

If the resume is a partial or good match, suggest how it can be better tailored for the job rol

    ''')

    res = llm_orchastrator.invoke([system_msg] + list(state['messages']))
    logging.info('Job relevancy agent called')
    return {'messages': [res]}

def should_continue(state: AgentState) -> str:
    messages = state['messages']
    last_message = messages[-1]

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            logging.info('decision maker called')
            return 'continue'
    logging.info('decision maker called')
    return 'stop'

graph = StateGraph(AgentState)

graph.add_node('Job relevancy agent', Job_relevancy_agent)
graph.add_node('tools(JD and Resume Parser)', ToolNode(tools))

graph.add_conditional_edges(
    'Job relevancy agent',
    should_continue,
    {
        'continue': 'tools(JD and Resume Parser)',
        'stop': END
    }
)

graph.add_edge('tools(JD and Resume Parser)', 'Job relevancy agent')

graph.add_edge(START,'Job relevancy agent')

app = graph.compile()

def run(state):
    state = app.invoke(state)
    msg = state['messages'][-1]
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            import logging
            logging.info(f"TOOL Used: {[tc['name'] for tc in msg.tool_calls]}")
            print(f"TOOL Used: {[tc['name'] for tc in msg.tool_calls]}")
    return state




with st.sidebar:
    st.markdown('''# Job Relevancy Agent''')
    jd = st.text_area(label='Paste the Job Description:', value='')
    resume_file = st.file_uploader(label='Resume', type='pdf')
    if resume_file:
        submit = st.button(label='Process',key='submit')
    else:
        submit = st.button(label='Process',key='submit',disabled=True)


if submit:
    with st.spinner(text="Cooking"):
        import time
        time.sleep(2)
        if resume_file:
            reader = pdf.PdfReader(resume_file)
            pages = reader.pages
            content = ""
            for page in pages:
                page_content = page.extract_text()
                content += page_content
            # print(content)
            msg = HumanMessage(content=f'''
            This is the job description data:
            {jd}
            
            This is the resume data:
            {content}                     
            ''')
        state = {'messages': [msg], 'user_data': None, 'jd': None}
        res= run(state)
        st.write(res['messages'][-1].content)




else:
    with st.container():
        st.markdown('''
        # This is an app to test resume against JDs
        This is the flow of this agent:
        ''')
        st.image('langgraph_diagram.png')
        st.markdown('''
        The agent start with the input from the user. The agent will call the JD and Resume Parser tools to extract the data from the resume and the JD. The agent will then call the Job Relevancy Agent to evaluate the resume against the JD and provide a verdict and suggestions to improve the resume.
                    
        This app uses the LangChain Google Generative AI and LangGraph to build the agent. The agent is a stateful agent that uses the state to store the data from the JD and Resume Parser tools. This data is then used by the Job Relevancy Agent to evaluate the resume against the JD. This is a simple example of a stateful agent that uses the state to store the data from the JD and Resume Parser tools.
                    ''')