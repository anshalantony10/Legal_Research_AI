from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
# from langchain.agents.react.agent import create_react_agent
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain import hub

# from langchain.agents import AgentExecutor, create_react_agent
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['SERPAPI_API_KEY'] = os.getenv('SERP_API_KEY')

def search(query):
    llm = OpenAI(temperature=0)
    prompt = hub.pull("hwchase17/react")
    tool_names = ["serpapi"]
    tools = load_tools(tool_names)
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    return agent.run(query)
    # agent = create_react_agent(llm,tools,prompt)

    # agent_executor = AgentExecutor(agent=agent, tools=tools)

    # agent_executor.invoke({"input": "hi"})

    # return agent.response


if __name__ == "__main__":
    print(search("what are my rights as a tenant in ireland?"))