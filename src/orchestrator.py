import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    SystemMessage
)

from langgraph.graph import (
    StateGraph,
    START
)

from langgraph.graph.message import add_messages
from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from langgraph.checkpoint.memory import MemorySaver


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")


# IMPORTANT:
# Import tools only from agent_tools.
# Do not add this import inside agent_tools.py.

from src.agent_tools import (
    query_telemetry_db,
    fetch_corridor_conditions,
    search_compliance_sop
)


# ==========================================
# 2. AGENT STATE
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ==========================================
# 3. LLM CONFIGURATION
# ==========================================

AGENT_LLM_SETTING = os.getenv(
    "Agent_llm",
    "OLLAMA"
).strip().upper()


if AGENT_LLM_SETTING == "OPENAI":

    print("Using OpenAI model")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv(
            "OPENAI_MODEL",
            "gpt-4o"
        ),
        temperature=0
    )


elif AGENT_LLM_SETTING == "DEEPSEEK":

    print("Using DeepSeek model")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv(
            "DEEPSEEK_MODEL",
            "deepseek-chat"
        ),
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )


else:

    print("Using local Ollama model")

    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5:3b"
        ),
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        ),
        temperature=0
    )


# ==========================================
# 4. TOOL BINDING
# ==========================================

fde_tools = [
    query_telemetry_db,
    fetch_corridor_conditions,
    search_compliance_sop
]

llm_with_tools = llm.bind_tools(fde_tools)


# ==========================================
# 5. LANGGRAPH NODES
# ==========================================

def reasoning_node(state: AgentState):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ==========================================
# 6. GRAPH ASSEMBLY
# ==========================================

print("Compiling LangGraph FDE Orchestrator...")

graph_builder = StateGraph(AgentState)

graph_builder.add_node(
    "reasoner",
    reasoning_node
)

graph_builder.add_node(
    "tools",
    ToolNode(fde_tools)
)

graph_builder.add_edge(
    START,
    "reasoner"
)

graph_builder.add_conditional_edges(
    "reasoner",
    tools_condition
)

graph_builder.add_edge(
    "tools",
    "reasoner"
)

fde_agent = graph_builder.compile(
    checkpointer=MemorySaver()
)


# ==========================================
# 7. CHAT LOOP
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("FDE Supply Chain Orchestrator Online")
    print(
        f"LLM: {AGENT_LLM_SETTING}"
    )
    print("=" * 55)

    prompt_path = (
        project_root
        / "src"
        / "prompts"
        / "system_prompt.txt"
    )

    try:

        with open(
            prompt_path,
            "r",
            encoding="utf-8"
        ) as file:

            system_instructions = file.read()

    except FileNotFoundError:

        print(
            f"Warning: Prompt file not found: {prompt_path}"
        )

        system_instructions = (
            "You are a helpful cold-chain logistics "
            "AI assistant."
        )

    system_prompt = SystemMessage(
        content=system_instructions
    )

    thread_config = {
        "configurable": {
            "thread_id": "production_test_1"
        }
    }

    fde_agent.invoke(
        {
            "messages": [system_prompt]
        },
        config=thread_config
    )

    while True:

        user_input = input(
            "\nDispatcher > "
        ).strip()

        if user_input.lower() in {
            "exit",
            "quit"
        }:
            print("Exiting agent...")
            break

        if not user_input:
            continue

        try:

            events = fde_agent.stream(
                {
                    "messages": [
                        (
                            "user",
                            user_input
                        )
                    ]
                },
                config=thread_config,
                stream_mode="updates"
            )

            for event in events:

                for node_name, node_state in event.items():

                    if node_name == "tools":

                        print(
                            "\n[System] Retrieving "
                            "external data..."
                        )

                    elif node_name == "reasoner":

                        latest_message = (
                            node_state["messages"][-1]
                        )

                        if latest_message.content:

                            print(
                                "\nFDE Agent:\n"
                                f"{latest_message.content}"
                            )

        except Exception as error:

            print(
                "\nAgent Error:\n"
                f"{error}"
            )