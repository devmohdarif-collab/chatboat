import os
import requests

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
import streamlit as st

# Load environment variables
load_dotenv()


# Initialize Tavily
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# -------------------------
# Tools
# -------------------------

@tool
def search_web(query: str) -> str:
    """Search the web for current or factual information."""

    print(f"\n🔎 Searching web: {query}")

    result = tavily_client.search(
        query=query,
        max_results=2
    )

    return str(result)


@tool
def search_weather(city: str) -> str:
    """Get the current weather for a city."""

    print(f"\n🌤️ Getting weather for: {city}")

    api_key = os.getenv("WEATHER_STACK_API")

    url = "https://api.weatherstack.com/current"

    params = {
        "access_key": api_key,
        "query": city,
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()
    return response.json()
    # data = response.json()

    # if "error" in data:
    #     return f"Weather API error: {data['error']}"

    # location = data.get("location", {})
    # current = data.get("current", {})

    # return (
    #     f"City: {location.get('name')}\n"
    #     f"Country: {location.get('country')}\n"
    #     f"Temperature: {current.get('temperature')}°C\n"
    #     f"Weather: {current.get('weather_descriptions')}\n"
    #     f"Humidity: {current.get('humidity')}%\n"
    #     f"Wind Speed: {current.get('wind_speed')} km/h"
    # )


# -------------------------
# LLM
# -------------------------

search_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)


# -------------------------
# Agent
# -------------------------

search_agent = create_agent(
    model=search_llm,
    tools=[
        search_web,
        search_weather
    ],
    system_prompt=(
        "You are a helpful assistant. Be concise and accurate. "
        "Use search_web for web searches. "
        "Use search_weather for weather questions."
    ),
)


# -------------------------
# Run Agent
# -------------------------

result = search_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "Find the capital of India and then find the weather of New York."
            )
        }
    ]
})


# Print only final answer
print("\n" + "=" * 50)
print("FINAL ANSWER")
# print("=" * 50)

print(result["messages"][-1].content)

print("=" * 50)
# -----------------------------
# Agent
# -----------------------------

agent = create_agent(
    model=search_llm,
    tools=[
        search_web,
        search_weather
    ],
    system_prompt=(
        "You are a helpful assistant. "
        "Use search_web for web searches. "
        "Use search_weather for weather questions. "
        "Be concise and accurate."
    ),
)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="AI Search Assistant",
    page_icon="🤖",
)

st.title("🤖 AI Search Assistant")
st.caption("Ask me about facts, web information, or current weather.")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
user_input = st.chat_input(
    "Ask something..."
)


if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })


    # Get agent response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = agent.invoke({
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            })

            answer = result["messages"][-1].content

        st.markdown(answer)


    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
