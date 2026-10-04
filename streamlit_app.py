
import os
import requests
import streamlit as st

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

# -------------------------
# Page configuration
# -------------------------

st.set_page_config(
    page_title="AI Search Assistant",
    page_icon="🤖",
)

# -------------------------
# Load API keys
# -------------------------

# Supports local .env files
load_dotenv()

# Supports Streamlit Community Cloud Secrets
try:
    for key in (
        "GROQ_API_KEY",
        "TAVILY_API_KEY",
        "WEATHER_STACK_API",
    ):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

required_keys = ["GROQ_API_KEY", "TAVILY_API_KEY"]
missing_keys = [
    key for key in required_keys if not os.getenv(key)
]

if missing_keys:
    st.error(
        "Missing API keys: " + ", ".join(missing_keys)
        + ". Add them in Streamlit App Settings → Secrets."
    )
    st.stop()

# -------------------------
# Initialize Tavily
# -------------------------

tavily_client = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"]
)

# -------------------------
# Tools
# -------------------------

@tool
def search_web(query: str) -> str:
    """Search the web for current or factual information."""

    try:
        result = tavily_client.search(
            query=query,
            max_results=3,
        )
        return str(result)
    except Exception as e:
        return f"Web search failed: {e}"


@tool
def search_weather(city: str) -> str:
    """Get the current weather for a city."""

    api_key = os.getenv("WEATHER_STACK_API")

    if not api_key:
        return (
            "Weather API key is missing. "
            "Configure WEATHER_STACK_API in Streamlit Secrets."
        )

    try:
        response = requests.get(
            "https://api.weatherstack.com/current",
            params={
                "access_key": api_key,
                "query": city,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return f"Weather API error: {data['error']}"

        location = data.get("location", {})
        current = data.get("current", {})

        return (
            f"City: {location.get('name', city)}\n"
            f"Country: {location.get('country', 'Unknown')}\n"
            f"Temperature: {current.get('temperature', 'N/A')}°C\n"
            f"Conditions: {', '.join(current.get('weather_descriptions', []))}\n"
            f"Humidity: {current.get('humidity', 'N/A')}%\n"
            f"Wind speed: {current.get('wind_speed', 'N/A')} km/h"
        )

    except requests.RequestException as e:
        return f"Weather request failed: {e}"
    except ValueError:
        return "Weather API returned an invalid response."


# -------------------------
# LLM and agent
# -------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    max_retries=2,
)

agent = create_agent(
    model=llm,
    tools=[search_web, search_weather],
    system_prompt=(
        "You are a helpful AI search assistant. "
        "Be concise and accurate. "
        "Use search_web for factual or current web information. "
        "Use search_weather for current weather questions. "
        "Never invent search results or weather data."
    ),
)

# -------------------------
# Streamlit chat interface
# -------------------------

st.title("🤖 AI Search Assistant")
st.caption(
    "Ask questions about facts, web information, "
    "or current weather."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user message
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = agent.invoke({
                    "messages": st.session_state.messages
                })

                answer = result["messages"][-1].content

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception:
            st.error(
                "The agent could not complete the request. "
                "Check the app logs and verify your API keys "
                "and model availability."
            )
