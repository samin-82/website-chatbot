import streamlit as st
import requests
import json
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

try:
    from config import NARA_API_KEY, NARA_BASE_URL, NARA_MODEL
except ImportError:
    st.error("config.py not found!")
    st.stop()

BASE_URL = "https://irvex.ir/api/v1"
HEADERS = {"Accept-Language": "fa"}


@tool
def get_landing_page_data() -> str:
    """Get the landing page report and general statistics of the tournament platform.
    Use this when the user asks about the platform overview, general information,
    or what the platform is about."""
    try:
        resp = requests.get(f"{BASE_URL}/tournaments/landing-page-report", headers=HEADERS, timeout=10)
        if resp.ok:
            data = resp.json()
            return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching landing page data: {e}"
    return "Could not fetch landing page data."


@tool
def get_tournaments_statistics() -> str:
    """Get statistics about tournaments: count of ongoing, upcoming, ended tournaments,
    and total winners. Use this when the user asks about tournament statistics,
    how many tournaments exist, or platform activity."""
    try:
        resp = requests.get(f"{BASE_URL}/tournaments/reports/statistics", headers=HEADERS, timeout=10)
        if resp.ok:
            data = resp.json()
            return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching tournament statistics: {e}"
    return "Could not fetch tournament statistics."


@tool
def get_tournaments_list(time_status: str = "ALL") -> str:
    """Get a list of tournaments on the platform.
    Filter by time_status: 'UPCOMING' for future tournaments, 'ONGOING' for active ones,
    'ENDED' for completed ones, or 'ALL' for all tournaments.
    Use this when the user wants to see available tournaments or browse the tournament list."""
    try:
        params = {"page": 1, "pageSize": 10}
        if time_status and time_status.upper() != "ALL":
            params["timeStatus"] = time_status.upper()
        resp = requests.get(f"{BASE_URL}/tournaments", headers=HEADERS, params=params, timeout=10)
        if resp.ok:
            data = resp.json()
            return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching tournaments: {e}"
    return "Could not fetch tournaments list."


@tool
def get_tournament_detail(tournament_id: str) -> str:
    """Get detailed information about a specific tournament by its ID.
    The tournament_id is the unique identifier of the tournament.
    Use this when the user asks about a specific tournament's details, rules,
    dates, or other specific information."""
    try:
        resp = requests.get(f"{BASE_URL}/tournaments/{tournament_id}", headers=HEADERS, timeout=10)
        if resp.ok:
            data = resp.json()
            return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching tournament detail: {e}"
    return f"Could not fetch details for tournament {tournament_id}."


@tool
def get_tournament_leaderboard(tournament_id: str) -> str:
    """Get the leaderboard and rankings for a specific tournament.
    The tournament_id is the unique identifier of the tournament.
    Use this when the user asks about rankings, top players, standings,
    or leaderboard of a specific tournament."""
    try:
        resp = requests.get(f"{BASE_URL}/tournaments/{tournament_id}/leaderboard", headers=HEADERS, timeout=10)
        if resp.ok:
            data = resp.json()
            return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching leaderboard: {e}"
    return f"Could not fetch leaderboard for tournament {tournament_id}."


@tool
def get_honor_leaderboard() -> str:
    """Get the overall honor and achievements leaderboard across all tournaments.
    Use this when the user asks about top players overall, honor rankings,
    achievements, or the hall of fame."""
    try:
        resp = requests.get(f"{BASE_URL}/achievements/leaderboard", headers=HEADERS, timeout=10)
        if resp.ok:
            data = resp.json()
            return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching honor leaderboard: {e}"
    return "Could not fetch honor leaderboard."


@tool
def get_capital_percentage_chart(tournament_id: str) -> str:
    """Get the capital performance chart data for a specific tournament.
    The tournament_id is the unique identifier of the tournament.
    Use this when the user asks about performance charts, capital distribution,
    or financial analytics of a tournament."""
    try:
        resp = requests.get(
            f"{BASE_URL}/analytics/tournaments/{tournament_id}/capital-percentage-chart",
            headers=HEADERS,
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching capital chart: {e}"
    return f"Could not fetch capital chart for tournament {tournament_id}."


TOOLS = [
    get_landing_page_data,
    get_tournaments_statistics,
    get_tournaments_list,
    get_tournament_detail,
    get_tournament_leaderboard,
    get_honor_leaderboard,
    get_capital_percentage_chart,
]

SYSTEM_PROMPT = """You are a helpful AI assistant for the irvex.ir tournament platform.
You can answer questions about tournaments, leaderboards, platform statistics, and more.

When the user asks a question:
1. Determine which tool(s) to use based on the question.
2. Call the appropriate tool(s) to get the data.
3. Format the response clearly and in the SAME LANGUAGE as the question (Persian if Persian, English if English).
4. If the tool returns JSON data, present it in a readable, user-friendly format.
5. If you cannot find the right tool, explain what you can do instead.

You have access to these capabilities:
- Platform overview and landing page data
- Tournament statistics (counts, activity)
- Tournament lists (with filters)
- Individual tournament details
- Tournament leaderboards/rankings
- Overall honor/achievements leaderboard
- Capital performance charts
"""

llm = ChatOpenAI(
    model=NARA_MODEL,
    api_key=NARA_API_KEY,
    base_url=NARA_BASE_URL,
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


st.set_page_config(page_title="AI Agent - Tournament Platform", page_icon="🏆")
st.title("🏆 AI Agent - Tournament Platform")
st.caption("Ask me anything about tournaments, rankings, and statistics")

with st.sidebar:
    st.header("Capabilities")
    st.markdown("""
    - Platform overview
    - Tournament statistics
    - Browse tournaments
    - Tournament details
    - Leaderboards & rankings
    - Honor/achievements
    - Performance charts
    """)
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

if user_input := st.chat_input("Ask about tournaments..."):
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.spinner("Thinking and searching..."):
        try:
            response = agent.invoke({
                "messages": st.session_state.chat_history,
            })
            output = response["messages"][-1].content
        except Exception as e:
            output = f"Sorry, an error occurred: {e}"

    st.chat_message("assistant").write(output)
    st.session_state.chat_history.append(AIMessage(content=output))
