import re
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Chat (Responses API + Vector Store + Plotly + Haiku)", page_icon="💬", layout="centered")

# --- Secrets / Config ---
API_KEY = st.secrets.get("OPENAI_API_KEY")
MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
VECTOR_STORE_ID = st.secrets.get("OPENAI_VECTOR_STORE_ID", "")

# Toggle the second bot on/off here if you like
ENABLE_SECOND_BOT = True

if not API_KEY:
    st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
    st.stop()

if not VECTOR_STORE_ID:
    st.error("Missing OPENAI_VECTOR_STORE_ID in .streamlit/secrets.toml (required for file_search).")
    st.stop()

client = OpenAI(api_key=API_KEY)

st.title("SAMAritan Beta")

st.subheader("An AI agent that helps improve your human-capital strategy, designed by Phanish Puranam & Markus Reitzig")

# --- Chat history in session state ---
# Entries can be:
#   {"role": "user"|"assistant", "content": "text"}  OR
#   {"role": "assistant", "plot": <plotly_figure>, "caption": "..."}
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Utilities ---
def build_transcript(history):
    """Compile chat history into a single text transcript for Responses 'input'."""
    lines = []
    for m in history:
        if "plot" in m:
            lines.append("Assistant: [chart]")
        else:
            speaker = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)

def extract_python_code(text: str):
    """
    Extract a Python code block wrapped as:
    ```python
    <code>
    ```
    Returns the inner code or None.
    """
    pattern = r"```python\s(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else None

def remove_python_blocks(text: str):
    """Remove all ```python ...``` fenced code blocks from the text."""
    return re.sub(r"```python\s.*?```", "", text, flags=re.DOTALL).strip()

# --- Render previous turns (including persistent plots) ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if "plot" in m:
            st.plotly_chart(m["plot"], theme="streamlit", use_container_width=True)
            if m.get("caption"):
                st.caption(m["caption"])
        else:
            st.markdown(m["content"])

# --- Input box ---
user_input = st.chat_input("Ask a question (the assistant can also visualize data) …")

if user_input:
    # 1) USER MESSAGE
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    transcript = build_transcript(st.session_state.messages)

    # 2) FIRST BOT — grounded answer (and optional chart)
    base_instructions = (
        "You are a careful, concise assistant providing individual information on Prof. Markus Reitzig's Book 'Get Better at Flatter'. "
        "Use ONLY the information retrieved from the file_search tool. "
        "If a retrieved chunk contains a page marker like '{:.page-1}', translate it into a citation in the following format:\n\n"
        "Reitzig, M. (2022). Get better at flatter. Springer International Publishing., p. <page number>\n\n"
        "Example: If the marker is '{:.page-3}', cite it as 'Reitzig, M. (2022). Get better at flatter. Springer International Publishing., p. 3'. "
        "If no page marker is present, omit the page reference. "
        "If the knowledge base does not contain the answer, reply with: "
        "\"I don't know based on the provided knowledge base.\" "
        "Do not rely on outside or general knowledge. Do not fabricate facts."
    )

    plotting_guidance = (
        "If the user asks to visualize, chart, graph, plot, or show a figure, produce Plotly-only Python code. "
        "Return the code wrapped in a single fenced block exactly like:\n"
        "```python\n"
        "# (imports if needed)\n"
        "# construct data from the retrieved context\n"
        "# create a Plotly figure assigned to the variable `fig`\n"
        "```\n"
        "Requirements:\n"
        "- Use Plotly only (no matplotlib).\n"
        "- Name the resulting figure variable `fig`.\n"
        "- Do NOT call fig.show().\n"
        "- You may include a brief natural-language explanation before the code block."
    )

    instructions = f"{base_instructions}\n\n{plotting_guidance}"

    req = {
        "model": MODEL,
        "input": transcript,
        "instructions": instructions,
        "tools": [{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
        }],
    }

    try:
        resp = client.responses.create(**req)
        assistant_text = resp.output_text or ""
    except Exception as e:
        assistant_text = f"Sorry, there was an error calling the API:\n\n```\n{e}\n```"
        resp = None

    # 3) RENDER FIRST BOT (text and/or chart)
    #    We'll also prepare clean_text_for_second_bot to feed into the haiku bot.
    clean_text_for_second_bot = None

    with st.chat_message("assistant"):
        code = extract_python_code(assistant_text)
        if code:
            # Show any explanatory text before code (code stays hidden)
            explanation = remove_python_blocks(assistant_text)
            if explanation:
                st.markdown(explanation)
                clean_text_for_second_bot = explanation  # feed explanation to haiku bot
            else:
                # If nothing but code, provide a small label for context
                clean_text_for_second_bot = "A chart was generated based on the answer."

            # Execute the Plotly code (expects a `fig` variable; no fig.show())
            try:
                safe_code = code.replace("fig.show()", "").strip()
                exec_globals = {"st": st}
                exec_locals = {}
                exec(safe_code, exec_globals, exec_locals)

                # Get figure object
                fig = exec_locals.get("fig", exec_globals.get("fig"))
                if fig is not None and hasattr(fig, "to_dict"):
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                    # Persist the chart in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "plot": fig,
                        "caption": None
                    })
                else:
                    st.info("I generated code but couldn't detect a Plotly figure named 'fig'.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Chart generation attempted, but no figure was detected."
                    })
            except Exception as ex:
                st.error(f"Plot execution error:\n{ex}")
                st.session_state.messages.append({"role": "assistant", "content": f"Plot execution error: {ex}"})
        else:
            # No code -> normal text
            st.markdown(assistant_text)
            clean_text_for_second_bot = assistant_text
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})

    # 4) SECOND BOT — transforms the first bot’s text into a haiku
    if ENABLE_SECOND_BOT and clean_text_for_second_bot:
        # Build a separate Responses call (no tools; no external knowledge).
        # We pass only the first bot's visible text (code blocks already removed).
        try:
            second_resp = client.responses.create(
                model=MODEL,
                input=(
                    "This is the previous assistant message:\n\n"
                    + clean_text_for_second_bot
                ),
                instructions=(
                    "You are a second assistant. Read the provided assistant message and compose a haiku "
                    "(three lines, 5-7-5 syllables) that captures its essence. "
                    "Do NOT add new facts. Do NOT include code or citations. "
                    "Output only the haiku (three lines)."
                ),
            )
            haiku_text = second_resp.output_text.strip()
        except Exception as e:
            haiku_text = f"_Haiku generation error:_ `{e}`"

        with st.chat_message("assistant"):
            st.markdown(f"🟣 *Poet bot*\n\n{haiku_text}")

        # Also persist the haiku as a normal assistant turn for transcript/history
        st.session_state.messages.append({"role": "assistant", "content": f"Poet bot:\n{haiku_text}"})


