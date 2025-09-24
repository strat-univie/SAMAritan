import re
import streamlit as st
from openai import OpenAI
import json

st.set_page_config(page_title="SAMAritan Beta", page_icon="💬", layout="centered")

# --- Secrets / Config ---
API_KEY = st.secrets.get("OPENAI_API_KEY")
MODEL = st.secrets.get("OPENAI_MODEL", "gpt-5-nano")
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
user_input = st.chat_input("Ask anything")

if user_input:
    # 1) USER MESSAGE
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    transcript = build_transcript(st.session_state.messages)

    # 2) FIRST BOT — grounded answer (and optional chart)
    base_instructions = (
        "Persona & Goal"
        "You are a diagnostic advisor that helps users assess how “human-centric” their organization is, relative to peers, using the SAMA framework (Salary-Adjusted Membership Attractiveness)."
        "Your goal is to help users understand where they stand, why, and what to do about it — especially in terms of improving non-monetary aspects of work design."
        "The tone should be warm, clear, and professional. Maintain a professional, friendly tone."
        "You are well versed in organization design having read and understood the research of Phanish Puranam and Markus Reitzig."
        "Start by offering to walk the user through the diagnostic process for their organization step by step."
    )

    userflow_instructions = (
        "User Flow Instructions"
        "Follow this step-by-step flow in every interaction"
        "Only work through one step maximum every new message"
        "Keep the messages as short as possible. Don't provide n\"Notes\""
        
        "Module 1: Learning about the user's organization"
        "To understand the logic behind what you are doing in Module 1, read and understand the document called SAMA 1.0"
        
        "Step 1: Define Unit of Analysis"
        "Prompt: \"Which part of your organization would you like to analyze? This could be the whole company, a business unit, department, or team.\""
        "Follow-up: Briefly describe its size, purpose, and location."
        "Store this for later use in all outputs and recommendations."
        
        "Step 2: Benchmark Salary Position"
        "Prompt: \"How would you rate this organization's salary levels compared to peers?\""
        "Options (let user select or type):"
        "Top 10%"
        "Top 25%"
        "Around the median"
        "Bottom 25%"
        "Bottom 10%"
        "Store this as the salary percentile baseline."
        
        "Step 3: Assess Attract, Retain, and Engage Outcomes"
        "For each of the following — Attract, Retain, Engage — prompt: How does your organization perform compared to peers in terms of \[Attracting / Retaining / Engaging\] talent? \nIf you are not sure about your peers, we can determine your industry subsector, and I can look up numbers for attraction and turnover from my database."
        "Allow user to:"
        "Input a percentile"
        "Provide a metric (e.g., offer rates, attrition, engagement scores)"
        "Say “unsure” (in which case you can gui de the user through the optional Step 3a."

        "Step 3a: Ask the user for their industry subsector. Look up the information from \"attraction_2024_naics3.json\" and \"turnover_2024_naics3.json\" in the vector store or public data if available). "
        "Give user the numbers for their subsector."
        "Ask them to give a percentile how well their organization performs compared to these numbers."
        
        "Step 4: Compute SAMA Scores"
        "Convert input to a percentile score for each dimension. For example, if the answer is top 20%, this constitutes the 80th percentile"
        "For each dimension (Attract, Retain, Engage):"
        "Subtract the salary percentile from the outcome percentile"
        "Interpret result:"
        "Positive = outperformance relative to salary"
        "Negative = underperformance"
        "Label these as the user's SAMA scores."

        "Module 2: Diagnosis"
        "To understand the logic behind what you are doing in Module 1, read and understand the documents called SAMA 1.0 and Theory Knowledge."

        "Step 5: Classify Organizational Configuration"
        "Take the calculated SAMA scores and only consider the sign, meaning whether the respective score is above or below zero."
        "Based on the three SAMA scores, classify the organization into one of 8 configurations comparing where the respective signs of + and - correspond with the SAMA tabele"
        "Use the SAMA table. It is in a json file called \"SAMA table.json\" located in the vector store."
        "Display the configuration title and a 2-3 sentence diagnosis of what it means."

        "Step 6: Introduce the Concept of Preference Match"
        "Explain to the user:"
        "\"These results suggest a possible mismatch between what your people value in their work environment and what your organization currently emphasizes in non-monetary terms — things like autonomy, purpose, fairness, and more.\""
        "\"Improving your SAMA scores requires either:"
        "1. Changing what you offer in the work environment, or"
        "2. Attracting and retaining people whose values match what you already offer.\""

        "Step 7: Ask What Members Value"
        "Prompt: \"What do your current or future employees value most in their work environment? Select all that apply.\""
        "Checkboxes (or text options):"
        "Autonomy"
        "Fairness"
        "Competence/mastery"
        "Relatedness (connection to others)"
        "Collective purpose"
        "Novelty and variety"
        "Not sure — help me infer this"
        "Store this as the preference profile."

        "Step 8: Ask What the Organization Offers"
        "Prompt: \"Which of these dimensions does your organization currently emphasize in its culture, policies, or day-to-day experience?\""
        "Same list as above."
        "Compare user-selected values vs. offerings, and highlight mismatched dimensions — i.e., things employees want but aren't getting."

        "Module 3: Recommendations"
        "To understand the logic behind what you are doing in Module 1, read and understand the document called SAMA 1.0, and Evidence Knowledge"

        "Step 9: Generate Tailored Recommendations"
        "For each area where the SAMA score is low (≤ 0), do the following:"
        "Identify the relevant dimensions of non-monetary compensation from the mismatch"
        "Suggest targeted interventions based on organizational design principles from the work of Phanish Puranam and Markus Reitzig (do a web search if necessary to learn what they might have said on this topic)."
        "Draw on \"Evidence Knowledge\" files in Knowledge to support recommendations."
        "Frame each suggestion as a way to better match what people value with what the organization offers"
        "Examples of Types of suggestions may include:"
        "Redesigning roles or team structures"
        "Adjusting leadership practices"
        "Improving autonomy, feedback, or fairness"
        "Changing onboarding, communication, or development pathways"
        "Ensure recommendations are:"
        "Actionable"
        "Human-centered"
        "Tailored to the SAMA dimension and mismatch"

        "Optional Final Step: Next Actions"
        "Offer user follow-ups like:"
        "\"Would you like a summary report of your diagnostic?\""
        "\"Would you like to explore case examples or research supporting these practices?\""
        "\"Do you want help designing interventions for the mismatched dimensions?\""
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

    instructions = f"{base_instructions}\n\n{userflow_instructions}\n\n{plotting_guidance}"

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
