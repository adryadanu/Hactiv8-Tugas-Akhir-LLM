# Import the necessary libraries
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Page Configuration and Title ---
st.title("💬 Kustomisasi LangGraph ReAct Chatbot")
st.caption("Chatbot yang bisa menyesuaikan gaya bahasa, domain, dan kreatifitasnya")

# --- 2. Sidebar for Settings ---
with st.sidebar:
    st.subheader("Pengaturan")
    
    # API Key
    google_api_key = st.text_input("Google AI API Key", type="password")
    
    # Domain Pengetahuan
    domain = st.selectbox(
        "Domain pengetahuan:",
        options=["Umum", "Kesehatan", "Edukasi", "Travel", "Personal Productivity", "Hobi"]
    )
    
    # Gaya bahasa
    style = st.selectbox(
        "Gaya bahasa:",
        options=["Formal", "Santai", "Persuasif", "Motivasi", "Humor"]
    )
    
    # Kreativitas (temperature)
    temperature = st.slider("Kreatifitas respon (temperature):", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    
    # Reset conversation
    reset_button = st.button("Hapus Percakapan", help="hapus semua pesan")

# --- 3. API Key and Agent Initialization ---
if not google_api_key:
    st.info("Masukkan Google AI API key Anda untuk memulai percakapan", icon="🗝️")
    st.stop()

# Re-init agent if api key, domain, style, or temperature changes
if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key) \
    or (getattr(st.session_state, "_last_domain", None) != domain) \
    or (getattr(st.session_state, "_last_style", None) != style) \
    or (getattr(st.session_state, "_last_temp", None) != temperature):

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=temperature
        )
        
        # Kustomisasi prompt berdasarkan domain dan style
        prompt_prefix = f"You are a helpful, friendly assistant specialized in {domain} domain. "
        prompt_prefix += f"Use a {style.lower()} tone. "
        prompt_prefix += "Respond concisely and clearly."

        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[],  # Bisa ditambah tools API eksternal nanti
            prompt=prompt_prefix
        )
        
        st.session_state._last_key = google_api_key
        st.session_state._last_domain = domain
        st.session_state._last_style = style
        st.session_state._last_temp = temperature
        st.session_state.pop("messages", None)  # Clear messages on new setup
    except Exception as e:
        st.error(f"API Key yang Anda masukkan salah: {e}")
        st.stop()

# --- 4. Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if reset_button:
    st.session_state.pop("agent", None)
    st.session_state.pop("messages", None)
    st.rerun()

# --- 5. Display Past Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. Handle User Input and Agent Communication ---
prompt = st.chat_input("Ketikkan pesanmu disini...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare messages for agent
    messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # Optional: contoh integrasi API eksternal sederhana berdasarkan domain
    # Misal, kalau domain Travel, bisa kita tambahkan rekomendasi destinasi (dummy)
    def external_api_integration(user_message, domain):
        if domain == "Travel":
            # Dummy API response, misal rekomendasi tempat wisata
            if "recommend" in user_message.lower():
                return "Sebagai rekomendasi, coba kunjungi Bali untuk liburan yang menyenangkan!"
        return None
    
    try:
        # Cek integrasi API eksternal
        external_response = external_api_integration(prompt, domain)
        if external_response:
            answer = external_response
        else:
            # Panggil agent untuk jawaban
            response = st.session_state.agent.invoke({"messages": messages})
            if "messages" in response and len(response["messages"]) > 0:
                answer = response["messages"][-1].content
            else:
                answer = "Maaf, kami tidak dapat memberikan respon"
    except Exception as e:
        answer = f"An error occurred: {e}"

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
