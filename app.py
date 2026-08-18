import os
import random
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI

st.set_page_config(page_title="산업안전지도사 AI 음성 면접관", layout="centered")

st.title("👷‍♂️ 산업안전지도사 AI 음성 모의면접관")
st.caption("NVIDIA Build API & Vector DB 기반 무제한 구술 면접 연습")

# 1. API 키 및 클라이언트 설정
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY", "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")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# 2. Vector DB 로드 (캐싱 및 빠른 로더 적용)
@st.cache_resource
def load_db():
    embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    return Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

try:
    vector_db = load_db()
except Exception as e:
    st.error("ChromaDB 로드 실패. build_db.py가 먼저 실행되었는지 확인하세요.")

# 3. 고품질 자연스러운 TTS 보이스 설정
def speak_js(text_to_speak=""):
    if not text_to_speak:
        return
    clean_text = text_to_speak.replace('"', "'").replace('\n', ' ')
    js_code = f"""
    <script>
    function playNaturalVoice(text) {{
        if (!('speechSynthesis' in window)) return;
        window.speechSynthesis.cancel();
        
        var speakNow = function() {{
            var voices = window.speechSynthesis.getVoices();
            var msg = new SpeechSynthesisUtterance(text);
            msg.lang = 'ko-KR';
            msg.rate = 1.0;
            msg.pitch = 1.0;
            
            var naturalVoice = voices.find(function(v) {{
                return v.lang.includes('ko') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Online'));
            }}) || voices.find(function(v) {{ return v.lang.includes('ko'); }});
            
            if (naturalVoice) {{
                msg.voice = naturalVoice;
            }}
            window.speechSynthesis.speak(msg);
        }};

        if (window.speechSynthesis.getVoices().length === 0) {{
            window.speechSynthesis.onvoiceschanged = speakNow;
        }} else {{
            speakNow();
        }}
    }}
    playNaturalVoice("{clean_text}");
    </script>
    """
    components.html(js_code, height=0)

# 세션 상태 초기화
if "question" not in st.session_state:
    st.session_state.question = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = ""

# 12개 출제 카테고리
SUBJECTS = [
    "1. 산업안전보건법",
    "2. 산업안전보건기준에 관한 규칙",
    "3. 중대재해 처벌 등에 관한 법률",
    "4. 가설공사 표준안전 작업지침",
    "5. 철골공사 표준안전 작업지침",
    "6. 추락재해방지 표준안전 작업지침",
    "7. 해체공사 표준안전 작업지침",
    "8. 터널공사 표준안전 작업지침",
    "9. 콘크리트공사 표준안전 작업지침",
    "10. 굴착공사 표준안전 작업지침",
    "11. 사업장 위험성평가에 관한 지침",
    "12. 기출문제 (11회~16회)"
]

st.subheader("📚 면접 법령/지침/기출 선택")
selected_subject = st.selectbox(
    "학습할 출제 범위를 선택하세요:",
    ["🎲 전체 (기출문제 기반 무작위)"] + SUBJECTS
)

RANDOM_SEEDS = ["안전", "보건", "작업", "지침", "기준", "수칙", "계획", "평가", "재해", "관리"]

col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 새로운 문제 내기", use_container_width=True):
        st.session_state.feedback = ""
        # 텍스트 입력창 초기화
        if "user_answer_key" in st.session_state:
            st.session_state["user_answer_key"] = ""
            
        try:
            with st.spinner("⚡ 기출 키워드 추출 및 문제 생성 중..."):
                seed_word = random.choice(RANDOM_SEEDS)
                search_query = f"{selected_subject} {seed_word}" if selected_subject != "🎲 전체 (기출문제 기반 무작위)" else f"기출 {seed_word}"
                
                # 경량 DB 조회 (k=1로 속도 최적화)
                past_docs = vector_db.similarity_search(search_query, k=1)
                past_context = past_docs[0].page_content if past_docs else search_query

                prompt = f"""
                당신은 산업안전지도사 면접관입니다.
                다음 [기출/법령 데이터]의 핵심 개념을 바탕으로 응시자에게 제시할 구술 면접 질문 1개만 작성하세요.

                [데이터]
                {past_context[:300]}

                [지침]
                - 인사말, 서론 없이 오직 면접 질문(2문장 이내)만 즉시 출력할 것.
                """

                # 생성 토큰 단축(max_tokens=90)으로 응답 속도 극대화
                response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=90
                )
                st.session_state.question = response.choices[0].message.content
                st.rerun()
        except Exception as err:
            st.error(f"오류가 발생했습니다: {err}")

if st.session_state.question:
    st.subheader("📋 면접 질문")
    st.info(st.session_state.question)
    speak_js(st.session_state.question)

st.divider()

st.subheader("🎤 나의 답안 입력")
user_answer_input = st.text_area("답변을 입력하세요:", key="user_answer_key", height=120)

if st.button("📝 답안 제출 및 채점받기", type="primary", use_container_width=True):
    if not st.session_state.question:
        st.warning("먼저 '새로운 문제 내기' 버튼을 눌러주세요.")
    elif not user_answer_input.strip():
        st.warning("답변을 입력해 주세요.")
    else:
        try:
            with st.spinner("⚡ 빠른 채점 진행 중..."):
                docs = vector_db.similarity_search(user_answer_input, k=1)
                ref_text = docs[0].page_content if docs else ""

                eval_prompt = f"""
                산업안전지도사 면접관으로서 다음 답변을 평가하세요.

                [질문]: {st.session_state.question}
                [답변]: {user_answer_input}
                [기준]: {ref_text[:300]}

                [반드시 아래 양식으로만 간결하게 출력]:
                1. 결과: (합격/불합격/보완필요)
                2. 점수: (100점 만점)
                3. 핵심 피드백: (2문장 이내)
                4. 모범 답안: (2문장 이내)
                """

                eval_response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.1,
                    max_tokens=250
                )
                st.session_state.feedback = eval_response.choices[0].message.content
                st.rerun()
        except Exception as err:
            st.error(f"채점 중 오류가 발생했습니다: {err}")

if st.session_state.feedback:
    st.divider()
    st.subheader("📊 채점 결과")
    st.markdown(st.session_state.feedback)
