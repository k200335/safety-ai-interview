import os
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI

st.set_page_config(page_title="산업안전지도사 AI 음성 면접관", layout="centered")

st.title("👷‍♂️ 산업안전지도사 AI 음성 모의면접관")
st.caption("NVIDIA Build API & Vector DB 기반 무제한 구술 면접 연습")

# 1. API 키 설정
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY", "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")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# 2. Vector DB 로드 (캐싱 적용)
@st.cache_resource
def load_db():
    embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    return Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

try:
    vector_db = load_db()
except Exception as e:
    st.error("ChromaDB 로드 실패. build_db.py가 먼저 실행되었는지 확인하세요.")

# 3. 고품질 자연스러운 TTS 보이스 설정 (브라우저 최신 한국어 음성 탐색)
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
            msg.rate = 0.95; // 자연스러운 속도
            msg.pitch = 1.0;
            
            // 자연스러운 고품질 한국어 보이스 우선 선발 (Google/Microsoft 자연어 보이스)
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

if "question" not in st.session_state:
    st.session_state.question = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = ""

col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 새로운 문제 내기", use_container_width=True):
        st.session_state.feedback = ""
        try:
            with st.spinner("🚀 고속으로 문제를 출제 중입니다..."):
                # 검색 속도 향상을 위해 상위 1개 핵심 문서만 추출
                docs = vector_db.similarity_search("산업안전보건법 사업주 의무 및 안전조치", k=1)
                context_text = docs[0].page_content if docs else "산업안전보건법 관련 주요 제반 수칙"

                prompt = f"""
                당신은 산업안전지도사 2차/3차 면접 시험의 수석 면접관입니다.
                아래 참고 자료를 바탕으로 응시자에게 물어볼 구술 면접 질문 1개만 만드세요.
                서론이나 문항 번호 없이 질문 문장만 단도직입적으로 2문장 이내로 작성하세요.

                [참고 자료]
                {context_text[:400]}
                """

                # 반응 속도가 빠른 70B 모델 적용
                response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=150
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
user_answer = st.text_area("답변을 입력하세요:", height=120)

if st.button("📝 답안 제출 및 채점받기", type="primary", use_container_width=True):
    if not st.session_state.question:
        st.warning("먼저 '새로운 문제 내기' 버튼을 눌러주세요.")
    elif not user_answer.strip():
        st.warning("답변을 입력해 주세요.")
    else:
        try:
            with st.spinner("답안 채점 중..."):
                docs = vector_db.similarity_search(user_answer, k=1)
                ref_text = docs[0].page_content if docs else ""

                eval_prompt = f"""
                당신은 산업안전지도사 면접관입니다.

                [질문]: {st.session_state.question}
                [응시자 답변]: {user_answer}
                [법령 기준]: {ref_text[:400]}

                다음 형식을 지켜 피드백하세요:
                1. 결과: (합격/불합격/보완필요)
                2. 점수: (100점 만점)
                3. 핵심 피드백
                4. 모범 답안
                """

                eval_response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.2,
                    max_tokens=400
                )
                st.session_state.feedback = eval_response.choices[0].message.content
                st.rerun()
        except Exception as err:
            st.error(f"채점 중 오류가 발생했습니다: {err}")

if st.session_state.feedback:
    st.divider()
    st.subheader("📊 채점 결과")
    st.markdown(st.session_state.feedback)