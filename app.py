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
            msg.rate = 0.95;
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

if "question" not in st.session_state:
    st.session_state.question = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = ""

# 12개 출제 카테고리 정의
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

# --- UI 영역 ---
st.subheader("📚 면접 법령/지침/기출 선택")
selected_subject = st.selectbox(
    "학습할 출제 범위를 선택하세요:",
    ["🎲 전체 (기출문제 기반 무작위)"] + SUBJECTS
)

# 무작위 검색을 보장하기 위한 다양화 시드 쿼리
RANDOM_SEEDS = ["안전", "보건", "작업", "지침", "기준", "수칙", "계획", "평가", "재해", "관리", "설비", "공사"]

col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 새로운 문제 내기", use_container_width=True):
        st.session_state.feedback = ""
        try:
            with st.spinner("🎯 기출문제(11~16회)에서 키워드를 추출하여 고품질 문제를 생성 중입니다..."):
                seed_word = random.choice(RANDOM_SEEDS)
                
                # 1단계: 기출문제 DB에서 임의의 기출 문맥/패턴 무작위 추출
                if selected_subject == "🎲 전체 (기출문제 기반 무작위)":
                    search_query = f"기출문제 {seed_word}"
                else:
                    search_query = f"{selected_subject} 기출 {seed_word}"
                
                past_docs = vector_db.similarity_search(search_query, k=2)
                past_context = "\n".join([d.page_content for d in past_docs]) if past_docs else search_query

                # 2단계: 추출된 기출 문맥을 바탕으로 법령 DB 연계 검색
                law_docs = vector_db.similarity_search(past_context[:100], k=1)
                law_context = law_docs[0].page_content if law_docs else ""

                # 3단계: AI가 기출 키워드 및 문제 유형을 능동적으로 해석하여 질문 생성
                prompt = f"""
                당신은 산업안전지도사 2차/3차 면접 시험의 수석 면접관입니다.
                제공된 [기출문제 데이터]에서 핵심 출제 주제와 문제 유형(키워드)을 직접 추출한 뒤, [관련 법령 및 지침]을 참조하여 실제 시험에 나올 법한 새로운 구술 면접 질문 1개를 생성하세요.

                [기출문제 데이터 (11회~16회 연관)]
                {past_context[:500]}

                [관련 법령 및 지침]
                {law_context[:400]}

                [작성 규칙]
                1. 고정된 문항이 아닌, 기출문제의 핵심 개념을 응용한 고품질 구술 면접 질문을 만드세요.
                2. 면접관이 질문하듯 서론, 인사말, 문항 번호 없이 오직 단도직입적인 질문(2문장 이내)만 출력하세요.
                3. 실무 상황이나 특정 법령 조항의 구체적 대책을 묻는 형식으로 작성하세요.
                """

                response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
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
