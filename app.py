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

# 2. Vector DB 로드
@st.cache_resource
def load_db():
    embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    return Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

try:
    vector_db = load_db()
except Exception as e:
    st.error("ChromaDB 로드 실패. build_db.py가 먼저 실행되었는지 확인하세요.")

# 3. 음성 출력 함수
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

if "question" not in st.session_state:
    st.session_state.question = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = ""

# 12개 출제 범위 정의
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
    "12. 기출문제 (11회~16회 전체)"
]

st.subheader("📚 면접 법령/지침/기출 선택")
selected_subject = st.selectbox(
    "학습할 출제 범위를 선택하세요:",
    ["🎲 전체 (무작위 기출 추출)"] + SUBJECTS
)

col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 새로운 문제 내기", use_container_width=True):
        st.session_state.feedback = ""
        if "user_answer_key" in st.session_state:
            st.session_state["user_answer_key"] = ""
            
        try:
            with st.spinner("⚡ 세부 조항 및 기출 패턴을 분석하여 출제 중입니다..."):
                subject_clean = selected_subject.split(". ")[-1] if ". " in selected_subject else selected_subject
                
                if selected_subject == "🎲 전체 (무작위 기출 추출)":
                    search_query = "산업안전지도사 2차 3차 면접 기출문제 조항"
                else:
                    search_query = f"{subject_clean} 구조 설치기준 준수사항 안전조치 조항"
                
                past_docs = vector_db.similarity_search(search_query, k=4)
                
                if past_docs:
                    chosen_doc = random.choice(past_docs).page_content
                else:
                    chosen_doc = subject_clean

                prompt = f"""
                당신은 산업안전지도사 2차/3차 면접 수석 출제위원입니다.

                [선택된 과목]: {subject_clean}
                [검색된 지침/조항 내용]:
                {chosen_doc[:700]}

                [엄격 출제 규칙]:
                1. 반드시 [{subject_clean}]의 세부 조항에 명시된 구체적인 '설치 기준', '구조', '간격', '치수', '재료 기준' 등에서만 출제하세요.
                2. 질문 작성 시 관련 법령/지침 조항이나 기준을 구체적으로 대답할 수 있는 질문을 작성하세요.
                3. 서론, 인사말, 문항 번호 없이 오직 단도직입적인 질문(1~2문장)만 출력하세요.
                """

                response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100
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
            with st.spinner("⚡ 법령 조항 근거 기반 정밀 채점 진행 중..."):
                docs = vector_db.similarity_search(user_answer_input, k=2)
                ref_text = "\n".join([d.page_content for d in docs]) if docs else ""

                eval_prompt = f"""
                당신은 산업안전지도사 수석 면접관입니다. 다음 답변을 법령 기준에 따라 명확한 조항 근거를 들어 평가하세요.

                [질문]: {st.session_state.question}
                [답변]: {user_answer_input}
                [근거 법령/지침 데이터]:
                {ref_text[:500]}

                [출력 양식]:
                1. 결과: (합격/불합격/보완필요)
                2. 점수: (100점 만점)
                3. 핵심 피드백: (지침/법령의 근거 조항 번호 및 정확한 치수/수치 지적)
                4. 모범 답안: (관련 법령/지침 조항 명칭 및 제0조 조항 근거를 명시하여 정확한 모범답안 작성)
                """

                eval_response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.1,
                    max_tokens=300
                )
                st.session_state.feedback = eval_response.choices[0].message.content
                st.rerun()
        except Exception as err:
            st.error(f"채점 중 오류가 발생했습니다: {err}")

if st.session_state.feedback:
    st.divider()
    st.subheader("📊 채점 결과 (법령 근거 포함)")
    st.markdown(st.session_state.feedback)
