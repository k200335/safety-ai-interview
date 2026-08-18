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

# 2. 임베딩 모델 로드 (캐싱)
@st.cache_resource
def get_embedding_function():
    return SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")

embedding_fn = get_embedding_function()

# 3. 과목별 선택적 Vector DB 로더
def get_subject_db(collection_name):
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_fn,
        collection_name=collection_name
    )

# 웹 브라우저 TTS 기반 음성 출력 함수 (자동 재생용)
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

# 12개 과목 매핑 (화면 표시명 : DB 컬렉션 명)
SUBJECT_MAP = {
    "1. 산업안전보건법": "sub_1",
    "2. 산업안전보건기준에 관한 규칙": "sub_2",
    "3. 중대재해 처벌 등에 관한 법률": "sub_3",
    "4. 가설공사 표준안전 작업지침": "sub_4",
    "5. 철골공사 표준안전 작업지침": "sub_5",
    "6. 추락재해방지 표준안전 작업지침": "sub_6",
    "7. 해체공사 표준안전 작업지침": "sub_7",
    "8. 터널공사 표준안전 작업지침": "sub_8",
    "9. 콘크리트공사 표준안전 작업지침": "sub_9",
    "10. 굴착공사 표준안전 작업지침": "sub_10",
    "11. 사업장 위험성평가에 관한 지침": "sub_11",
    "12. 기출문제 (11회~16회 전체)": "sub_12"
}

st.subheader("📚 면접 법령/지침/기출 선택")
selected_display_name = st.selectbox(
    "학습할 출제 범위를 선택하세요:",
    list(SUBJECT_MAP.keys())
)

target_collection = SUBJECT_MAP[selected_display_name]

col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 새로운 문제 내기", use_container_width=True):
        st.session_state.feedback = ""
        if "user_answer_key" in st.session_state:
            st.session_state["user_answer_key"] = ""
            
        try:
            with st.spinner(f"⚡ [{selected_display_name}] 전용 DB에서 문제를 추출 중입니다..."):
                subject_db = get_subject_db(target_collection)
                past_docs = subject_db.similarity_search("설치기준 구조 준수사항 규정 조항", k=3)
                
                if past_docs:
                    chosen_doc = random.choice(past_docs).page_content
                else:
                    chosen_doc = selected_display_name

                prompt = f"""
                당신은 산업안전지도사 수석 면접관입니다.

                [선택 과목 문서 내용]:
                {chosen_doc[:700]}

                [출제 규칙]:
                1. 오직 위 [선택 과목 문서 내용] 안에 명시된 세부 기술 기준 및 조항에 근거해서만 질문을 생성하세요.
                2. 구술 면접 질문 어조(~에 대해 설명하시오, ~의 기준을 말하시오 등)로 오직 질문 1개(1~2문장)만 출력하세요.
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
            with st.spinner("⚡ 선택 과목 전용 DB 기준 채점 진행 중..."):
                subject_db = get_subject_db(target_collection)
                docs = subject_db.similarity_search(user_answer_input, k=2)
                ref_text = "\n".join([d.page_content for d in docs]) if docs else ""

                eval_prompt = f"""
                당신은 산업안전지도사 수석 면접관입니다.

                [질문]: {st.session_state.question}
                [답변]: {user_answer_input}
                [과목 기준 지문]:
                {ref_text[:500]}

                [출력 양식]:
                1. 결과: (합격/불합격/보완필요)
                2. 점수: (100점 만점)
                3. 핵심 피드백: (조항 및 수치 기준 지적)
                4. 모범 답안: (관련 조항 근거 명시하여 작성)
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
    st.subheader("📊 채점 결과")
    st.markdown(st.session_state.feedback)
    
    # 채점 결과를 음성으로 들어볼 수 있는 버튼 추가
    if st.button("🔊 채점 결과 음성으로 듣기", use_container_width=True):
        # 마크다운 기호 제거 후 순수 텍스트 추출
        clean_feedback = st.session_state.feedback.replace("*", "").replace("#", "").replace("-", "")
        speak_js(clean_feedback)
