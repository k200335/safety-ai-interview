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

# 웹 브라우저 TTS 기반 음성 출력 함수
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
            with st.spinner(f"⚡ [{selected_display_name}] 지도사 실전형 문제 추출 중..."):
                subject_db = get_subject_db(target_collection)
                past_docs = subject_db.similarity_search("설치기준 구조 준수사항 안전조치 규정 조항", k=3)
                
                if past_docs:
                    chosen_doc = random.choice(past_docs).page_content
                else:
                    chosen_doc = selected_display_name

                prompt = f"""
                당신은 산업안전지도사 2차 기술필기 및 3차 구술면접의 수석 출제위원입니다.

                [선택 과목명]: {selected_display_name}
                [선택 과목 DB 원문 텍스트]:
                {chosen_doc[:1000]}

                [엄격 출제 규칙 - 지도사 시험 실전 양식 100% 준수]:
                1. "어디서 사용하지 말아야 하나?", "무엇인가?" 같은 단답형/퀴즈형 질문은 절대로 출제하지 마세요.
                2. 반드시 위 [DB 원문 텍스트]에 나오는 법령/지침 조항 중 항목이 여러 개 나열된 조항(예: 각 호의 준수사항, 구조 기준)을 선별하세요.
                3. 질문 형태는 오직 다음 형태 중 하나로만 작성하세요:
                   - "000의 설치기준(또는 안전조치/준수사항) N가지(또는 세부기준)를 설명하시오."
                   - "000 작업 시 준수하여야 할 사항을 법령 기준에 따라 설명하시오."
                4. 서론이나 인사말 없이 오직 단도직입적인 질문 1문장만 출력하세요.
                """

                response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
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
user_answer_input = st.text_area("답변을 입력하세요:", key="user_answer_key", height=150)

if st.button("📝 답안 제출 및 채점받기", type="primary", use_container_width=True):
    if not st.session_state.question:
        st.warning("먼저 '새로운 문제 내기' 버튼을 눌러주세요.")
    elif not user_answer_input.strip():
        st.warning("답변을 입력해 주세요.")
    else:
        try:
            with st.spinner("⚡ 원문 조항 대조 및 채점 진행 중..."):
                subject_db = get_subject_db(target_collection)
                docs = subject_db.similarity_search(st.session_state.question, k=2)
                ref_text = "\n\n".join([d.page_content for d in docs]) if docs else ""

                eval_prompt = f"""
                당신은 오직 제공된 [DB 근거 원문]만 보고 채점하는 엄격한 산업안전지도사 면접관입니다.

                [선택 과목명]: {selected_display_name}
                [질문]: {st.session_state.question}
                [사용자 답변]: {user_answer_input}
                [DB 근거 원문]:
                {ref_text[:1200]}

                [절대 채점 규칙]:
                1. 산업안전지도사 시험은 법령/지침의 수치, 키워드, 항목을 정확히 암기했는지가 핵심입니다.
                2. '출처 근거' 및 '모범 답안'은 반드시 위 [DB 근거 원문] 텍스트에 직접 적혀 있는 조항/수치/문장을 100% 그대로 원문 복사하여 출력하세요.
                3. 원문에 없는 조항 번호나 다른 규정을 절대로 지어내지 마세요.

                [출력 양식]:
                1. 출처 근거: (DB 근거 원문에 직접 표기된 법령/지침 명칭 및 제0조 조항 그대로 작성)
                2. 결과: (합격/불합격/보완필요)
                3. 점수: (0~100점)
                4. 핵심 피드백: (수치 미비, 조항 누락 등 채점 사유 1~2문장)
                5. 모범 답안: (DB 근거 원문의 해당 조항 내용 및 각 호 항목을 토시 하나 바꾸지 말고 원문 그대로 100% 기술)
                """

                eval_response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.0,
                    max_tokens=600
                )
                st.session_state.feedback = eval_response.choices[0].message.content
                st.rerun()
        except Exception as err:
            st.error(f"채점 중 오류가 발생했습니다: {err}")

if st.session_state.feedback:
    st.divider()
    st.subheader("📊 채점 결과")
    st.markdown(st.session_state.feedback)
    
    if st.button("🔊 채점 결과 음성으로 듣기", use_container_width=True):
        clean_feedback = st.session_state.feedback.replace("*", "").replace("#", "").replace("-", "")
        speak_js(clean_feedback)
