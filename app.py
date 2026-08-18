import os
import random
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI

st.set_page_config(page_title="산업안전지도사 AI 음성 면접관", layout="centered")

st.title("👷‍♂️ 산업안전지도사 AI 음성 모의면접관")
st.caption("NVIDIA Build API & Vector DB 기반 초고속 구술 면접 시스템")

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
if "raw_doc" not in st.session_state:
    st.session_state.raw_doc = ""

# 12개 과목 매핑
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

EXAM_PATTERNS = [
    "각 호 준수사항 설치기준 제1호 제2호 제3호",
    "작업계획서 포함사항 작성 내용 규정",
    "작업 시작 전 점검사항 점검 항목 준수",
    "안전조치 관리기준 높이 수치 간격 규정",
    "특별안전보건교육 내용 교육시간 대상"
]

st.subheader("📚 면접 법령/지침/기출 선택")
selected_display_name = st.selectbox(
    "학습할 출제 범위를 선택하세요:",
    list(SUBJECT_MAP.keys())
)

target_collection = SUBJECT_MAP[selected_display_name]

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🎲 새로운 문제 내기", use_container_width=True):
        st.session_state.feedback = ""
        st.session_state.raw_doc = ""
        if "user_answer_key" in st.session_state:
            st.session_state["user_answer_key"] = ""
            
        try:
            with st.spinner(f"⚡ [{selected_display_name}] 문제 생성 중..."):
                subject_db = get_subject_db(target_collection)
                pattern_query = random.choice(EXAM_PATTERNS)
                past_docs = subject_db.similarity_search(pattern_query, k=10)
                
                itemized_docs = [d.page_content for d in past_docs if any(char in d.page_content for char in ["1.", "2.", "①", "②", "가.", "나.", "1호"])]
                
                chosen_text = random.choice(itemized_docs) if itemized_docs else (past_docs[0].page_content if past_docs else selected_display_name)
                st.session_state.raw_doc = chosen_text  # 원본 텍스트 미리 세션 저장

                prompt = f"""
                당신은 산업안전지도사 수석 출제위원입니다.
                [선택 과목명]: {selected_display_name}
                [선택 과목 DB 원문 텍스트]: {chosen_text[:1000]}

                [출제 규칙]:
                1. 단답형/퀴즈형 질문 금지.
                2. 양식: "[법령/지침명]에 따른 [대상]의 [설치기준/준수사항/안전조치] N가지를 설명하시오."
                3. 오직 질문 1문장만 출력.
                """

                # 최신 고속 3.3 70B 모델 적용
                response = client.chat.completions.create(
                    model="meta/llama-3.3-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
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
user_answer_input = st.text_area("답변을 입력하세요 (모르는 경우 아래 '정답 확인' 클릭):", key="user_answer_key", height=120)

col_act1, col_act2 = st.columns(2)

# 1. 답안 제출 및 초고속 채점
with col_act1:
    if st.button("📝 답안 제출 및 채점받기", type="primary", use_container_width=True):
        if not st.session_state.question:
            st.warning("먼저 '새로운 문제 내기' 버튼을 눌러주세요.")
        elif not user_answer_input.strip():
            st.warning("답변을 입력해 주세요.")
        else:
            try:
                with st.spinner("⚡ 3초 정밀 채점 진행 중..."):
                    ref_text = st.session_state.raw_doc if st.session_state.raw_doc else ""

                    eval_prompt = f"""
                    당신은 산업안전지도사 면접관입니다.
                    [과목]: {selected_display_name}
                    [질문]: {st.session_state.question}
                    [답변]: {user_answer_input}
                    [DB 원문]: {ref_text[:1000]}

                    [양식]:
                    1. 출처 근거: (원문에 표기된 조항/제목)
                    2. 결과: (합격/불합격/보완필요)
                    3. 점수: (0~100점)
                    4. 핵심 피드백: (수치/키워드 감점 사유 1문장)
                    5. 모범 답안: (원문 수치 및 문장 100% 그대로 기술)
                    """

                    eval_response = client.chat.completions.create(
                        model="meta/llama-3.3-70b-instruct",
                        messages=[{"role": "user", "content": eval_prompt}],
                        temperature=0.0,
                        max_tokens=400
                    )
                    st.session_state.feedback = eval_response.choices[0].message.content
                    st.rerun()
            except Exception as err:
                st.error(f"채점 중 오류가 발생했습니다: {err}")

# 2. ⚡ 0.1초 즉시 정답 확인 (LLM을 거치지 않아 지연시간 0초!)
with col_act2:
    if st.button("💡 정답 및 해설 바로 확인 (0초)", use_container_width=True):
        if not st.session_state.question:
            st.warning("먼저 '새로운 문제 내기' 버튼을 눌러주세요.")
        else:
            ref_text = st.session_state.raw_doc if st.session_state.raw_doc else "원문 지문을 불러올 수 없습니다."
            st.session_state.feedback = f"""
            ### 📖 DB 법령/지침 원문 모범 답안 (즉시 출력)
            
            **[출처 과목]:** {selected_display_name}
            
            **[원문 기준 조항 및 내용]:**
            ```
            {ref_text}
            ```
            """
            st.rerun()

if st.session_state.feedback:
    st.divider()
    st.subheader("📊 채점 결과 / 모범 답안")
    st.markdown(st.session_state.feedback)
    
    if st.button("🔊 결과/답안 음성으로 듣기", use_container_width=True):
        clean_feedback = st.session_state.feedback.replace("*", "").replace("#", "").replace("-", "").replace("`", "")
        speak_js(clean_feedback)
