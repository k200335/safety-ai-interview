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

# 2. Vector DB 로드 (캐싱 적용)
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
        # 새 문제 클릭 시 답안 입력창 자동 초기화
        if "user_answer_key" in st.session_state:
            st.session_state["user_answer_key"] = ""
            
        try:
            with st.spinner("⚡ 선택한 과목의 기출문제를 분석하여 출제 중입니다..."):
                # 선택된 과목명 정제 (숫자 서식 제거)
                subject_clean = selected_subject.split(". ")[-1] if ". " in selected_subject else selected_subject
                
                if selected_subject == "🎲 전체 (무작위 기출 추출)":
                    search_query = "산업안전지도사 2차 3차 면접 기출문제"
                else:
                    search_query = f"{subject_clean} 기출문제 2차 3차 구술"
                
                # DB에서 선택 과목 관련 지문 상위 3개 추출 후 무작위 선택
                past_docs = vector_db.similarity_search(search_query, k=3)
                
                if past_docs:
                    chosen_doc = random.choice(past_docs).page_content
                else:
                    chosen_doc = subject_clean

                # 선택 범위 엄격 제한 프롬프트
                prompt = f"""
                당신은 산업안전지도사 2차/3차 면접 수석 출제위원입니다.

                [선택된 과목]: {subject_clean}
                [검색된 기출/지침 지문]:
                {chosen_doc[:600]}

                [엄격 출제 지침]:
                1. 반드시 위에 제시된 [{subject_clean}] 지문 범위 내에 존재하는 내용으로만 질문을 만드세요. 선택된 범위 외의 타 과목/지침 내용은 절대 섞지 마세요.
                2. 위 지문에서 실제 출제되었던 기출문제를 그대로 복원하거나, 해당 지침 조항에 근거한 동일한 유형/형태의 유사 문제를 1개만 만드세요.
                3. 질문 어조는 실제 구술 면접 말투(~에 대해 설명하시오, ~의 기준 3가지를 말하시오 등)를 엄격히 준수하세요.
                4. 서론, 인사말, 문항 번호 없이 오직 단도직입적인 질문(1~2문장 이내)만 출력하세요.
                """

                response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2, # 픽션 방지 및 엄격한 범위 준수
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
            with st.spinner("⚡ 채점 진행 중..."):
                docs = vector_db.similarity_search(user_answer_input, k=1)
                ref_text = docs[0].page_content if docs else ""

                eval_prompt = f"""
                산업안전지도사 면접관으로서 다음 답변을 평가하세요.

                [질문]: {st.session_state.question}
                [답변]: {user_answer_input}
                [법령/지침 기준]: {ref_text[:300]}

                [출력 양식]:
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
