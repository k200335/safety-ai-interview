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

# 과목별 검색 키워드 매핑
SUBJECT_KEYWORDS = {
    "1. 산업안전보건법": ["산업안전보건법 사업주 의무", "안전보건관리체계", "근로자 작업중지권", "안전보건교육"],
    "2. 산업안전보건기준에 관한 규칙": ["산업안전보건기준에 관한 규칙", "보호구 착용", "안전조치 기준", "작업장 환경"],
    "3. 중대재해 처벌 등에 관한 법률": ["중대재해 처벌 등에 관한 법률", "경영책임자 의무", "안전보건 확보의무", "중대산업재해"],
    "4. 가설공사 표준안전 작업지침": ["가설공사 표준안전 작업지침", "가설도로", "비계 및 작업발판", "동바리 설치"],
    "5. 철골공사 표준안전 작업지침": ["철골공사 표준안전 작업지침", "철골 건립", "인양 작업", "구조물 가공"],
    "6. 추락재해방지 표준안전 작업지침": ["추락재해방지 표준안전 작업지침", "안전난간", "추락방지와이어", "개구부 보호"],
    "7. 해체공사 표준안전 작업지침": ["해체공사 표준안전 작업지침", "해체계획서", "압쇄작업", "파쇄 작업"],
    "8. 터널공사 표준안전 작업지침": ["터널공사 표준안전 작업지침", "NATM공법", "막장 안전", "환기 및 낙반 예방"],
    "9. 콘크리트공사 표준안전 작업지침": ["콘크리트공사 표준안전 작업지침", "거푸집 동바리", "타설 작업", "양생 작업"],
    "10. 굴착공사 표준안전 작업지침": ["굴착공사 표준안전 작업지침", "흙막이 지보공", "사면 붕괴 예방", "굴착기 안전"],
    "11. 사업장 위험성평가에 관한 지침": ["사업장 위험성평가에 관한 지침", "유해위험요인 파악", "위험성 결정", "수시 및 정기평가"],
    "12. 기출문제": ["산업안전지도사 2차 3차 기출문제", "구술 면접 기출", "단골 기출 문항"]
}

# --- UI 영역 ---
st.subheader("📚 면접 법령/지침/기출 선택")
selected_subject = st.selectbox(
    "학습할 출제 범위를 선택하세요:",
    ["🎲 전체 (무작위)"] + list(SUBJECT_KEYWORDS.keys())
)

col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 새로운 문제 내기", use_container_width=True):
        st.session_state.feedback = ""
        try:
            with st.spinner("🎯 기출 문제 패턴을 분석하여 고품질 문제를 출제 중입니다..."):
                # 키워드 선정
                if selected_subject == "🎲 전체 (무작위)":
                    all_keywords = [kw for kws in SUBJECT_KEYWORDS.values() for kw in kws]
                    query_keyword = random.choice(all_keywords)
                else:
                    query_keyword = random.choice(SUBJECT_KEYWORDS[selected_subject])

                # 1단계: 기출문제 DB 검색
                past_docs = vector_db.similarity_search(f"기출문제 {query_keyword}", k=1)
                past_context = past_docs[0].page_content if past_docs else ""

                # 2단계: 관련 법령/지침 DB 검색
                law_docs = vector_db.similarity_search(query_keyword, k=1)
                law_context = law_docs[0].page_content if law_docs else ""

                prompt = f"""
                당신은 산업안전지도사 2차/3차 면접 시험의 수석 면접관입니다.
                실제 출제되었던 기출문제의 유형과 관련 법령 지침을 바탕으로 응시자에게 제시할 최고 품질의 구술 면접 질문 1개를 만드세요.

                [기출문제 참조]
                {past_context[:400]}

                [관련 법령 및 지침]
                {law_context[:400]}

                [작성 지침]
                1. 실제 면접관이 질문하듯 단도직입적이고 명확하게 질문하세요.
                2. 인사말, 서론, 문항 번호 없이 오직 질문 문장(2문장 이내)만 출력하세요.
                3. "상황을 설명하고 대책을 묻는" 실무형 면접 질문 스타일을 적극 활용하세요.
                """

                response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.75,
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
