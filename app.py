import os
import re
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI

st.set_page_config(page_title="산업안전지도사 법령 완전 암기 카드", layout="centered")

st.title("👷‍♂️ 산업안전지도사 법령·지침 완전 암기 시스템")
st.caption("조항 100% 정밀 매핑 | 오검색 0% | 원하는 조항부터 연속 학습")

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
    clean_text = text_to_speak.replace('"', "'").replace('\n', ' ').replace('`', '')
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
if "target_article" not in st.session_state:
    st.session_state.target_article = 1
if "current_doc" not in st.session_state:
    st.session_state.current_doc = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = ""
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False

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

st.subheader("📚 학습 범위 및 시작 조항 설정")

col_sel1, col_sel2 = st.columns([2, 1])

with col_sel1:
    selected_display_name = st.selectbox(
        "학습할 법령/지침을 선택하세요:",
        list(SUBJECT_MAP.keys())
    )

target_collection = SUBJECT_MAP[selected_display_name]

with col_sel2:
    start_art = st.number_input(
        "시작 조항 번호 (제N조):",
        min_value=1,
        value=st.session_state.target_article,
        step=1
    )
    if start_art != st.session_state.target_article:
        st.session_state.target_article = start_art

# 🎯 오검색 0%를 위한 조항 시작점 정밀 필터링 로직
def get_exact_article_doc(collection_name, article_num):
    subject_db = get_subject_db(collection_name)
    query_str = f"제{article_num}조"
    
    docs = subject_db.similarity_search(query_str, k=40)
    
    # 조항이 시작되는 정밀 정규식 패턴 (예: "제7조(", "제 7 조(", "제7조 (", "제7조의")
    regex_pattern = re.compile(rf'제\s*{article_num}\s*조(\s*\(|\s*의|\s+[가-힣])')
    
    # 1순위: 문단 시작 또는 본문에서 '제N조(...)' 형태로 조항이 직접 시작하는 문서 추출
    for d in docs:
        content = d.page_content.strip()
        lines = content.split('\n')
        for line in lines[:3]: # 문단 상단 3줄 이내에 조항 정의가 있는지 확인
            if regex_pattern.search(line):
                return content
                
    # 2순위: 1순위에 없으면 단순 포함 문서 중 필터링
    pattern_simple = f"제{article_num}조"
    for d in docs:
        if pattern_simple in d.page_content:
            return d.page_content
            
    return docs[0].page_content if docs else f"[{selected_display_name}] 제{article_num}조 원문을 찾을 수 없습니다."

# 현재 설정된 조항 문서 로드
st.session_state.current_doc = get_exact_article_doc(target_collection, st.session_state.target_article)

# 이동 및 조작 버튼 (3열)
col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    if st.button("⬅️ 이전 조항", use_container_width=True):
        if st.session_state.target_article > 1:
            st.session_state.target_article -= 1
            st.session_state.feedback = ""
            st.session_state.show_answer = False
            st.rerun()

with col_nav2:
    if st.button(f"🔄 제{st.session_state.target_article}조 불러오기", use_container_width=True):
        st.session_state.feedback = ""
        st.session_state.show_answer = False
        st.rerun()

with col_nav3:
    if st.button("다음 조항 ➡️", use_container_width=True):
        st.session_state.target_article += 1
        st.session_state.feedback = ""
        st.session_state.show_answer = False
        st.rerun()

st.divider()

# 📋 [문제 카드] 표시
doc_lines = [l.strip() for l in st.session_state.current_doc.split('\n') if l.strip()]
header_text = doc_lines[0] if doc_lines else f"제{st.session_state.target_article}조"

st.subheader("📋 [문제] 법령/지침 조항 암기")
st.info(f"**문제:** 다음 조항의 세부 내용 및 각 호 항목을 원문 그대로 설명하시오.\n\n👉 **{header_text}**")
speak_js(f"{header_text} 세부 내용 및 각 호 항목을 설명하시오.")

st.divider()

st.subheader("🎤 답안 작성 및 암기 대조")
user_answer_input = st.text_area("머릿속으로 읊어본 후 핵심 단어/수치를 적어보세요:", key="user_answer_key", height=130)

col_act1, col_act2 = st.columns(2)

# 1. 0초 모범 답안 바로 확인
with col_act2:
    if st.button("💡 모범 답안(원문) 바로 확인 (0초)", use_container_width=True):
        st.session_state.show_answer = True
        st.session_state.feedback = ""

# 2. AI 초정밀 대조 채점 (선택적 사용)
with col_act1:
    if st.button("📝 정밀 AI 채점 받기", type="primary", use_container_width=True):
        if not user_answer_input.strip():
            st.warning("채점받을 답변을 입력해 주세요.")
        else:
            try:
                with st.spinner("⚡ DB 원문 조항 대조 채점 중..."):
                    eval_prompt = f"""
                    당신은 산업안전지도사 수석 면접관입니다.

                    [과목]: {selected_display_name}
                    [출제 조항]: 제{st.session_state.target_article}조
                    [사용자 답변]: {user_answer_input}
                    [DB 법령 원문]:
                    {st.session_state.current_doc[:1200]}

                    [채점 규칙]:
                    원문의 수치, 키워드, 각 호 항목이 정확히 들어갔는지 엄격히 평가하세요.

                    [출력 양식]:
                    1. 출처 근거: {header_text}
                    2. 결과: (합격/불합격/보완필요)
                    3. 점수: (0~100점)
                    4. 핵심 피드백: (수치/단어 누락 지적 1~2문장)
                    5. 모범 답안: (DB 법령 원문 100% 그대로 기술)
                    """

                    eval_response = client.chat.completions.create(
                        model="meta/llama-3.1-70b-instruct",
                        messages=[{"role": "user", "content": eval_prompt}],
                        temperature=0.0,
                        max_tokens=500
                    )
                    st.session_state.feedback = eval_response.choices[0].message.content
                    st.session_state.show_answer = False
                    st.rerun()
            except Exception as err:
                st.error(f"채점 중 오류가 발생했습니다: {err}")

# 모범 답안(원문) 즉시 표시
if st.session_state.show_answer:
    st.divider()
    st.subheader("📖 [원문 모범 답안] 토시 하나 안 틀린 법령 원문")
    st.success(st.session_state.current_doc)

# AI 채점 결과 표시
if st.session_state.feedback:
    st.divider()
    st.subheader("📊 AI 채점 결과")
    st.markdown(st.session_state.feedback)
