import os
import re
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI

st.set_page_config(page_title="산업안전지도사 법령 완전 암기 카드", layout="centered")

st.title("👷‍♂️ 산업안전지도사 법령·지침 완전 암기 시스템")
st.caption("100% 원문 매핑 | 조·항·호·목 정밀 분리 | 실전 회상 암기")

# 1. API 키 및 클라이언트 설정
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY", "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")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# 2. 임베딩 모델 로드
@st.cache_resource
def get_embedding_function():
    return SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")

embedding_fn = get_embedding_function()

# 3. Vector DB 로더
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
    clean_text = text_to_speak.replace('"', "'").replace('\n', ' ').replace('`', '').replace('\\', '')
    js_code = f"""
    <script>
    (function() {{
        if (!('speechSynthesis' in window)) return;
        window.speechSynthesis.cancel();
        
        var speakNow = function() {{
            window.speechSynthesis.cancel();
            var voices = window.speechSynthesis.getVoices();
            var msg = new SpeechSynthesisUtterance("{clean_text}");
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
            setTimeout(speakNow, 150);
        }}
    }})();
    </script>
    """
    components.html(js_code, height=0)

# 세션 상태 초기화
if "target_article" not in st.session_state:
    st.session_state.target_article = 1
if "feedback" not in st.session_state:
    st.session_state.feedback = ""
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False

# 과목 매핑
SUBJECT_MAP = {
    "1. 산업안전보건법": "sub_1",
    "2. 산업안전보건법 시행령": "sub_2",
    "3. 산업안전보건법 시행규칙": "sub_3",
    "4. 산업안전보건기준에 관한 규칙": "sub_4",
    "5. 중대재해 처벌 등에 관한 법률": "sub_5",
    "6. 중대재해 처벌 등에 관한 법률 시행령": "sub_6",
    "7. 가설공사 표준안전 작업지침": "sub_7",
    "8. 철골공사 표준안전 작업지침": "sub_8",
    "9. 추락재해방지 표준안전 작업지침": "sub_9",
    "10. 해체공사 표준안전 작업지침": "sub_10",
    "11. 터널공사 표준안전 작업지침 (NATM)": "sub_11",
    "12. 콘크리트공사 표준안전 작업지침": "sub_12",
    "13. 운반하역 표준안전 작업지침": "sub_13",
    "14. 발파 표준안전 작업지침": "sub_14",
    "15. 사업장 위험성평가에 관한 지침": "sub_15",
    "16. 굴착공사 표준안전 작업지침": "sub_16"
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
        st.session_state["user_answer_key"] = ""

def get_full_article_content(collection_name, article_num):
    try:
        subject_db = get_subject_db(collection_name)
        all_docs = subject_db.get()
    except Exception:
        return f"제{article_num}조", f"⚠️ [{selected_display_name}] DB 데이터가 서버에 존재하지 않습니다."

    if not all_docs or not all_docs['documents']:
        return f"제{article_num}조", "원문 데이터를 불러올 수 없습니다."
    
    raw_full_text = "\n".join(all_docs['documents'])
    
    curr_pattern = re.compile(rf'제\s*{article_num}\s*조(\s*\([^)]+\)|\s+[가-힣])?')
    next_pattern = re.compile(rf'제\s*{article_num + 1}\s*조(\s*\([^)]+\)|\s+[가-힣])?')

    curr_match = curr_pattern.search(raw_full_text)

    if curr_match:
        start_idx = curr_match.start()
        next_match = next_pattern.search(raw_full_text, start_idx)
        
        if next_match:
            end_idx = next_match.start()
            article_raw = raw_full_text[start_idx:end_idx].strip()
        else:
            article_raw = raw_full_text[start_idx:start_idx + 8000].strip()

        item_match = re.search(r'(?<!\d)(1\.|①|가\.|1호)(?!\d)', article_raw)
        
        if item_match:
            split_idx = item_match.start()
            q_text = article_raw[:split_idx].strip()
            a_text_raw = article_raw[split_idx:].strip()
            
            a_text = re.sub(r'(?<!\d)(\d+\.)', r'\n\1', a_text_raw)
            a_text = re.sub(r'([①-⑮])', r'\n\1', a_text)
            a_text = re.sub(r'([가-하]\.)', r'\n   \1', a_text)
            a_text = re.sub(r'(\b제\d+절\b)', r'\n\1', a_text).strip()
        else:
            q_text = article_raw
            a_text = "하위 세부 항목(호)이 없는 조항입니다."

        return q_text, a_text

    return f"제{article_num}조", f"[{selected_display_name}] 제{article_num}조 원문을 찾지 못했습니다."

q_text, a_text = get_full_article_content(target_collection, st.session_state.target_article)

col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    if st.button("⬅️ 이전 조항", use_container_width=True):
        if st.session_state.target_article > 1:
            st.session_state.target_article -= 1
            st.session_state.feedback = ""
            st.session_state.show_answer = False
            st.session_state["user_answer_key"] = ""
            st.rerun()

with col_nav2:
    if st.button(f"🔄 제{st.session_state.target_article}조 불러오기", use_container_width=True):
        st.session_state.feedback = ""
        st.session_state.show_answer = False
        st.session_state["user_answer_key"] = ""
        st.rerun()

with col_nav3:
    if st.button("다음 조항 ➡️", use_container_width=True):
        st.session_state.target_article += 1
        st.session_state.feedback = ""
        st.session_state.show_answer = False
        st.session_state["user_answer_key"] = ""
        st.rerun()

st.divider()

# 📋 [문제 카드] (24px - 20% 보정 폰트 적용)
st.subheader("📋 [문제] 조항 암기")

q_html = f"""
<div style="
    background-color: #e8f4f8; 
    border-left: 6px solid #2980b9; 
    padding: 20px; 
    border-radius: 8px; 
    font-family: sans-serif;
    color: #1c2833;">
    <div style="font-size: 20px; font-weight: bold; color: #2980b9; margin-bottom: 10px;">[출제 조항]:</div>
    <div style="font-size: 24px; line-height: 1.7; font-weight: 600; color: #111;">
        {q_text.replace('\n', '<br>')}
    </div>
    <hr style="border: 0.5px solid #aeb6bf; margin: 15px 0;">
    <div style="font-size: 21px; font-weight: bold; color: #d35400;">
        👉 문제: 위 조항의 세부 내용 및 각 호 항목을 원문 그대로 인출(설명)하시오.
    </div>
</div>
"""
q_box_height = max(220, int(len(q_text) * 0.8) + 140)
components.html(q_html, height=q_box_height, scrolling=True)

# 출제 문제 자동 음성 출력
speak_js(f"{q_text} 세부 내용을 설명하시오.")

st.divider()

st.subheader("🎤 답안 작성 및 암기 대조")
user_answer_input = st.text_area("머릿속으로 읊어본 후 핵심 단어/수치를 적어보세요:", key="user_answer_key", height=130)

col_act1, col_act2 = st.columns(2)

with col_act2:
    if st.button("💡 모범 답안(원문 각 호) 바로 확인 (0초)", use_container_width=True):
        st.session_state.show_answer = True
        st.session_state.feedback = ""

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
                    [출제 조항]: {q_text}
                    [사용자 답변]: {user_answer_input}
                    [DB 법령 원문 정답]:
                    {a_text[:2000]}

                    [채점 규칙]:
                    원문의 수치, 키워드, 각 호 항목이 정확히 들어갔는지 엄격히 평가하세요.

                    [출력 양식]:
                    1. 출처 근거: {q_text[:50]}
                    2. 결과: (합격/불합격/보완필요)
                    3. 점수: (0~100점)
                    4. 핵심 피드백: (수치/단어 누락 지적 1~2문장)
                    5. 모범 답안: (DB 법령 원문 정답 100% 그대로 기술)
                    """

                    eval_response = client.chat.completions.create(
                        model="meta/llama-3.1-70b-instruct",
                        messages=[{"role": "user", "content": eval_prompt}],
                        temperature=0.0,
                        max_tokens=700
                    )
                    st.session_state.feedback = eval_response.choices[0].message.content
                    st.session_state.show_answer = False
                    st.rerun()
            except Exception as err:
                st.error(f"채점 중 오류가 발생했습니다: {err}")

# 모범 답안 원문 표시 (24px - 20% 보정 폰트 적용)
if st.session_state.show_answer:
    st.divider()
    st.subheader(f"📖 [모범 답안 원문] 세부 각 호 항목 전체")
    
    a_html = f"""
    <div style="
        background-color: #e8f8f5; 
        border-left: 6px solid #27ae60; 
        padding: 20px; 
        border-radius: 8px; 
        font-family: sans-serif;
        color: #145a32;">
        <div style="font-size: 24px; line-height: 1.7; font-weight: 600; white-space: pre-wrap; color: #111;">
            {a_text}
        </div>
    </div>
    """
    a_box_height = max(400, int(len(a_text) * 0.9) + 160)
    components.html(a_html, height=a_box_height, scrolling=True)
    
    st.write("")
    if st.button("🔊 모범 답안 음성으로 듣기", use_container_width=True):
        clean_ans = a_text.replace("*", "").replace("#", "").replace("-", "").replace("`", "")
        speak_js(clean_ans)

if st.session_state.feedback:
    st.divider()
    st.subheader("📊 AI 채점 결과")
    st.markdown(st.session_state.feedback)
    
    if st.button("🔊 채점 결과 음성으로 듣기", use_container_width=True):
        clean_fb = st.session_state.feedback.replace("*", "").replace("#", "").replace("-", "").replace("`", "")
        speak_js(clean_fb)
