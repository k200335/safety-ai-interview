import os
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 간단한 Document 클래스 정의 (langchain.schema 의존성 제거)
class Document:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {}

PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"

embedding_fn = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 실제 data 폴더 내 파일명과 DB 컬렉션 키 매핑
FILE_MAP = {
    "sub_1": "산업안전보건법.txt",
    "sub_2": "산업안전보건법 시행령.txt",
    "sub_3": "산업안전보건법 시행규칙.txt",
    "sub_4": "산업안전보건기준에 관한 규칙.txt",
    "sub_5": "중대재해 처벌 등에 관한 법률.txt",
    "sub_6": "중대재해 처벌 등에 관한 법률 시행령.txt",
    "sub_7": "가설공사 표준안전 작업지침.txt",
    "sub_8": "철골공사표준안전작업지침.txt",
    "sub_9": "추락재해방지표준안전작업지침.txt",
    "sub_10": "해체공사표준안전작업지침.txt",
    "sub_11": "터널공사 표준안전 작업지침_NATM공법.txt",
    "sub_12": "콘크리트공사 표준안전 작업지침.txt",
    "sub_13": "운반하역 표준안전 작업지침.txt",
    "sub_14": "발파 표준안전 작업지침.txt",
    "sub_15": "사업장 위험성평가에 관한 지침.txt"
}

def parse_law_by_articles(text):
    """
    텍스트를 글자 수가 아닌 '제N조' 문두 기준으로 통째 분할하는 파서
    """
    article_pattern = re.compile(r'(?=\n제\s*\d+\s*조|\n제\s*\d+\s*절|^제\s*\d+\s*조)')
    blocks = article_pattern.split(text)
    
    docs = []
    for block in blocks:
        clean_block = block.strip()
        if clean_block:
            docs.append(Document(page_content=clean_block))
    return docs

def rebuild_database():
    print("🧹 기존 Chroma DB 재구축 시작...")
    
    for collection_key, file_name in FILE_MAP.items():
        file_path = os.path.join(DATA_DIR, file_name)
        
        if not os.path.exists(file_path):
            print(f"⚠️ 경고: {file_name} 파일이 data 폴더에 없습니다. 건너끁니다.")
            continue
            
        print(f"📦 [{file_name}] DB 생성 중...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        documents = parse_law_by_articles(raw_text)
        
        db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_fn,
            collection_name=collection_key,
            persist_directory=PERSIST_DIR
        )
        db.persist()
        print(f"✅ [{file_name}] 총 {len(documents)}개 조항 저장 완료!")

    print("\n🎉 모든 법령 및 지침 DB 재구축이 완료되었습니다!")

if __name__ == "__main__":
    rebuild_database()