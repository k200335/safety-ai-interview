import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 임베딩 모델 로드
embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")

data_dir = "./data"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# ChromaDB 컬렉션 이름 규칙(영문/숫자/_/- 만 허용)에 맞게 변환하는 함수
def clean_collection_name(name):
    # 알파벳, 숫자, 언더스코어, 하이픈만 남기고 변환
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    return cleaned[:63] # 영문 기준 최대 길이 제약 대응

print("🚀 과목별 독립 ChromaDB 컬렉션 구축 시작...")

for file in os.listdir(data_dir):
    if file.endswith(".pdf"):
        file_path = os.path.join(data_dir, file)
        subject_name = os.path.splitext(file)[0] # 예: "4. 가설공사 표준안전 작업지침"
        
        # 컬렉션 이름 영문/숫자 형태로 정제
        collection_name = f"sub_{file.split('.')[0].strip()}" # 예: "sub_4"
        
        print(f"📦 처리 중: [{subject_name}] -> Collection: [{collection_name}]")
        
        # PDF 데이터 로드 및 분할
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        
        # 과목별 독립 Collection으로 각각 저장
        Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )

print("✅ 모든 과목이 개별 DB/Collection으로 완전히 분리 저장되었습니다!")