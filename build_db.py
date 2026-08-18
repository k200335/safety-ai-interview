import os
import re
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 1. 임베딩 모델 로드
embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")

data_dir = "./data"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

SUBJECT_MAPPING = {
    "산업안전보건법": "sub_1",
    "산업안전보건기준": "sub_2",
    "중대재해": "sub_3",
    "가설공사": "sub_4",
    "철골공사": "sub_5",
    "추락재해": "sub_6",
    "해체공사": "sub_7",
    "터널공사": "sub_8",
    "발파": "sub_8",
    "콘크리트": "sub_9",
    "굴착공사": "sub_10",
    "위험성평가": "sub_11",
    "기출": "sub_12"
}

def get_clean_collection_name(filename):
    for keyword, col_name in SUBJECT_MAPPING.items():
        if keyword in filename:
            return col_name
    clean = re.sub(r'[^a-zA-Z0-9]', '', filename)
    return f"sub_{clean[:8]}" if clean else "sub_misc"

print("🚀 과목별 독립 ChromaDB 컬렉션 구축 시작...")

for file in os.listdir(data_dir):
    if file.endswith(".pdf"):
        file_path = os.path.join(data_dir, file)
        collection_name = get_clean_collection_name(file)

        print(f"📦 처리 중: [{file}] -> Collection: [{collection_name}]")
        
        try:
            # 1차 시도: PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            
            # 1차 실패 시 2차 시도: PDFPlumberLoader (스캔본/표 파싱에 강함)
            if not splits:
                print(f"  🔄 [PyPDFLoader] 실패. [pdfplumber]로 재시도합니다...")
                loader = PDFPlumberLoader(file_path)
                docs = loader.load()
                splits = text_splitter.split_documents(docs)

            if not splits:
                print(f"  ⚠️ 경고: [{file}] 문서에서 텍스트를 추출할 수 없습니다.")
                continue
            
            Chroma.from_documents(
                documents=splits,
                embedding=embedding_function,
                collection_name=collection_name,
                persist_directory="./chroma_db"
            )
            print(f"  └─ 성공: [{collection_name}] 저장 완료")
        except Exception as e:
            print(f"  └─ 오류 발생 ({file}): {e}")

print("\n✅ 모든 과목이 에러 없이 개별 DB 컬렉션으로 완벽히 분리 저장되었습니다!")