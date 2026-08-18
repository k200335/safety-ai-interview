import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

def build_or_update_db():
    print("1. 'data' 폴더에서 PDF 파일 읽는 중...")
    
    # data 폴더 확인
    if not os.path.exists("./data") or not os.listdir("./data"):
        print("❌ 오류: 'data' 폴더가 없거나 PDF 파일이 없습니다. data 폴더에 PDF를 넣어주세요.")
        return

    # PDF 읽기
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()
    print(f"   -> 총 {len(documents)}페이지의 문서를 읽었습니다.")

    print("2. 텍스트 조각으로 나누는 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    print(f"   -> 총 {len(docs)}개의 텍스트 조각으로 나누었습니다.")

    print("3. AI 검색용 Vector DB(ChromaDB) 저장 중...")
    # 한국어 처리에 특화된 무료 임베딩 모델 (최초 실행 시 자동 다운로드)
    embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    # chroma_db 폴더에 데이터베이스 파일 생성 및 저장
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory="./chroma_db"
    )
    print("✅ 성공! 'chroma_db' 폴더에 데이터베이스가 잘 생성되었습니다.")

if __name__ == "__main__":
    build_or_update_db()