import pandas as pd
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt
import pickle



# CSV 파일 로드
df = pd.read_csv('data.csv', encoding='utf-8', low_memory=False)


# 한국어 전처리
okt = Okt()


# 텍스트 전처리 함수
def preprocess_text(text):
    return ' '.join(okt.nouns(text) + okt.phrases(text))


# 문서 임베딩
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')


# 데이터프레임의 모든 열을 문자열로 결합하고 전처리
df['combined'] = df.apply(lambda row: preprocess_text(' '.join(row.astype(str))), axis=1)


# 임베딩 생성
embeddings = model.encode(df['combined'].tolist())


# 전처리 데이터프레임 저장
with open('processed_df.pkl', 'wb') as f:
    pickle.dump(df, f)


# 임베딩 저장
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)


print("모든 데이터가 성공적으로 저장되었습니다")