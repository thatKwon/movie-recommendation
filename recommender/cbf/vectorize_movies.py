import pandas as pd
import numpy as np
import re as re
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

#로컬 DB 연결
client = MongoClient("mongodb://localhost:27017")
####################################################
#print(client.list_database_names())
mydb = client["recsys_dev"]

#데이터를 list 형태로 변환 후 pandas dataframe으로 변환
cursor = mydb.movies.find()
movies = pd.DataFrame(list(cursor))

cursor = mydb.peoples.find()
peoples = pd.DataFrame(list(cursor))
####################################################
#print(mydb.list_collection_names())
#print(movies.head())
#print(peoples.head())

#v를 리스트 형태로 반환
def to_list(v):
    if isinstance(v, list): return v
    if v is None or (isinstance(v, float) and pd.isna(v)): return []
    return [v]

#seq(리스트)를 하나의 문자열로 합침 ex) ["ABC", None, "DE"] -> "ABC DE" 
def safe_join(seq):
    return " ".join(str(s) for s in seq if s)

"""
peoples id와 이름 매핑해주는 딕셔너리
ex) 1824 : "송강호",
    1898 : "이병헌", ...
"""
id2name = {row["_id"]: (row.get("korean") or row.get("original") or "") for _, row in peoples.iterrows()}

#사람ID(리스트)를 받아 이름 문자열로 변경 ex) ids = [1824, 1898] -> ["송강호", "이병헌"]
def get_cast_names(ids):
    ids = to_list(ids)
    return " ".join(id2name.get(i, "") for i in ids if i in id2name)
####################################################
#sample_ids = [list(id2name.keys())[0], list(id2name.keys())[1]]
#print(get_cast_names(sample_ids))

#한 행에서 핵심 정보들을 모아 하나의 문자열로 만드는 함수. 벡터화(TF-IDF)의 입력으로 사용하기 위한 전처리
def build_text(x):
    try:
        parts = [
            x.get('title', '') or '', #title이 None, NaN일 때도 ''로 처리됨
            x.get('title_eng', '') or '',
            safe_join(to_list(x.get('genres'))), 
            safe_join(to_list(x.get('countries'))),
            get_cast_names(x.get('main_cast_people_ids')),
            get_cast_names(x.get('support_cast_people_ids')),
        ]
        text = " ".join(p for p in parts if p)  # parts 중 빈 문자열 제거하고 연결
        return re.sub(r'\s+', ' ', text).strip() #앞뒤 공백 제거, 연속된 공백을 하나의 공백으로 바꿈
    except Exception as e:
        print(f"build_text() 중 오류: {e}")
        return ""

movies["text"] = movies.apply(build_text, axis=1) 

try:
    movies["text"] = movies.apply(build_text, axis=1) #새 칼럼 생성
    print("텍스트 전처리 완료")
except Exception as e:
    print(f"텍스트 전처리 중 오류 발생: {e}")

####################################################
#print(movies.iloc[0])
#print(build_text(movies.iloc[0]))

# text를 벡터화해 TF-IDF 행렬 생성
X = None
try:
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(movies["text"])
    print("TF-IDF 벡터화 완료")
except Exception as e:
    print(f"TF-IDF 벡터화 중 오류 발생: {e}")



# TF-IDF 벡터를 MongoDB에 저장
mydb.embeddings.drop()
for i, row in tqdm(movies.iterrows(), total=len(movies)):
    mydb.embeddings.insert_one({
        "_id": int(row["_id"]),
        #한 영화의 TF-IDF 벡터를 MongoDB에 저장 가능한 형태(list)로 바꿈
        "vector": X[i].toarray().tolist()[0]
    })

# TF-IDF 벡터를 MongoDB에 저장
try:
    mydb.embeddings.drop()
    print("embeddings 초기화 완료")

    for i, row in tqdm(movies.iterrows(), total=len(movies)):
        try:
            mydb.embeddings.insert_one({
                "_id": int(row["_id"]),
                "vector": X[i].toarray().tolist()[0] #한 영화의 TF-IDF 벡터를 MongoDB에 저장 가능한 형태(list)로 바꿈
            })
        except Exception as inner_e:
            print(f"영화 ID {row.get('_id')} 삽입 중 오류: {inner_e}")
            continue
    print("콘텐츠 필터링: 벡터화 및 저장 완료")
except Exception as e:
    print(f"MongoDB에 벡터 저장 중 오류 발생: {e}")

client.close()