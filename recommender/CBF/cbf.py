import sys
import json
import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

def main():
    #Node.js에서 전달된 영화 ID 받기
    if len(sys.argv) < 2:
        print(json.dumps({"error: movie_id 인자가 없습니다."}), file=sys.stderr)
        return

    try:
        movie_id = int(sys.argv[1])
    except ValueError:
        print(json.dumps({"error: movie_id가 정수가 아닙니다."}), file=sys.stderr)
        return

    #DB 연결
    client = MongoClient(
        "mongodb://localhost:27017",
        serverSelectionTimeoutMS=30000, 
        connectTimeoutMS=30000,
        socketTimeoutMS=120000,
        retryWrites=True,
        maxPoolSize=100
    )
    mydb = client["recsys_dev"]

    #전체 movieID를 ids 리스트에 등록
    ids = []
    vectors_cursor = mydb.embeddings.find({}, {"_id": 1}, batch_size=5000)
    for doc in vectors_cursor:
        ids.append(doc["_id"])

    if movie_id not in ids:
        print(json.dumps({f"error: movie_id {movie_id}를 찾을 수 없습니다."}), file=sys.stderr)
        client.close()
        return

    #대상 영화 벡터 가져오고 코사인 유사도 계산을 위해 형태변환
    target_doc = mydb.embeddings.find_one({"_id": movie_id}, {"vector": 1})
    if not target_doc:
        print(json.dumps({"error": "찾으려는 영화의 벡터가 없습니다."}), file=sys.stderr)
        client.close()
        return

    target_vec = np.array(target_doc["vector"], dtype=np.float32).reshape(1, -1)

    #전체 영화와의 유사도 계산 (batch 방식)
    sims = []
    cursor = mydb.embeddings.find({}, {"_id": 1, "vector": 1}, batch_size=5000)
    batch_ids = []
    batch_vecs = []

    for doc in cursor:
        batch_ids.append(doc["_id"])
        batch_vecs.append(doc["vector"])

        #batch 크기 도달 시 계산 수행
        if len(batch_vecs) >= 5000:
            X_batch = np.array(batch_vecs, dtype=np.float32)
            sim_batch = cosine_similarity(target_vec, X_batch)[0]
            sims.extend(list(zip(batch_ids, sim_batch))) #ID, 유사도 함께 저장 (영화ID, 유사도)
            batch_ids, batch_vecs = [], []  # 초기화

    #마지막 남은 batch 처리
    if batch_vecs:
        X_batch = np.array(batch_vecs, dtype=np.float32)
        sim_batch = cosine_similarity(target_vec, X_batch)[0]
        sims.extend(list(zip(batch_ids, sim_batch)))
    #sims = [(ID1, 유사도1), (ID2, 유사도2), ... , (IDN, 유사도N)]
    client.close()

    #결과 정렬 및 필터링
    sims = [(i, s) for i, s in sims if i != movie_id and s > 0.2] #유사도 0.2 이상만 필터링
    sims.sort(key=lambda x: x[1], reverse=True) #유사도 기준 내림차순 정렬
    top_similar = [{"id": int(i), "similarity": float(s)} for i, s in sims[:10]] #상위 10개만 선택, 딕셔너리 형태로 변환

    print(json.dumps({"movie_id": movie_id, "top_similar": top_similar}))

if __name__ == "__main__":
    main()
    