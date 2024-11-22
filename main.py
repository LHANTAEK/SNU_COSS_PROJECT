import os
from typing import TypedDict, List
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt
from langchain_upstage import UpstageGroundednessCheck
import random

# 환경 변수 로드 및 API 키 설정
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY")
upstage_ground_checker = UpstageGroundednessCheck()


# 데이터 및 모델 로드
with open('processed_df.pkl', 'rb') as f:
    df = pd.read_pickle(f)
with open('embeddings.pkl', 'rb') as f:
    embeddings = pd.read_pickle(f)
# index = faiss.read_index('faiss_index.bin')
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
okt = Okt()


# GraphState 정의
class GraphState(TypedDict):
    question: str
    answer: str
    context: str
    relevance: str
    iteration: int
    dataframe: pd.DataFrame


# 무작위로 행을 선택하는 함수
def select_diverse_context():
    # 다양한 질문을 생성하기 위해서 1~3개의 랜덤한 수의 행을 선택
    # 1개의 경우 나올 수 있는 질문이 제한적일 수 있으므로 2~3개의 행도 랜덤하게 선택
    k = random.randint(1, 3)
    return df.sample(n=k)


# 노드 함수들
def generate_question(state: GraphState) -> GraphState:
    context = select_diverse_context()
    context_text = "\n".join(context['과목명'].astype(str) + ": " + context['강의개요'].astype(str))
    
    prompt = f"""
    당신은 대학 강의에 대해 학생들이 할 법한 질문을 생성하는 AI 에이전트입니다. 
    '카테고리'와 '학문 분야'에서 각각 하나를 무작위로 선택하세요. 
    그리고 '주의사항'을 참고하고 포함될 맥락(context_text)을 고려해서 하나의 질문만을 생성해주세요.
    그 질문은 '학문 분야'의 학생이 '카테고리'와 같은 관심사를 가지고 있을 때 궁금해할 만한 것이어야 합니다.
    
    '카테고리':
    1. 기수강 강의 정보 관련
    2. 사전 지식 및 보유 기술 관련
    3. 강의 간 비교
    4. 얻을 수 있는 역량 및 지식
    5. 커리어 및 커리큘럼 설계 관련
    6. 단순 정보 요약

    '학문 분야':
    1. 이공계 (예: 컴퓨터공학, 물리학, 생명과학 등)
    2. 인문사회계 (예: 철학, 역사학, 사회학 등)
    3. 상경계 (예: 경영학, 경제학, 회계학 등)
    4. 어문계 (예: 영어영문학, 국어국문학, 독어독문학 등)
    5. 예체능계 (예: 음악, 미술, 체육 등)
    6. 의학계 (예: 의학, 간호학, 약학 등)
    7. 농수해양계 (예: 식품영양학, 수산학, 환경보건학 등)
    8. 사범계 (예: 초등교육, 특수교육학 등)
    9. 사회복지계 (예: 사회복지학, 노인복지학, 아동가족복지학 등)


    {context_text}

    '주의사항':
    1. 이전의 맥락과는 다른, 새롭고 다양한 질문을 생성해주세요.
    2. "이 강의가", "여기서는" 등의 모호한 표현 대신 구체적인 과목명이나 분야를 사용하세요.
    3. 구어체로 자연스럽게 질문을 생성하세요.
    4. 질문에는 '카테고리'와 '학문 분야'가 드러나지 않도록 합니다.

    예시: 
    1. 통계학과에서 통계수학 1을 안 듣고 통계수학 2부터 들어도 될까요?
    2. 경영학과와 경제학과에서 배우는 과목은 어떤 점에서 차이가 있나요?
    3. 컴퓨터공학과에 입학했는데 1학년 때부터 코딩을 배우나요?
    4. 데이터사이언티스트를 목표로 하는 학생에게 추천할 만한 통계학과 강의가 있을까요?

    이전 질문 (있다면): {state["question"]}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    
    new_question = response.choices[0].message.content.strip()
    return GraphState(**{**state, "question": new_question, "context": context_text, "iteration": state["iteration"] + 1})


def answer_question(state: GraphState) -> GraphState:
    prompt = f"""
    당신은 대학 강의에 대한 학생들의 질문에 답변하는 AI 어시스턴트입니다. 
    
    다음 정보를 바탕으로 질문에 답해주세요:

    {state["context"]}

    질문: {state["question"]}

    주어진 정보를 바탕으로 정확하고 간결하게 줄글로 답변해주세요.

    주어진 정보에서 답변하기 어려운 경우에는 인터넷 검색을 통해 추가 정보를 추가해서 답변해도 됩니다.

    하지만 답변의 길이는 공백포함 400자를 넘지 않아야 합니다.


    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    
    new_answer = response.choices[0].message.content.strip()
    return GraphState(**{**state, "answer": new_answer})


def check_groundedness(state: GraphState) -> GraphState:
    result = upstage_ground_checker.run({
        "context": state["context"],
        "answer": state["answer"]
    })
    return GraphState(**{**state, "relevance": result})


def update_dataframe(state: GraphState) -> GraphState:
    if state["relevance"] == "grounded":
        new_row = pd.DataFrame({
            "Question": [state["question"]],
            "Answer": [state["answer"]]
        })
        updated_df = pd.concat([state["dataframe"], new_row], ignore_index=True)
        return GraphState(**{**state, "dataframe": updated_df})
    return state


def should_continue(state: GraphState) -> str:
    if state["dataframe"].shape[0] >= 1000 or state["iteration"] >= 10000:
        return "end"
    if state["relevance"] == "grounded":
        return "continue"
    return "generate_question"


# 그래프 생성
workflow = StateGraph(GraphState)


# 노드 추가
workflow.add_node("generate_question", generate_question)
workflow.add_node("answer_question", answer_question)
workflow.add_node("check_groundedness", check_groundedness)
workflow.add_node("update_dataframe", update_dataframe)


# 엣지 추가
workflow.add_edge("generate_question", "answer_question")
workflow.add_edge("answer_question", "check_groundedness")
workflow.add_edge("check_groundedness", "update_dataframe")


# 조건부 엣지 추가
workflow.add_conditional_edges(
    "update_dataframe",
    should_continue,
    {
        "continue": "generate_question",
        "end": END,
        "generate_question": "generate_question"
    }
)


# 시작점 설정
workflow.set_entry_point("generate_question")


# 그래프 컴파일
app = workflow.compile()


# 그래프 실행
initial_state = GraphState(
    question="",
    answer="",
    context="",
    relevance="",
    iteration=0,
    dataframe=pd.DataFrame(columns=["Question", "Answer"])
)


config = RunnableConfig(recursion_limit=12000)  # 최대 내부 루프 최대 12000회 반복
result = app.invoke(initial_state, config=config)


# 결과 저장
final_df = result["dataframe"]
final_df.to_csv("course_qa_results.csv", index=False)
print(f"결과가 course_qa_results.csv 파일에 저장되었습니다. 총 {result['dataframe'].shape[0]}개의 Q&A 쌍이 생성되었습니다.")
