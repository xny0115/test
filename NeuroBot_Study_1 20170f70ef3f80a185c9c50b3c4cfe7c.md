# NeuroBot_Study_1

**프로젝트명**: NeuroBot_Study_1

**버전**: v0.1

**목적**:
신경망(MLP), 딥러닝, 자가학습 구조에 대한 개념을 실전 구현을 통해 체득하기 위한 학습 중심 챗봇 개발 프로젝트.
또한, 초기에는 유사도 기반 선택형 구조로 시작하지만, 추후에는 디코더 기반 생성형 챗봇으로 확장 가능한 구조를 목표로 한다.

---

**1. 개발 개요**

- **개발 언어**: Python
- **기반 프레임워크**: PyTorch
- **인터페이스**: Streamlit 기반 GUI (웹 인터페이스 형태)
- **모델 구조**: MLP (Multilayer Perceptron) → 추후 디코더 구조로 확장 예정
- **응답 방식**: 초기에는 신경망 기반 유사도 매칭 (벡터 검색 기반 선택 응답), 이후 생성 기반 디코더 방식으로 진화 가능
- **초기 학습자료**: 수작업 QA 200~500쌍 이상 확보 예정 (또는 범용 공개 QA 데이터셋 병행 활용)
- **임베딩 모델**: Sentence-BERT 기반 한국어 특화 모델 사용 (예: `jhgan/ko-sbert-sts`, `snunlp/KR-SBERT-V40K-klueNLI`)

---

**2. 모델 아키텍처 (MLP_v0.1)**

- 입력 차원: 768 (문장 임베딩)
- 은닉층 1: 2048 노드
- 은닉층 2: 1024 노드
- 출력 차원: 768
- 총 파라미터 수: 약 4,460,288개
- 활성화 함수: ReLU
- 출력 처리: 임베딩 벡터를 응답 풀 내 벡터들과 유사도 비교하여 가장 가까운 응답을 선택 (생성이 아닌 분류/검색 구조)
- ※ 생성형 확장 시 출력층 구조 및 활성화 함수(softmax 등) 재설계 예정

---

**3. 학습 구조**

- **초기 학습**: 수작업 QA 데이터셋 기반 사전 학습 (또는 공개 소규모 대화 데이터셋 활용)
- **대화 중 자동 저장**: 사용자의 입력 및 응답 로그 저장
- **수동 학습 트리거**: "학습해" 버튼을 통해 모델 재훈련
- **학습 데이터 필터링**: 사용자 피드백(좋아요/싫어요, 학습 제외 체크 등)을 통해 학습 데이터 질을 수동 제어 가능
- **학습 로그 및 상태 정보 출력**: 학습 횟수, 입력 수, 최근 손실(loss) 등
- **향후 확장성 고려**: 기존 학습 데이터셋은 생성형 구조로 전환 시에도 그대로 재사용 가능 (단, 모델 파라미터는 재훈련 필요)
- **학습률 스케줄링**: 초기에는 고정 학습률 사용, 추후 학습률 감소 스케줄링 적용 예정 (예: `StepLR`, `ReduceLROnPlateau`)

---

**4. GUI 구성 요소 (Streamlit)**

- 텍스트 입력창 (사용자 입력)
- 텍스트 출력창 (모델 응답)
- 상태 패널 (입력 수, 학습 횟수 등 표시)
- 학습 버튼 (수동 학습 트리거)
- 진행률 바 또는 상태 텍스트 (학습중 상태 표시)
- 선택 가능한 모델 구조 옵션 (추후 확장용)
- 대화 학습 여부 선택 토글 (좋아요/싫어요, 학습 예외 체크 등)

---

**5. 향후 확장 방향 (v0.2~)**

- 은닉층 수 확대 (최대 4~6층)
- 노드 수 증가 (2048~8192)
- Dropout, BatchNorm 추가
- Embedding 유사도 회수 기능 도입 (임베딩 기반 벡터 저장 및 검색 구조 통합)
- 자가 평가 루프: 손실값 추적 기반 학습 리텐션 알고리즘 추가 (예: 일정 정확도 이상일 때만 데이터 저장)
- ✅ **디코더 기반 생성 구조로 전환**: 입력 문장 기반 단어 시퀀스를 생성하는 신경망 구조로의 확장
- 출력 구조 변경: 디코더 구조에 적합한 출력층 및 softmax 기반 토큰 예측 구조 도입 예정
- 학습률 스케줄링 기능 본격 적용 및 실험 (동적 학습률 조절)

---

**비고**

- 초기 버전에서는 Rule 기반 응답은 완전히 배제됨
- 완전 신경망 기반 챗봇의 의미 해석과 구조 실습이 핵심 목적임
- 학습 로그, 학습률, 손실값 변화 등을 실시간 관찰 가능하도록 설계됨
- 데이터 품질 관리를 위한 사용자 상호작용 요소 추가 예정 (좋아요/싫어요, 학습 필터)
- 향후 구조 확장 시 기존 데이터셋 재활용 가능하도록 설계됨

[NeuroBot_Study_1] 개발 단계별 작업지시서 (v0.2 - GPT 최적화 세분화 버전)

■ 목적: 신경망(Multi-layer Perceptron)을 기반으로 유사도 매칭 챗봇을 구현하며, GUI 및 자가학습 구조를 포함하는 학습 중심 프로젝트. GPT 세션 제한을 고려하여 각 작업을 완결된 단위로 세분화.

■ 전체 개발 흐름 요약:

1단계: 초기 프로젝트 환경 설정
2단계: MLP 신경망 모델 핵심 구현
3단계: 학습 데이터 준비 및 관리 시스템 구축
4단계: PyTorch 학습 루프 및 모델 관리 구현
5단계: Streamlit GUI 기본 골격 및 입력/출력 인터페이스 구현
6단계: Sentence-BERT 기반 응답 유사도 비교 로직 구현
7단계: 사용자 피드백 기반 자가학습 및 재훈련 기능 구현
8단계: 향후 생성형 챗봇 확장 고려한 구조적 기반 마련

---

■ 단계별 작업 내용 (세분화):

(각 항목 완료 시 [ ] → [x] 로 체크)

---

[1단계] 초기 프로젝트 환경 설정

- 책임자: 미정 / 완료 예정: XXXX-MM-DD
    - [ ]  1.1 프로젝트 최상위 디렉터리 생성: NeuroBot_Study_1
    - [ ]  1.2 주요 모듈 디렉터리 및 파일 생성:
        - [main.py](http://main.py/) (메인 실행 스크립트)
        - [model.py](http://model.py/) (신경망 모델 정의)
        - [data.py](http://data.py/) (학습 데이터 처리 및 관리)
        - [ui.py](http://ui.py/) (Streamlit GUI 관련 코드)
        - [utils.py](http://utils.py/) (공통 유틸리티 함수, 예: 임베딩 로드)
    - [ ]  1.3 config.json 설정 파일 템플릿 생성 및 초기값 정의:
        - 기본 경로 설정 (데이터, 모델 저장 위치)
        - 학습률 (learning_rate), Epoch 수 (num_epochs) 등 초기 학습 파라미터
        - GPT 요청: "Python 프로젝트에서 설정값 관리에 사용되는 `config.json` 파일의 기본적인 템플릿과 로드하는 파이썬 코드 예시를 알려줘."
    - [ ]  1.4 .gitignore 파일 생성 및 파이썬, PyTorch, Streamlit 프로젝트에 필요한 기본 제외 목록 추가.
        - GPT 요청: "PyTorch, Streamlit, Python 프로젝트에서 보통 Git으로 관리하지 않는 파일들을 `.gitignore`에 추가해줘."
    - [ ]  1.5 requirements.txt 파일 생성 및 필요한 라이브러리 목록 (PyTorch, transformers, streamlit, sentence-transformers 등) 명시.
        - GPT 요청: "내가 PyTorch, Streamlit, Sentence-Transformers를 사용하는 Python 프로젝트를 시작하려고 해. `requirements.txt`에 어떤 라이브러리들을 어떤 버전으로 명시하면 좋을까?"

[2단계] MLP 신경망 모델 핵심 구현 ([model.py](http://model.py/))

- 책임자: 미정 / 완료 예정: XXXX-MM-DD
    - [ ]  2.1 PyTorch nn.Module을 상속받는 MLP 클래스 정의:
        - **init** 함수에서 입력 768 → 은닉 2048 → 은닉 1024 → 출력 768의 Linear 레이어 및 ReLU 활성화 함수 정의.
        - GPT 요청: "PyTorch의 `nn.Module`을 상속받아 `MLP` 클래스를 정의해줘. 입력 차원은 768, 첫 번째 은닉층은 2048, 두 번째 은닉층은 1024, 출력 차원은 768로 하고, 은닉층마다 `ReLU` 활성화 함수를 적용해줘."
    - [ ]  2.2 forward 함수 작성: 입력 텐서를 정의된 레이어들을 순서대로 통과시키는 로직 구현.
        - GPT 요청: "이전에 정의한 MLP 클래스에 `forward` 메서드를 추가해줘. `x`를 입력받아 정의된 Linear 레이어와 ReLU를 순서대로 거쳐 최종 출력을 반환하도록 해줘."
    - [ ]  2.3 모델 인스턴스 생성 후 print(model)을 통해 총 파라미터 수 확인 (약 4,460,288개 일치 여부).
        - GPT 요청: "정의된 `MLP` 클래스의 인스턴스를 생성하고, `print(model)`을 통해 모델의 구조와 총 파라미터 수를 확인하는 파이썬 코드를 작성해줘."
    - [ ]  2.4 GPU 사용 가능 여부 자동 감지 및 모델을 적절한 디바이스(CPU/CUDA)로 이동시키는 유틸리티 함수 구현 ([utils.py](http://utils.py/) 또는 [main.py](http://main.py/) 초기화 부분).
        - GPT 요청: "PyTorch에서 `torch.cuda.is_available()`을 사용하여 GPU 사용 가능 여부를 확인하고, 모델을 CPU 또는 CUDA 장치로 옮기는 `get_device()` 함수와 모델을 해당 장치로 옮기는 코드 예시를 알려줘."

[3단계] 학습 데이터 준비 및 관리 시스템 구축 ([data.py](http://data.py/))

- 책임자: 미정 / 완료 예정: XXXX-MM-DD
    - [ ]  3.1 QA 데이터셋 저장 포맷 정의: 질문-응답 쌍을 포함하는 JSON 파일 구조 설계 (data/qa_data.json 예시).
        - GPT 요청: "질문(question)과 응답(answer)을 키-값 쌍으로 가지는 챗봇용 QA 데이터셋을 저장할 JSON 파일의 예시 포맷을 보여줘."
    - [ ]  3.2 수작업 QA 데이터 200~500쌍 제작 및 data/qa_data.json 파일에 저장.
        - 이 단계는 수동 작업이므로, GPT에 코드 요청 대신 "일상 대화에서 챗봇 학습에 유용한 200~500쌍의 질문-응답 예시 아이디어를 줘" 와 같은 요청을 할 수 있습니다.
    - [ ]  3.3 PyTorch Dataset 클래스를 상속받는 QADataset 클래스 정의 ([data.py](http://data.py/)):
        - 초기화 시 JSON 파일에서 데이터를 로드.
        - **len** 및 **getitem** 메서드 구현.
        - GPT 요청: "PyTorch에서 JSON 파일로 저장된 질문-응답 데이터를 로드하여 사용할 수 있는 `Dataset` 클래스를 정의해줘. `__len__`과 `__getitem__` 메서드를 포함해야 해."
    - [ ]  3.4 대화 중 사용자 입력 및 모델 응답 로그를 자동 축적하는 함수 (save_dialogue_log) 정의:
        - 로그 파일 포맷 (JSON 또는 CSV) 설계 및 파일에 추가 저장.
        - GPT 요청: "사용자 질문과 챗봇 응답, 그리고 '좋아요/싫어요' 같은 피드백 상태를 저장할 수 있는 대화 로그 JSON 파일의 포맷을 설계하고, 새로운 대화를 추가하는 파이썬 함수 `save_dialogue_log`를 작성해줘."
    - [ ]  3.5 축적된 대화 로그 데이터를 불러오고 학습에 적합한 형태로 변환하는 함수 (load_and_process_logs) 구현.

[4단계] PyTorch 학습 루프 및 모델 관리 구현 ([main.py](http://main.py/) 또는 [train.py](http://train.py/))

- 책임자: 미정 / 완료 예정: XXXX-MM-DD
    - [ ]  4.1 QADataset 및 DataLoader를 사용하여 학습 데이터 준비.
        - GPT 요청: "PyTorch에서 `QADataset` 인스턴스를 사용하여 `DataLoader`를 생성하는 방법을 알려줘. 배치 사이즈는 16으로 하고, 데이터 셔플링을 적용해줘."
    - [ ]  4.2 손실 함수(Loss Function) 정의: nn.MSELoss 또는 nn.CosineEmbeddingLoss 등 적절한 손실 함수 선택 및 초기화.
        - GPT 요청: "PyTorch MLP 모델의 출력(임베딩 벡터)과 정답 임베딩 벡터 간의 유사도를 측정하기 좋은 손실 함수 추천해줘."
    - [ ]  4.3 옵티마이저(Optimizer) 정의: torch.optim.Adam 사용 및 모델 파라미터 연결.
    - [ ]  4.4 단일 Epoch 학습 루프 구현 (train_epoch 함수):
        - 데이터로더에서 배치 가져오기.
        - 입력 및 정답 임베딩을 디바이스로 이동.
        - 모델 forward 실행 및 출력 임베딩 계산.
        - 손실 계산 (loss.item()으로 값 확인).
        - optimizer.zero_grad(), loss.backward(), optimizer.step()으로 역전파 및 가중치 업데이트.
        - GPT 요청: "PyTorch에서 단일 에포크 학습 루프를 구현하는 함수 `train_epoch(model, dataloader, criterion, optimizer, device)`를 작성해줘. 손실값을 반환하도록 해줘."
    - [ ]  4.5 학습률 스케줄러 정의 및 적용 (torch.optim.lr_scheduler.StepLR 또는 ReduceLROnPlateau).
        - GPT 요청: "PyTorch `torch.optim.Adam` 옵티마이저와 함께 `torch.optim.lr_scheduler.StepLR`을 사용하여 학습률을 주기적으로 감소시키는 코드 예시를 보여줘."
    - [ ]  4.6 모델(state_dict()) 및 옵티마이저 상태를 저장/로드하는 함수 (save_model, load_model) 구현.
        - GPT 요청: "PyTorch 모델의 `state_dict()`와 옵티마이저의 상태를 `.pt` 파일로 저장하고, 나중에 이들을 다시 불러와 모델을 로드하는 파이썬 함수를 만들어줘."

[5단계] Streamlit GUI 기본 골격 및 입력/출력 인터페이스 구현 ([ui.py](http://ui.py/), [main.py](http://main.py/))

- 책임자: 미정 / 완료 예정: XXXX-MM-DD
    - [ ]  5.1 streamlit run main.py로 기본 Streamlit 앱 실행 환경 확인 및 초기 페이지 설정.
    - [ ]  5.2 텍스트 입력창 (st.text_input) 및 대화 내용 출력창 (st.empty(), st.write 활용) 구현.
        - GPT 요청: "Streamlit으로 챗봇 입력창과 응답 출력창을 만드는 기본적인 코드 알려줘."
    - [ ]  5.3 '학습 버튼' (st.button('학습해')) 생성.
    - [ ]  5.4 학습 진행률을 시각적으로 보여주는 '진행률 바' (st.progress) 또는 '학습중' 상태 텍스트 ([st.info](http://st.info/)) 구현.
        - GPT 요청: "Streamlit에서 학습 진행률을 보여주는 `st.progress` 바와 '학습 중...' 메시지를 표시하는 방법을 알려줘."
    - [ ]  5.5 현재 입력된 대화 수, 전체 학습 횟수, 최근 손실(loss) 등을 표시하는 '상태 패널' 구성.
    - [ ]  5.6 '좋아요/싫어요' 피드백 토글 (st.checkbox 또는 st.button 조합) 구현: 각 챗봇 응답 옆에 위치.
        - GPT 요청: "Streamlit으로 챗봇 응답 옆에 해당 응답에 대한 '좋아요' 또는 '학습 제외'를 표시할 수 있는 토글 버튼을 만드는 방법 알려줘."

[6단계] Sentence-BERT 기반 응답 유사도 비교 로직 구현 ([utils.py](http://utils.py/), [main.py](http://main.py/))

- 책임자: 미정 / 완료 예정: XXXX-MM-DD
    - [ ]  6.1 sentence-transformers 라이브러리를 사용하여 한국어 특화 임베딩 모델 로드 (예: jhgan/ko-sbert-sts).
        - GPT 요청: "`sentence_transformers` 라이브러리를 사용하여 `jhgan/ko-sbert-sts` 모델을 로드하고, 주어진 문장 리스트를 임베딩 벡터로 변환하는 코드 예시를 알려줘."
    - [ ]  6.2 모델 응답 풀 (저장된 QA 데이터셋의 응답 문장들) 임베딩 벡터로 변환 및 메모리 또는 파일에 캐싱 (response_embeddings.pt).
        - GPT 요청: "미리 정의된 응답 문장 목록을 `sentence-transformers` 모델로 임베딩하고, 이 임베딩 벡터들을 PyTorch `.pt` 파일로 저장하는 코드를 만들어줘."
    - [ ]  6.3 사용자 입력 문장을 실시간으로 임베딩 벡터로 변환하는 함수 구현.
    - [ ]  6.4 사용자 입력 임베딩과 캐싱된 응답 임베딩 벡터들 간의 코사인 유사도 (torch.nn.functional.cosine_similarity) 계산 함수 구현.
        - GPT 요청: "두 개의 임베딩 벡터 간의 코사인 유사도를 계산하는 PyTorch 함수 `calculate_cosine_similarity(vec1, vec2)`를 작성해줘."
    - [ ]  6.5 계산된 유사도 중 가장 높은 값의 응답 문장 인덱스를 찾아 해당 응답을 반환하는 로직 구현.

[7단계] 사용자 피드백 반영 + 재학습 구조 ([data.py](http://data.py/), [main.py](http://main.py/))

- 책임자: 미정 / 완료 예정: XXXX-MM-DD
    - [ ]  7.1 대화 로그(사용자 입력, 모델 응답, 피드백 상태)를 저장하는 함수 (save_dialogue_log)에 피드백 정보(liked, excluded_from_training) 추가 및 저장 로직 구현.
    - [ ]  7.2 '학습해' 버튼 클릭 시 트리거될 retrain_model 함수 정의:
        - 저장된 대화 로그 파일 로드.
        - '좋아요' 표시되었거나 '학습 예외'로 체크되지 않은 대화 쌍만 필터링.
        - 필터링된 대화 쌍을 임베딩 벡터로 변환하여 새로운 학습 데이터셋 구성.
        - GPT 요청: "대화 로그 JSON 파일에서 `liked`가 True이거나 `excluded_from_training`이 False인 대화 쌍만 필터링하여 새로운 질문-응답 리스트를 반환하는 파이썬 함수를 만들어줘."
    - [ ]  7.3 retrain_model 함수 내에서 모델 재훈련 루프 실행 (4단계에서 구현한 학습 루프 재활용).
    - [ ]  7.4 재학습 횟수, 새로 학습된 데이터 수, 재학습 후 손실값 변화 등을 GUI 상태 패널에 업데이트.

[8단계] 향후 생성형 확장 고려를 위한 기반 마련

- 책임자: 미정 / 완료 예정: XXXX-MM-DD
    - [ ]  8.1 [model.py](http://model.py/) 내 MLP 클래스에 주석으로 생성형 모델(TransformerDecoder 등)로 확장 시 필요한 **init** 및 forward 메서드의 변화 방향 간략히 설명.
    - [ ]  8.2 출력층에 softmax 활성화 함수를 적용할 수 있는 여지 마련 (예: forward 함수 내에 조건부 로직 또는 별도의 generate_output 함수).
        - GPT 요청: "PyTorch에서 `nn.Linear` 레이어의 출력에 `softmax`를 적용하여 확률 분포를 얻는 간단한 함수 예시를 보여줘. (생성형 모델 고려)"
    - [ ]  8.3 자가 평가 루프(손실값 추적 기반 학습 데이터 리텐션 알고리즘)를 위한 기본적인 함수 스켈레톤 작성 ([utils.py](http://utils.py/) 또는 [data.py](http://data.py/)):
        - filter_data_by_loss(data_pairs, model, threshold): 모델의 현재 손실값을 기반으로 학습 데이터를 선별하는 아이디어 주석으로 남기기.
        - GPT 요청: "PyTorch 딥러닝 모델의 손실값을 기반으로 학습 데이터를 필터링하는 파이썬 함수의 스켈레톤과 아이디어를 주석으로 설명해줘."

---

■ 완료 후 예상 출력:

- Streamlit GUI 기반의 신경망 기반 유사도 매칭 챗봇 ([main.py](http://main.py/) 실행 시 GUI 구동).
- 사용자 피드백('좋아요/싫어요', '학습 예외')에 따라 학습 데이터를 선별하고 모델을 재학습 가능한 구조.
- 학습 로그 및 상태 정보(입력 수, 학습 횟수, 손실값)를 실시간으로 표시.
- 향후 디코더 기반 생성형 챗봇으로의 확장 및 자가 평가 루프 구현을 위한 견고한 기반 확보.