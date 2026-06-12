# 🦽 Barrier-Free Navigator

> 교통약자를 위한 편의시설 추천 및 접근 경로 탐색 서비스

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

<br>

## 소개

휠체어 이용자·노약자·고령자·일반인이 **지금 실제로 갈 수 있는 곳**을 찾아주는 서비스입니다.

단순 최단거리 검색이 아닌, 장애인 화장실·엘리베이터·경사로 보유 현황을 점수화해 목적지를 추천하고 실제 보행자 도로를 따라 경로를 안내합니다.

<br>

## 주요 기능

### 🗺️ 편의시설 검색 탭

| 기능 | 설명 |
|---|---|
| 🔍 **키워드 장소 검색** | 카카오 API 기반 출발지 검색 및 선택 |
| 🏆 **편의시설 추천** | 유형별 가중치로 TOP 10 시설 추천 |
| 📊 **점수 분해 차트** | 거리·화장실·엘리베이터·경사로·주차 항목별 점수 기여도 바 차트 시각화 |
| 🗂️ **카테고리 필터** | 의료·복지·공공·교육·생활·문화 6개 카테고리로 시설 분류 필터링 |
| 🗺️ **실제 보행 경로** | OSMnx 보행자 도로망 기반 경로 (산길·등산로 자동 회피, 실패 시 직선 대체) |
| ♿ **휠체어 모드** | 계단 완전 배제 + 엘리베이터·경사로가 끊기지 않는 접근 가능 경로만 탐색 |
| ⏱️ **예상 소요시간** | 이용자 유형별 보행 속도 기반 ETA (휠체어 모드: 엘리베이터 대기시간 포함) |

### 🧭 길찾기 탭

| 기능 | 설명 |
|---|---|
| 📍 **출발지·목적지 검색** | 카카오 API 기반 두 지점 키워드 검색 |
| 🚶 **OSMnx 보행 경로** | 알고리즘 경유지를 OSMnx 도로망에 실제 반영한 경로 안내 |
| 🏢 **보행 인프라 마커** | OSM Overpass API로 경로 주변 엘리베이터·보도턱·경사로 위치 표시 |
| ⚠️ **오류 원인 표시** | 경로 탐색 실패 시 원인 상세 메시지 표시 및 재시도 지원 |

<br>

## 이용자 유형

| 유형 | 특징 |
|---|---|
| 🚶 일반 | 거리 중심 추천, 일반 보행 속도 (5 km/h) |
| ♿ 휠체어 사용자 | 엘리베이터·경사로 가중치 높음, 접근 가능 경로만 탐색 (3 km/h) |
| 👴 노약자·고령자 | 화장실·엘리베이터 가중치 높음, 느린 보행 속도 (3.3 km/h) |

<br>

## 기술 스택

- **Frontend** — Streamlit, Folium, streamlit-folium
- **경로 탐색** — OSMnx, NetworkX, heapq, collections
- **데이터** — requests, python-dotenv
- **공간 인덱스** — scipy KD-Tree (설치 시 자동 활성화)

<br>

## 시작하기

### 1. 설치

```bash
git clone https://github.com/사용자명/barrier-free-navigator.git
cd barrier-free-navigator
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 키를 입력합니다.

```env
DATA_GO_KR_KEY=공공데이터포털_인증키
KAKAO_REST_KEY=카카오_REST_API_키
```

| API | 발급처 | 비고 |
|---|---|---|
| 장애인편의시설 현황 | [공공데이터포털](https://www.data.go.kr) → 활용신청 | 무료 |
| 카카오 로컬 API | [카카오 개발자센터](https://developers.kakao.com) → 앱 생성 | 카카오맵 활성화 필수 |

### 3. 실행

```bash
streamlit run app.py
```

<br>

## 배포 (Streamlit Community Cloud)

1. [share.streamlit.io](https://share.streamlit.io) 접속 후 GitHub 연결
2. 저장소·브랜치·`app.py` 선택
3. **Advanced settings → Secrets** 에 API 키 입력

```toml
DATA_GO_KR_KEY = "공공데이터포털_인증키"
KAKAO_REST_KEY = "카카오_REST_API_키"
```

<br>

## 팀원별 역할

| 팀원 | 파일 | 담당 기능 | 자료구조 | 알고리즘 |
|---|---|---|---|---|
| 이승주 (조장) | `module1_data.py` | 데이터 수집·정규화·필터링·지역명 자동완성 | List, Dict, Trie, Set | 선형탐색, 조건 필터링, 접두사 탐색(DFS) |
| 조병호 | `module2_path.py` | 거리 계산·그래프 구성·최단경로·보행 경로 | Graph (dict of dict), Priority Queue (heapq), KD-Tree | Haversine, Dijkstra, A* |
| 강민구 | `module3_score.py` | 편의시설 점수화·추천 | Heap (heapq), Dict, Set | Greedy 가중 점수화, Greedy Set Cover, Heap Sort (Top-N) |
| 변정인 | `module4_access.py` + `app.py` | 접근 경로 탐색·UI 렌더링 | Queue (deque), Set, Priority Queue (heapq) | BFS 제약 탐색, 접근성 제약 Dijkstra |

<br>

## 한계

- 시설 간 그래프는 실제 도로망이 아닌 반경 기반 연결 → 보행 경로와 차이 있을 수 있음
- OSMnx 보행 경로는 네트워크 상태에 따라 계산 실패 시 직선 경로로 대체됨
- 편의시설 보유 정보는 등록 기준이라 현장 상태와 다를 수 있음
- Streamlit Community Cloud 재시작 시 API 캐시 초기화됨
