# app.py — 팀원 4
# Streamlit 메인 진입점: 전체 파이프라인 조립 & UI
# UI/UX 설계 원칙: 최소 입력 · 단계적 공개 · 스캔 가능성 · 접근성 · 설명 가능한 결과

import json
import os
import random
import time

import requests

import folium
import streamlit as st
from streamlit_folium import st_folium

import config
from modules.module5_welfare_job import (
    fetch_central_welfare,
    fetch_disability_jobs,
    fetch_local_welfare,
    normalize_jobs,
    normalize_welfare,
)
from modules.module1_data import (
    Trie,
    build_region_trie,
    fetch_facilities,
    filter_facilities,
    normalize,
    parse_region,
    reverse_geocode,
    search_places,
)
from modules.module2_path import (
    astar,
    build_graph,
    dijkstra,
    fetch_overpass_accessibility,
    get_walking_route,
    get_walking_route_waypoints,
    haversine,
    path_to_coords,
    reconstruct_path,
)
from modules.module3_score import get_top_n, greedy_coverage_recommend, rank_facilities
from modules.module4_access import (
    bfs_accessible_path,
    dijkstra_accessible,
    render_exploration_map,
    render_facility_card,
    render_map,
)

CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "cache.json")

# 이용자 유형 아이콘 (스캔 가능성 원칙: 아이콘 + 텍스트 병행)
_USER_TYPE_LABEL = {
    "일반":       "🚶 일반",
    "휠체어 사용자": "♿ 휠체어 사용자",
    "노약자·고령자": "👴 노약자·고령자",
}


# ---------------------------------------------------------------------------
# st.cache_data 함수
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_build_graph(
    facilities_key: tuple, user_lat: float, user_lon: float, radius_m: int
) -> tuple:
    """build_graph + dijkstra + dijkstra_accessible 결과 캐싱

    facilities_key 각 원소: (name, lat, lon, has_elevator, has_ramp)
    반환: (graph, distances, prev_map, acc_dist, acc_prev)
    """
    facilities = [
        {"lat": lat, "lon": lon, "has_elevator": has_e, "has_ramp": has_r}
        for _, lat, lon, has_e, has_r in facilities_key
    ]
    nodes = [{"lat": user_lat, "lon": user_lon}] + facilities
    graph = build_graph(facilities, user_lat, user_lon, radius_m)
    distances, prev_map, _, _       = dijkstra(graph, start=0)
    acc_dist, acc_prev, _, _        = dijkstra_accessible(graph, nodes, start=0)
    return graph, distances, prev_map, acc_dist, acc_prev


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_walking_route(
    user_lat: float, user_lon: float, dest_lat: float, dest_lon: float,
    wheelchair: bool = False,
) -> tuple[list[tuple], int]:
    """OSMnx 보행 경로를 캐싱 (같은 출발·목적지·모드 조합은 재다운로드 없이 재사용)

    wheelchair=True 이면 계단 배제 경로 + 엘리베이터 감지 포함.
    실패 시 RuntimeError — st.cache_data는 예외를 캐싱하지 않아 재시도 가능.
    """
    return get_walking_route(user_lat, user_lon, dest_lat, dest_lon, wheelchair=wheelchair)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_overpass(route_coords_key: tuple) -> list[dict]:
    """경로 주변 Overpass 접근성 인프라 조회 캐싱 (좌표 튜플을 키로 사용)"""
    return fetch_overpass_accessibility(list(route_coords_key))


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_walking_route_waypoints(
    waypoints: tuple,
    wheelchair: bool = False,
) -> tuple[list[tuple], int]:
    """알고리즘 경유지 좌표를 순서대로 통과하는 보행 경로 캐싱

    waypoints: ((lat, lon), ...) 튜플 — 출발지 포함, 알고리즘 path 노드 순서
    """
    return get_walking_route_waypoints(list(waypoints), wheelchair=wheelchair)


@st.cache_data(show_spinner=False, ttl=300)  # 5분 — 실시간 고장 정보
def _cached_subway_elevators(seoul_key: str) -> list[dict]:
    """서울 열린데이터광장 지하철 엘리베이터 실시간 현황
    API: ElevatorInfoNew (서울교통공사 1~9호선)
    """
    url = f"http://openapi.seoul.go.kr:8088/{seoul_key}/json/ElevatorInfoNew/1/1000/"
    try:
        resp = requests.get(url, timeout=6)
        rows = resp.json().get("ElevatorInfoNew", {}).get("row", [])
        return rows
    except Exception:
        return []


def _calc_route_stats(route_coords: list[tuple]) -> tuple[float, int]:
    total_m = sum(
        haversine(route_coords[i][0], route_coords[i][1],
                  route_coords[i + 1][0], route_coords[i + 1][1])
        for i in range(len(route_coords) - 1)
    )
    return total_m, len(route_coords) - 1


def _render_cat_buttons(key_prefix: str) -> None:
    """카테고리 필터 버튼 렌더링 — st.stop() 분기마다 key_prefix로 키 충돌 방지"""
    _sel = st.session_state.get("selected_category")
    labels = ["전체"] + list(config.CATEGORIES.keys())
    for row_labels, row_key in [(labels[:5], "a"), (labels[5:], "b")]:
        cols = st.columns(len(row_labels))
        for ci, (col, label) in enumerate(zip(cols, row_labels, strict=False)):
            with col:
                is_sel = (label == "전체" and _sel is None) or (label == _sel)
                if st.button(label, key=f"cat_{key_prefix}_{row_key}{ci}",
                             type="primary" if is_sel else "secondary",
                             use_container_width=True):
                    st.session_state.selected_category = None if label == "전체" or is_sel else label
                    st.rerun()


# ---------------------------------------------------------------------------
# 샘플 데이터 (API 키 없을 때)
# ---------------------------------------------------------------------------

def _make_sample_facilities(center_lat: float, center_lon: float) -> list[dict]:
    random.seed(42)
    eval_pool = [
        "장애인사용가능화장실", "승강기", "주출입구 접근로",
        "주출입구 높이차이 제거", "장애인전용주차구역",
    ]
    names = [
        "중앙역 복합편의센터", "시민광장 복지관", "공원 안내센터",
        "도서관 장애인실", "쇼핑몰 편의시설", "병원 로비",
        "지하철역 대합실", "구청 민원실", "복지관 주출입구", "공공화장실",
    ]
    samples = []
    for i, name in enumerate(names):
        dlat = (random.random() - 0.5) * 0.01
        dlon = (random.random() - 0.5) * 0.01
        eval_info = ", ".join(opt for opt in eval_pool if random.random() > 0.4)
        samples.append({
            "faclNm":      name,
            "lcMnad":      f"샘플 주소 {i + 1}",
            "faclLat":     str(center_lat + dlat),
            "faclLng":     str(center_lon + dlon),
            "salStaDivCd": "Y",
            "wfcltId":     f"SAMPLE-{i:04d}",
            "evalInfo":    eval_info,
        })
    return samples


# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="교통약자 편의시설 탐색기",
    page_icon="♿",
    layout="wide",
)

# ── CSS: 모바일 우선 · 다크/라이트 모드 대응 ─────────────────
st.markdown(
    """
    <style>
    /* 버튼 최소 터치 높이 48px (모바일 접근성) */
    .stButton > button { min-height: 48px; font-size: 1rem; }
    /* 라디오 항목 간격 */
    div[data-testid="stRadio"] > div { gap: 6px; }
    /* 메트릭 카드 배경 — CSS 변수로 다크모드 자동 대응 */
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border-radius: 8px;
        padding: 8px 12px;
        border: 1px solid rgba(128,128,128,0.2);
    }
    /* 경고 메시지 줄간격 */
    div[data-testid="stWarning"] p { line-height: 1.8; }
    /* 사이드바 헤더 여백 */
    section[data-testid="stSidebar"] h3 { margin-top: 1rem; }

    /* ── 모바일 반응형 (≤768px) ── */
    @media (max-width: 768px) {
        /* 타이틀 폰트 축소 */
        h1 { font-size: 1.35rem !important; }
        h2 { font-size: 1.15rem !important; }
        h3 { font-size: 1.05rem !important; }

        /* 메트릭 카드: 3열 → 작은 화면에서 글자 줄어들지 않도록 패딩 축소 */
        div[data-testid="stMetric"] { padding: 6px 8px; }
        div[data-testid="stMetricLabel"] > div  { font-size: 0.72em !important; }
        div[data-testid="stMetricValue"] > div  { font-size: 1.1em !important; }

        /* 시설 카드 패딩 축소 */
        .bfn-card { padding: 10px 12px; }
        .bfn-name { font-size: 0.95em !important; }

        /* 배지 폰트 축소 */
        .bdg { font-size: 0.74em; padding: 2px 6px; }

        /* 버튼 전체 너비 */
        .stButton > button { width: 100%; }

        /* 사이드바 내 select/slider 너비 보정 */
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stSlider { width: 100%; }

        /* 추천 이유 폰트 */
        .bfn-reasons { font-size: 0.78em; gap: 6px; }
    }

    /* ── 초소형 화면 (≤480px) ── */
    @media (max-width: 480px) {
        h1 { font-size: 1.15rem !important; }
        .bfn-card { padding: 8px 10px; }
        div[data-testid="stMetricValue"] > div { font-size: 1em !important; }
    }

    /* ── 시설 카드 (다크/라이트 모드 공통) ── */
    .bfn-card {
        background: var(--secondary-background-color);
        color: var(--text-color);           /* 다크모드에서 흰 글자 자동 상속 */
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .bfn-muted  { opacity: 0.55; }
    .bfn-addr   { opacity: 0.68; font-size: 0.82em; }
    .bfn-slabel { opacity: 0.50; font-size: 0.72em; }

    /* 추천 점수 — 라이트: 파란색, 다크: 밝은 파란색 */
    .bfn-score { font-weight: 700; font-size: 1.05em; color: #1E88E5; }
    [data-theme="dark"]  .bfn-score { color: #90CAF9; }
    @media (prefers-color-scheme: dark) { .bfn-score { color: #90CAF9; } }

    /* ── 편의시설 배지 — 좌측 컬러 보더 + 투명 배경, 텍스트는 테마 색상 ── */
    .bdg {
        border-radius: 4px; padding: 2px 8px;
        font-size: 0.8em; white-space: nowrap;
        color: var(--text-color);           /* 다크/라이트 자동 대응 */
    }
    .bdg-t { background: rgba(33,150,243,0.12);  border-left: 3px solid #42A5F5; }
    .bdg-e { background: rgba(171,71,188,0.12);  border-left: 3px solid #AB47BC; }
    .bdg-r { background: rgba(76,175,80,0.12);   border-left: 3px solid #66BB6A; }
    .bdg-p { background: rgba(255,152,0,0.12);   border-left: 3px solid #FFA726; }

    /* ── 추천 이유 ✓ — 라이트: 진한 초록, 다크: 밝은 초록 ── */
    .bfn-reasons { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; font-size:0.82em; }
    .bfn-reasons span { color: #2E7D32; }
    [data-theme="dark"]  .bfn-reasons span { color: #A5D6A7; }
    @media (prefers-color-scheme: dark) { .bfn-reasons span { color: #A5D6A7; } }

    /* ── 온보딩 카드 — 배경·글자 모두 테마 변수 ── */
    .bfn-onboard {
        text-align: center; padding: 28px 16px;
        background: var(--secondary-background-color);
        color: var(--text-color);
        border-radius: 12px;
    }
    .bfn-onboard h3 { margin: 10px 0 6px 0; color: var(--text-color); }
    .bfn-onboard .ob-title { margin: 0; font-weight: 600; }
    .bfn-onboard .ob-desc  { margin: 6px 0 0 0; font-size: 0.88em; opacity: 0.65; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("♿ 교통약자 편의시설 탐색기")
st.caption("휠체어·노약자·유아동반자를 위한 배리어프리 편의시설 추천 & 길찾기 | Dijkstra / A* / BFS 직접 구현")

# ---------------------------------------------------------------------------
# session_state 초기화
# ---------------------------------------------------------------------------

for _key in [
    "norm_facilities", "user_lat", "user_lon", "place_results", "selected_place",
    "route_dest_results", "route_dest_place", "route_nav_coords", "route_nav_facilities",
    "route_nav_elev_cnt", "route_nav_infra",
]:
    if _key not in st.session_state:
        st.session_state[_key] = None

# 설정 초기화 플래그 처리 — 위젯 렌더링 전에 실행해야 session_state 수정 가능
_RESET_DEFAULTS = {
    "place_query_input":    "",
    "user_type_radio":      "일반",
    "region_input":         "",
    "facl_type_sel":        "전체",
    "num_facilities_slider": 100,
    "need_toilet_cb":       False,
    "need_elevator_cb":     False,
    "need_ramp_cb":         False,
    "need_parking_cb":      False,
    "recommend_mode_radio": "heap",
    "route_algo_sel":       "dijkstra",
    "radius_m_slider":      config.GRAPH_RADIUS_M,
    "selected_category":    None,
}
if st.session_state.pop("_do_reset_settings", False):
    for _k, _v in _RESET_DEFAULTS.items():
        st.session_state[_k] = _v
    # 데이터 키는 None으로 초기화 (삭제하면 이후 속성 접근 시 AttributeError)
    for _k in [
        "place_results", "selected_place", "norm_facilities", "user_lat", "user_lon",
        "route_dest_results", "route_dest_place", "route_nav_coords", "route_nav_facilities",
        "route_nav_elev_cnt", "route_nav_infra",
    ]:
        st.session_state[_k] = None
    # 위젯 키는 삭제해야 Streamlit이 기본값으로 재렌더링
    for _k in ["place_radio", "goal_radio", "selected_goal_idx", "dest_radio_nav"]:
        st.session_state.pop(_k, None)

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 배리어프리 내비게이터")
    st.caption("교통약자를 위한 편의시설 추천 & 경로 안내")
    st.divider()

    # ── STEP 1 ────────────────────────────────────────────────
    _step1_done = bool(st.session_state.selected_place)
    st.markdown("**① 출발 위치** " + ("✅" if _step1_done else "← 여기서 시작하세요"))

    place_query = st.text_input(
        "장소 검색",
        placeholder="장소명 또는 주소 입력 (예: 강남역)",
        label_visibility="collapsed",
        key="place_query_input",
    )
    place_search_btn = st.button("🔎 검색", use_container_width=True)

    if place_search_btn:
        if not place_query.strip():
            st.warning("검색어를 입력해 주세요.")
        elif not config.KAKAO_REST_KEY:
            st.error("카카오 REST API 키가 없습니다.")
        else:
            with st.spinner("검색 중..."):
                results = search_places(place_query, config.KAKAO_REST_KEY, max_results=10)
            st.session_state.place_results  = results
            st.session_state.selected_place = None

    if st.session_state.place_results is not None:
        results = st.session_state.place_results
        if not results:
            st.warning("검색 결과가 없습니다. 다른 검색어를 시도해 보세요.")
        else:
            st.caption("아래 목록에서 출발지를 선택하세요")
            selected_idx = st.radio(
                "장소 선택",
                options=range(len(results)),
                format_func=lambda i: (
                    f"{results[i]['name']}\n"
                    f"{results[i]['address']}"
                    + (f"  ·  {results[i]['category']}" if results[i]["category"] else "")
                ),
                label_visibility="collapsed",
                key="place_radio",
            )
            st.session_state.selected_place = results[selected_idx]

    if st.session_state.selected_place:
        sp = st.session_state.selected_place
        st.success(f"✅ **{sp['name']}**\n\n📍 {sp['address']}")

    st.divider()

    # ── STEP 2 ────────────────────────────────────────────────
    st.markdown("**② 이용자 유형** — 유형마다 추천 가중치가 달라집니다")
    user_type = st.radio(
        "이용자 유형",
        options=list(config.USER_TYPES.keys()),
        format_func=lambda x: _USER_TYPE_LABEL.get(x, x),
        label_visibility="collapsed",
        key="user_type_radio",
    )
    user_cfg = config.USER_TYPES[user_type]
    if user_cfg["accessible_path"]:
        st.info("♿ 엘리베이터·경사로가 있는 경로만 탐색합니다.", icon=None)

    st.divider()

    # ── STEP 3 ────────────────────────────────────────────────
    st.markdown("**③ 편의시설 필터** — 필요한 시설만 추려서 검색합니다")
    need_toilet   = st.checkbox("🚻 장애인 화장실 필요", key="need_toilet_cb")
    need_elevator = st.checkbox("🛗 엘리베이터 필요",    key="need_elevator_cb")
    need_ramp     = st.checkbox("♿ 경사로 필요",        key="need_ramp_cb")
    need_parking  = st.checkbox("🅿️ 장애인 주차 필요",  key="need_parking_cb")

    st.divider()

    # ── 검색 버튼 ─────────────────────────────────────────────
    if not st.session_state.selected_place:
        st.caption("① 출발 위치를 먼저 선택해야 검색할 수 있습니다.")
    search_btn = st.button(
        "🔍 편의시설 검색",
        type="primary",
        use_container_width=True,
        disabled=not bool(st.session_state.selected_place),
    )

    # ── 고급 설정 ─────────────────────────────────────────────
    with st.expander("⚙️ 고급 설정 (선택)"):
        st.caption("기본값으로도 충분히 동작합니다.")
        region_input = st.text_input(
            "검색 지역 (시/군/구)",
            placeholder="예: 서울특별시 강남구  (비우면 자동 감지)",
            help="비워두면 출발 위치 좌표로 자동 추출합니다.",
            key="region_input",
        )

        if region_input:
            if st.session_state.norm_facilities:
                _addresses = [f["address"] for f in st.session_state.norm_facilities if f.get("address")]
                if _addresses:
                    _region_trie: Trie = build_region_trie(_addresses)
                    _suggestions = _region_trie.prefix_search(region_input, max_results=5)
                    if _suggestions:
                        st.markdown(
                            "**Trie 자동완성** — 아래 지역을 그대로 입력하면 정확히 검색됩니다:"
                        )
                        for _sug in _suggestions:
                            st.code(_sug, language=None)
            else:
                st.caption("💡 편의시설 검색 후 Trie 자동완성이 활성화됩니다.")

        facl_type_sel = st.selectbox(
            "시설 유형",
            options=config.FACILITY_TYPES,
            index=0,
            help="'전체'는 유형 필터 없음.",
            key="facl_type_sel",
        )
        facl_ty_cd = "" if facl_type_sel == "전체" else facl_type_sel

        num_facilities = st.slider(
            "최대 조회 시설 수",
            min_value=10, max_value=500, value=200, step=10,
            help="많을수록 추천 품질이 높아지지만 첫 요청이 느려집니다.",
            key="num_facilities_slider",
        )

        radius_m = st.slider(
            "그래프 탐색 반경 (m)",
            min_value=200, max_value=20000,
            value=config.GRAPH_RADIUS_M, step=100,
            help="반경 내 시설끼리 경로 그래프로 연결됩니다.",
            key="radius_m_slider",
        )

        st.markdown("**추천 알고리즘**")
        recommend_mode = st.radio(
            "추천 방식",
            options=["heap", "greedy"],
            format_func=lambda x: "📊 가중 점수 순위" if x == "heap" else "🌿 Greedy (다양성 우선)",
            label_visibility="collapsed",
            help=(
                "가중 점수: 거리·편의시설을 가중합산하여 순위를 매깁니다.\n"
                "Greedy: 아직 커버되지 않은 편의시설 유형을 우선 선택합니다."
            ),
            key="recommend_mode_radio",
        )

        st.markdown("**경로 알고리즘**")
        route_algo = st.selectbox(
            "경로 탐색 알고리즘",
            options=["dijkstra", "astar", "bfs"],
            format_func=lambda x: {
                "dijkstra": "Dijkstra — 최단거리 보장",
                "astar":    "A* — 방향 힌트로 빠른 탐색",
                "bfs":      "BFS — 경유지 최소화",
            }[x],
            help="휠체어 유형은 접근성 제약 Dijkstra로 자동 전환됩니다.",
            key="route_algo_sel",
        )

    st.divider()
    # 설정 초기화 버튼
    if st.button("↺ 설정 초기화", use_container_width=True):
        st.session_state["_do_reset_settings"] = True
        st.rerun()

    if st.button("🗑️ 캐시 초기화", use_container_width=True):
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        st.cache_data.clear()
        for _k in ["norm_facilities", "user_lat", "user_lon", "place_results", "selected_place"]:
            st.session_state[_k] = None
        # 장소 검색 텍스트·라디오 위젯 초기화
        for _k in ["place_query_input", "place_radio", "goal_radio", "selected_goal_idx"]:
            if _k in st.session_state:
                del st.session_state[_k]
        st.success("캐시가 초기화되었습니다.")

    st.divider()
    st.caption(
        "**알고리즘:** Dijkstra · A* · BFS · Greedy\n"
        "**자료구조:** Graph · Priority Queue · Heap · Queue · Set"
    )

# ---------------------------------------------------------------------------
# 검색 실행 → session_state 에 저장
# ---------------------------------------------------------------------------

if search_btn:
    sp = st.session_state.selected_place
    if not sp:
        st.warning("출발 위치를 먼저 검색하고 선택해 주세요.")
        st.stop()

    # 선택된 장소의 좌표를 바로 사용 (geocode API 추가 호출 불필요)
    st.session_state.user_lat = sp["lat"]
    st.session_state.user_lon = sp["lon"]

    if not config.DATA_GO_KR_KEY:
        st.warning(
            "공공데이터포털 API 키가 없어 샘플 데이터로 실행합니다. "
            "`.env`에 `DATA_GO_KR_KEY`를 설정하면 실제 데이터를 사용합니다."
        )
        raw = _make_sample_facilities(st.session_state.user_lat, st.session_state.user_lon)
    else:
        if region_input.strip():
            sido, gungu = parse_region(region_input)
        else:
            # region_input 이 비어있으면 출발지 좌표로 시도/구 자동 감지
            with st.spinner("지역 자동 감지 중..."):
                sido, gungu = reverse_geocode(
                    st.session_state.user_lat, st.session_state.user_lon,
                    config.KAKAO_REST_KEY,
                )
            if sido:
                st.toast(f"📍 지역 자동 감지: {sido} {gungu}", icon="✅")
            else:
                # 역지오코딩 실패 시 키워드에서 추출 시도
                sido, gungu = parse_region(place_query)

        prog_bar  = st.progress(0)
        prog_text = st.empty()

        def _cb(cur: int, total: int) -> None:
            prog_bar.progress(cur / total)
            prog_text.caption(
                f"편의시설 상세 정보 수집 중... {cur}/{total} "
                "(캐시 후 즉시 응답)"
            )

        _sel_cat = st.session_state.get("selected_category")
        _priority = config.CATEGORIES.get(_sel_cat, []) if _sel_cat else []

        raw = fetch_facilities(
            config.DATA_GO_KR_KEY, sido, gungu,
            num=num_facilities, facl_ty_cd=facl_ty_cd,
            progress_cb=_cb,
            user_lat=st.session_state.user_lat,
            user_lon=st.session_state.user_lon,
            priority_types=_priority if _priority else None,
        )
        prog_bar.empty()
        prog_text.empty()
        st.session_state["searched_sido"]  = sido
        st.session_state["searched_gungu"] = gungu

    with st.spinner("시설 데이터 처리 중..."):
        st.session_state.norm_facilities = normalize(raw, kakao_key=config.KAKAO_REST_KEY)



# ---------------------------------------------------------------------------
# 편의시설 검색 탭 렌더링 함수
# ---------------------------------------------------------------------------

def _tab1_content() -> None:
    """편의시설 검색 탭 — return으로 조기 종료 (st.stop() 대체)"""
    # ---------------------------------------------------------------------------
    # 렌더링 (session_state 에 데이터가 있을 때)
    # ---------------------------------------------------------------------------

    if st.session_state.norm_facilities is None:
        st.markdown("---")

        # ── 3단계 사용법 ──────────────────────────────────────────
        _card = (
            '<div class="bfn-onboard" style="border-top:4px solid {accent};">'
            '<div style="font-size:2.4em;">{icon}</div>'
            '<h3>{step}</h3>'
            '<p class="ob-title">{title}</p>'
            '<p class="ob-desc">{desc}</p>'
            '</div>'
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                _card.format(accent="#42A5F5", icon="📍", step="1단계",
                             title="출발 위치 입력",
                             desc="강남역, 서울시청 등<br>장소명 또는 주소로 검색 후 선택"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                _card.format(accent="#AB47BC", icon="👤", step="2단계",
                             title="이용자 유형 선택",
                             desc="휠체어·노약자·고령자·일반 중<br>해당 유형 선택"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                _card.format(accent="#66BB6A", icon="🔍", step="3단계",
                             title="검색 버튼 클릭",
                             desc="주변 편의시설을 자동 탐색하고<br>TOP 5를 추천합니다"),
                unsafe_allow_html=True,
            )

        # ── 기능 안내 ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 주요 기능")
        fa, fb, fc = st.columns(3)
        with fa:
            st.info("**♿ 휠체어 모드**\n\n엘리베이터·경사로가 끊기지 않는 접근 가능 경로만 탐색합니다.")
        with fb:
            st.info("**🗺️ 실제 보행 경로**\n\nOSM 보행자 도로망 기반으로 산길·건물 관통 없이 경로를 표시합니다.")
        with fc:
            st.info("**📊 알고리즘 비교**\n\nDijkstra vs A* 탐색 노드를 지도에서 직접 비교할 수 있습니다.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("💡 **검색 팁:** 장소명(강남역, 가천대학교)과 주소(강남대로 15) 모두 검색 가능합니다.  \n"
                   "지역을 비워두면 출발지 좌표로 자동 감지합니다.")
        return

    user_lat = st.session_state.user_lat
    user_lon = st.session_state.user_lon

    # 편의시설 필터 (현재 체크박스 상태로 매 렌더링마다 적용)
    filters = {
        "need_toilet":   need_toilet,
        "need_elevator": need_elevator,
        "need_ramp":     need_ramp,
        "need_parking":  need_parking,
    }
    facilities = filter_facilities(st.session_state.norm_facilities, filters)

    # 카테고리 필터 — 선택된 카테고리의 시설 유형만 표시 (client-side)
    _sel_cat = st.session_state.get("selected_category")
    if _sel_cat and _sel_cat in config.CATEGORIES:
        _cat_types = set(config.CATEGORIES[_sel_cat])
        facilities = [f for f in facilities if f.get("fac_type") in _cat_types]

    if not facilities:
        _total = len(st.session_state.norm_facilities)
        _filter_labels = {
            "need_toilet":   "🚻 장애인 화장실",
            "need_elevator": "🛗 엘리베이터",
            "need_ramp":     "♿ 경사로",
            "need_parking":  "🅿️ 장애인 주차",
        }
        _active = [_filter_labels[k] for k, v in filters.items() if v]
        _hint_items = [f"- {lb} 필터 해제" for lb in _active]
        if _sel_cat:
            _hint_items.insert(0, "- 아래 카테고리에서 다른 항목을 선택하거나 '전체'로 초기화")
        if not _hint_items:
            _hint_items = ["- 조회 시설 수를 늘려보세요"]
        st.warning(
            f"조회된 {_total}개 시설 중 현재 조건을 만족하는 곳이 없습니다.\n\n"
            f"다음 중 하나를 시도해 보세요:\n" + "  \n".join(_hint_items)
        )
        st.markdown("**📂 카테고리 재선택**")
        _render_cat_buttons("e")
        return

    # 출발지로부터 거리 계산 후 가까운 순 정렬 (진단 정보 및 UX용)
    _fac_dists = sorted(
        [(haversine(user_lat, user_lon, f["lat"], f["lon"]), f) for f in facilities],
        key=lambda x: x[0],
    )
    nearest_dist_m   = _fac_dists[0][0] if _fac_dists else float("inf")
    nearest_fac_name = _fac_dists[0][1]["name"] if _fac_dists else "없음"
    facilities       = [f for _, f in _fac_dists]  # 가까운 순으로 재정렬

    # 출발지와 연결 가능한 시설 수 (반경 내 직선거리 기준)
    in_radius_count = sum(1 for d, _ in _fac_dists if d <= radius_m)

    # ── 즉각적 피드백 원칙: 메트릭 행으로 상태 한눈에 표시 ──────
    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("🏢 시설 수",    f"{len(facilities)}개")
    _m2.metric("🗺️ 탐색 반경",  f"{radius_m}m")
    _m3.metric("🔗 반경 내",    f"{in_radius_count}개",
               delta=None if in_radius_count > 0 else "0 — 반경 확대 필요",
               delta_color="off")
    _m4.metric("📏 최근접",     f"{nearest_dist_m:.0f}m")

    # 그래프 구축 & 최단거리 (session_state 캐시)
    # 핵심: 그래프는 전체 norm_facilities로 구성 — 필터로 노드가 줄면 중간 경유지가
    # 사라져 원거리 시설이 연결 불가(inf)가 되는 문제를 방지한다.
    norm_facilities = st.session_state.norm_facilities
    # st.cache_data 는 대용량 tuple 키를 매번 재해싱 → MD5 hash로 대체해 O(1) 조회
    _key_items = [(round(f["lat"], 5), round(f["lon"], 5),
                   f.get("has_elevator", False), f.get("has_ramp", False))
                  for f in norm_facilities]
    _graph_cache_key = "gc_" + json.dumps(
        [_key_items, user_lat, user_lon, radius_m], separators=(",", ":")
    ).__hash__().__format__("x")

    if _graph_cache_key not in st.session_state:
        with st.spinner("최단경로 계산 중..."):
            _fac = [{"lat": f["lat"], "lon": f["lon"],
                     "has_elevator": f.get("has_elevator", False),
                     "has_ramp":     f.get("has_ramp", False)}
                    for f in norm_facilities]
            _nodes = [{"lat": user_lat, "lon": user_lon}] + _fac
            _g = build_graph(_fac, user_lat, user_lon, radius_m)
            _d, _pm, _, _  = dijkstra(_g, start=0)
            _ad, _ap, _, _ = dijkstra_accessible(_g, _nodes, start=0)
            st.session_state[_graph_cache_key] = (_g, _d, _pm, _ad, _ap)

    graph, distances, prev_map, acc_dist, acc_prev = st.session_state[_graph_cache_key]
    all_nodes = [{"lat": user_lat, "lon": user_lon}] + list(norm_facilities)

    reachable = sum(
        1 for i in range(1, len(all_nodes))
        if distances.get(i, float("inf")) < float("inf")
    )
    if reachable == 0:
        suggest_m = int(nearest_dist_m * 1.3)
        st.warning(
            f"출발지에서 반경 {radius_m}m 내에 연결된 시설이 없습니다.\n\n"
            f"- 가장 가까운 시설: **{nearest_fac_name}** ({nearest_dist_m:.0f}m)\n"
            f"- 탐색 반경을 **{suggest_m}m** 이상으로 늘려보세요.\n"
            f"- 또는 검색 지역을 출발지 근처 구·동 단위로 좁혀보세요.  \n"
            f"  예) '서울특별시 강남구' → '역삼동'"
        )
        st.markdown("**📂 카테고리 재선택**")
        _render_cat_buttons("r")
        return

    # 필터된 시설을 전체 그래프의 노드 ID에 매핑
    # filter_facilities 는 동일 dict 객체를 반환하므로 id() 비교가 유효함
    _norm_id_map = {id(f): idx + 1 for idx, f in enumerate(norm_facilities)}
    _filt_dist = {
        local_i + 1: distances.get(_norm_id_map.get(id(fac), -1), float("inf"))
        for local_i, fac in enumerate(facilities)
    }
    _filt_acc = {
        local_i + 1: acc_dist.get(_norm_id_map.get(id(fac), -1), float("inf"))
        for local_i, fac in enumerate(facilities)
    }

    # 추천 점수화 — 필터된 시설만 대상으로, 전체 그래프 거리 사용
    weights = user_cfg["weights"]
    _score_dist_local = _filt_acc if user_cfg["accessible_path"] else _filt_dist
    if recommend_mode == "greedy":
        top_facilities = greedy_coverage_recommend(facilities, _score_dist_local, weights, n=config.TOP_N)
    else:
        scored = rank_facilities(facilities, _score_dist_local, weights)
        top_facilities = get_top_n(scored, n=config.TOP_N)

    # _node_id 를 필터 내 로컬 인덱스 → 전체 그래프 원래 node_id 로 교정
    # (경로 탐색 함수들은 전체 그래프 기준 node_id 를 사용함)
    _local_to_full = {
        local_i + 1: _norm_id_map.get(id(fac))
        for local_i, fac in enumerate(facilities)
    }
    for _tf in top_facilities:
        _lid = _tf.get("_node_id")
        if _lid is not None and _lid in _local_to_full:
            _tf["_node_id"] = _local_to_full[_lid]

    # 카드/차트 거리 표시용 score_dist 는 전체 그래프 기준 (node_id 교정 후)
    score_dist = acc_dist if user_cfg["accessible_path"] else distances

    if not top_facilities:
        st.warning("추천 가능한 시설이 없습니다.")
        st.markdown("**📂 카테고리 재선택**")
        _render_cat_buttons("t")
        return

    # ---------------------------------------------------------------------------
    # 목적지 선택 (session_state 로 유지)
    # ---------------------------------------------------------------------------

    goal_options = [f"#{i + 1}  {fac['name']}" for i, fac in enumerate(top_facilities)]

    if "selected_goal_idx" not in st.session_state:
        st.session_state.selected_goal_idx = 0
    # 검색 결과가 바뀌어 이전 인덱스가 범위를 벗어난 경우 초기화
    if st.session_state.selected_goal_idx >= len(goal_options):
        st.session_state.selected_goal_idx = 0

    selected_label = st.radio(
        "경로 탐색할 시설 선택",
        options=goal_options,
        index=st.session_state.selected_goal_idx,
        horizontal=True,
        key="goal_radio",
    )
    selected_goal_idx = goal_options.index(selected_label)
    st.session_state.selected_goal_idx = selected_goal_idx
    goal_node = top_facilities[selected_goal_idx].get("_node_id")

    # ---------------------------------------------------------------------------
    # 경로 탐색
    # ---------------------------------------------------------------------------

    route_coords = None
    algo_label   = ""

    if goal_node is not None:
        if user_cfg["accessible_path"]:
            # 접근성 제약 Dijkstra — 엘리베이터·경사로 없는 중간 노드 제외, 실거리 최소화
            path = reconstruct_path(acc_prev, goal_node)
            algo_label = "접근성 제약 Dijkstra"
        elif route_algo == "astar":
            path, _, _ = astar(graph, 0, goal_node, all_nodes)
            algo_label = "A* 최단경로"
        elif route_algo == "bfs":
            # facilities 전달 (all_nodes 아님): bfs 내부에서 nodes[neighbor-1]로 접근하므로
            # user_node(index 0)가 포함된 all_nodes를 넘기면 neighbor=1 시설이 user_node로 오인됨
            path, _, _ = bfs_accessible_path(graph, norm_facilities, 0, goal_node)
            algo_label = "BFS (최소 홉)"
        else:
            path = reconstruct_path(prev_map, goal_node)
            algo_label = "Dijkstra 최단경로"

        if path:
            # 알고리즘 경로의 각 노드 좌표를 경유지로 추출 → OSMnx에 그대로 전달
            # 중간 시설 노드도 실제 도로 경로에 반영됨
            _waypoints = tuple(
                (all_nodes[n]["lat"], all_nodes[n]["lon"]) for n in path
            )
            with st.spinner("🗺️ 보행 경로 계산 중… (최대 60초)"):
                try:
                    walk_coords, _ = _cached_walking_route_waypoints(
                        _waypoints,
                        wheelchair=user_cfg.get("accessible_path", False),
                    )
                    route_coords = walk_coords
                    algo_label += f" + OSMnx 보행 경로 ({len(path) - 1}구간)"
                except Exception as _wc_e:
                    if user_cfg.get("accessible_path"):
                        st.warning(f"♿ 계단 없는 접근 경로를 찾을 수 없습니다. 직선 경로로 대체합니다.\n\n{_wc_e}", icon="⚠️")
                    else:
                        st.warning("실제 보행 경로 계산에 실패했습니다. 직선 경로로 대체합니다.", icon="⚠️")
                    route_coords = path_to_coords(path, all_nodes)
                    algo_label += " (직선 근사)"
        else:
            goal_fac = all_nodes[goal_node]
            with st.spinner("🗺️ 보행 경로 계산 중… (최대 60초)"):
                try:
                    walk_coords, _ = _cached_walking_route(
                        user_lat, user_lon, goal_fac["lat"], goal_fac["lon"],
                        wheelchair=user_cfg.get("accessible_path", False),
                    )
                    route_coords = walk_coords
                    algo_label = "OSMnx 보행 경로 (알고리즘 연결 없음)"
                except Exception as _wc_e:
                    if user_cfg.get("accessible_path"):
                        st.warning(f"♿ 계단 없는 접근 경로를 찾을 수 없습니다. 직선 경로로 대체합니다.\n\n{_wc_e}", icon="⚠️")
                    else:
                        st.warning("실제 보행 경로 계산에 실패했습니다. 직선 경로로 대체합니다.", icon="⚠️")
                    route_coords = [(user_lat, user_lon), (goal_fac["lat"], goal_fac["lon"])]
                    algo_label += " (연결 경로 없음 — 직선 표시)"

    # 경로 거리 & 소요 시간 표시
    if route_coords and len(route_coords) >= 2:
        total_dist_m, segments = _calc_route_stats(route_coords)
        speed_mpm  = user_cfg["speed_mpm"]
        eta_min    = total_dist_m / speed_mpm
        dist_label = (
            f"{total_dist_m / 1000:.2f} km" if total_dist_m >= 1000
            else f"{total_dist_m:.0f} m"
        )
        speed_kmh  = round(speed_mpm * 60 / 1000, 1)
        st.success(
            f"**{algo_label}** | 유형: {user_type} | 경유 구간: {segments}개 | "
            f"총 거리: {dist_label} | 예상 소요: 약 {eta_min:.0f}분 ({speed_kmh} km/h)"
        )

    # ---------------------------------------------------------------------------
    # 지도 + 시설 카드
    # ---------------------------------------------------------------------------

    col_map, col_list = st.columns([3, 2])

    with col_map:
        st.subheader("🗺️ 지도")
        _render_cat_buttons("m")
        _goal_fac_for_map = all_nodes[goal_node] if goal_node is not None else None
        folium_map = render_map(
            facilities=facilities,
            recommended=top_facilities,
            user_lat=user_lat,
            user_lon=user_lon,
            route_coords=route_coords,
            goal_fac=_goal_fac_for_map,
        )
        st_folium(folium_map, width="100%", height=520)

    with col_list:
        st.subheader(f"🏆 추천 TOP {config.TOP_N}")
        st.caption(f"이용자 유형·편의시설 보유 현황 기반 상위 {config.TOP_N}곳")
        with st.container(border=True):
            for rank, fac in enumerate(top_facilities, start=1):
                _nid   = fac.get("_node_id")
                _d_m   = score_dist.get(_nid, float("inf")) if _nid else float("inf")
                _reasons: list[str] = []
                if _d_m < float("inf"):
                    _dist_label = f"{_d_m / 1000:.1f} km" if _d_m >= 1000 else f"{_d_m:.0f} m"
                    _reasons.append(f"출발지에서 {_dist_label}")
                if fac.get("has_toilet"):
                    _reasons.append("장애인화장실 보유")
                if fac.get("has_elevator"):
                    _reasons.append("엘리베이터 보유")
                if fac.get("has_ramp"):
                    _reasons.append("경사로 보유")
                if fac.get("has_parking"):
                    _reasons.append("전용주차 보유")
                render_facility_card(fac, fac.get("score", 0.0), rank, reasons=_reasons)

    # ---------------------------------------------------------------------------
    # 추천 점수 분해 차트
    # ---------------------------------------------------------------------------

    _COMP_META = [
        ("dist",   "최단거리",   "#42A5F5"),
        ("toilet", "화장실",    "#AB47BC"),
        ("elev",   "엘리베이터","#66BB6A"),
        ("ramp",   "경사로",    "#FFA726"),
        ("park",   "주차",      "#EF5350"),
    ]

    _bar_rows: list[tuple] = []
    for _fac in top_facilities:
        _nid    = _fac.get("_node_id")
        _dist_m = score_dist.get(_nid, float("inf")) if _nid else float("inf")
        _dist_km = _dist_m / 1000 if _dist_m < float("inf") else 999
        _comps = {
            "dist":   1 / (1 + _dist_km) * weights["dist"],
            "toilet": int(bool(_fac.get("has_toilet")))   * weights["toilet"],
            "elev":   int(bool(_fac.get("has_elevator"))) * weights["elev"],
            "ramp":   int(bool(_fac.get("has_ramp")))     * weights["ramp"],
            "park":   int(bool(_fac.get("has_parking")))  * weights["park"],
        }
        _bar_rows.append((_fac["name"], _comps, sum(_comps.values())))

    # evalInfo 누락 여부 — 편의시설 데이터가 없는 시설 비율 계산
    _no_eval_count = sum(
        1 for _fac in top_facilities
        if not any(_fac.get(k) for k in ("has_toilet", "has_elevator", "has_ramp", "has_parking"))
    )
    _eval_missing = _no_eval_count == len(top_facilities)  # 전체가 누락

    _max_score = max((r[2] for r in _bar_rows), default=1.0) or 1.0

    _legend_html = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;">'
        f'<span style="width:9px;height:9px;border-radius:2px;background:{_c};flex-shrink:0;"></span>'
        f'<span style="font-size:0.77em;opacity:0.75;">{_lb}</span></span>'
        for _, _lb, _c in _COMP_META
    )

    _rows_html = ""
    for _fname, _comps, _total in _bar_rows:
        _segs  = "".join(
            f'<div style="width:{_comps[_k] / _max_score * 100:.2f}%;'
            f'background:{_c};height:100%;flex-shrink:0;" title="{_lb} {_comps[_k]:.3f}"></div>'
            for _k, _lb, _c in _COMP_META if _comps[_k] > 0
        )
        _rows_html += (
            f'<div style="display:flex;flex-direction:column;gap:3px;margin-bottom:8px;">'
            f'  <div style="font-size:0.78em;opacity:0.85;word-break:keep-all;">{_fname}</div>'
            f'  <div style="display:flex;align-items:center;gap:6px;">'
            f'    <div style="flex:1;display:flex;height:14px;border-radius:3px;overflow:hidden;'
            f'background:rgba(128,128,128,0.12);">{_segs}</div>'
            f'    <div style="flex-shrink:0;width:34px;font-size:0.74em;text-align:right;opacity:0.55;">'
            f'{_total:.2f}</div>'
            f'  </div>'
            f'</div>'
        )

    _w_hint = " / ".join(f"{_lb} {weights[_k]}" for _k, _lb, _ in _COMP_META)
    _eval_warn_html = (
        '<div style="font-size:0.76em;color:#F97316;margin-top:8px;">'
        '⚠️ 편의시설 상세 데이터(화장실·엘리베이터·경사로·주차) 미수신 — '
        '공공 API 일일 쿼터 초과 상태입니다. 쿼터 리셋 후 재검색하면 정상 표시됩니다.'
        '</div>'
        if _eval_missing else ""
    )
    st.markdown(
        f'<div class="bfn-card" style="margin-top:4px;">'
        f'  <div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'flex-wrap:wrap;gap:6px;margin-bottom:10px;">'
        f'    <span style="font-weight:600;font-size:0.9em;">📊 추천 점수 구성</span>'
        f'    <span style="font-size:0.74em;opacity:0.45;">이용자: {user_type} · 가중치: {_w_hint}</span>'
        f'  </div>'
        f'  <div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px;">{_legend_html}</div>'
        f'  {_rows_html}'
        f'  {_eval_warn_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------------
    # Dijkstra vs A* 비교 패널
    # ---------------------------------------------------------------------------

    if goal_node is not None:
        with st.expander("🔬 알고리즘 비교 — Dijkstra / A* / BFS (고급)", expanded=False):
            st.caption("동일한 그래프에서 세 알고리즘을 실행하여 성능을 비교합니다.")

            # ── 알고리즘 실행 ───────────────────────────────────────
            t0 = time.perf_counter()
            _, _, d_visited, d_order = dijkstra(graph, start=0)
            d_time_ms = (time.perf_counter() - t0) * 1000
            d_path = reconstruct_path(prev_map, goal_node)
            d_hops = len(d_path) - 1 if d_path else None
            d_dist = sum(
                haversine(all_nodes[d_path[i]]["lat"], all_nodes[d_path[i]]["lon"],
                          all_nodes[d_path[i+1]]["lat"], all_nodes[d_path[i+1]]["lon"])
                for i in range(len(d_path) - 1)
            ) if d_path else None

            t0 = time.perf_counter()
            a_path, a_visited, a_order = astar(graph, 0, goal_node, all_nodes)
            a_time_ms = (time.perf_counter() - t0) * 1000
            a_hops = len(a_path) - 1 if a_path else None
            a_dist = sum(
                haversine(all_nodes[a_path[i]]["lat"], all_nodes[a_path[i]]["lon"],
                          all_nodes[a_path[i+1]]["lat"], all_nodes[a_path[i+1]]["lon"])
                for i in range(len(a_path) - 1)
            ) if a_path else None

            t0 = time.perf_counter()
            b_path, b_visited, b_order = bfs_accessible_path(graph, norm_facilities, 0, goal_node)
            b_time_ms = (time.perf_counter() - t0) * 1000
            b_hops = len(b_path) - 1 if b_path else None
            b_dist = sum(
                haversine(all_nodes[b_path[i]]["lat"], all_nodes[b_path[i]]["lon"],
                          all_nodes[b_path[i+1]]["lat"], all_nodes[b_path[i+1]]["lon"])
                for i in range(len(b_path) - 1)
            ) if b_path else None

            # ── 지표 비교표 ─────────────────────────────────────────
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("#### Dijkstra")
                st.metric("처리 노드 수", f"{d_visited}개")
                st.metric("실행 시간",   f"{d_time_ms:.2f} ms")
                st.metric("경유 홉 수",  f"{d_hops}개" if d_hops is not None else "경로 없음")
                st.metric("경로 거리",   f"{d_dist:.0f} m" if d_dist is not None else "-")
                st.info("전체 탐색 → 실거리 최단 보장")

            with c2:
                st.markdown("#### A*")
                st.metric("처리 노드 수", f"{a_visited}개",
                          delta=f"{a_visited - d_visited:+d}개" if d_visited else None)
                st.metric("실행 시간",   f"{a_time_ms:.2f} ms",
                          delta=f"{a_time_ms - d_time_ms:+.2f} ms")
                st.metric("경유 홉 수",  f"{a_hops}개" if a_hops is not None else "경로 없음")
                st.metric("경로 거리",   f"{a_dist:.0f} m" if a_dist is not None else "-")
                st.info("방향 휴리스틱 → 탐색 범위 축소")

            with c3:
                st.markdown("#### BFS (접근성 제약)")
                st.metric("처리 노드 수", f"{b_visited}개",
                          delta=f"{b_visited - d_visited:+d}개" if d_visited else None)
                st.metric("실행 시간",   f"{b_time_ms:.2f} ms",
                          delta=f"{b_time_ms - d_time_ms:+.2f} ms")
                st.metric("경유 홉 수",  f"{b_hops}개" if b_hops is not None else "경로 없음")
                st.metric("경로 거리",   f"{b_dist:.0f} m" if b_dist is not None else "-")
                st.info("접근 가능 노드만 통과 → 홉 최소화")

            if d_visited:
                _node_reduction = (1 - a_visited / d_visited) * 100
                _time_ratio     = d_time_ms / a_time_ms if a_time_ms > 0.001 else 1.0
                _speed_txt = (
                    f"실행 시간은 A*가 Dijkstra보다 **{_time_ratio:.1f}배** 빠릅니다."
                    if _time_ratio > 1.05
                    else "두 알고리즘의 실행 시간은 유사합니다."
                )
                st.markdown(
                    f"**결과:** A*가 Dijkstra 대비 **{_node_reduction:.1f}%** 적은 노드를 탐색했습니다. "
                    f"{_speed_txt}  \n"
                    "BFS는 엘리베이터·경사로 보유 시설만 경유하므로 탐색 구조가 다릅니다."
                    if _node_reduction > 0
                    else f"**결과:** 이 그래프에서는 Dijkstra와 A*의 탐색 범위가 유사합니다. {_speed_txt}"
                )

            st.divider()
            st.markdown("#### 탐색 노드 시각화 (Dijkstra vs A*)")
            st.caption(
                "🔵 파란 원: Dijkstra 전용  |  🟣 보라 원: 공통  |  🟠 주황 원: A* 전용  |  진할수록 먼저 탐색"
            )
            exp_map = render_exploration_map(
                d_order, a_order, all_nodes, user_lat, user_lon, goal_node
            )
            st_folium(exp_map, width="100%", height=400, key="exploration_map")

            st.divider()
            st.markdown("#### 탐색 단계 애니메이션")
            st.caption("▶ 재생을 누르면 알고리즘이 노드를 탐색하는 과정을 깜빡임 없이 재생합니다.")

            anim_algo = st.radio(
                "표시 알고리즘",
                options=["both", "dijkstra", "astar", "bfs"],
                format_func=lambda x: {
                    "both":     "Dijkstra + A* 비교",
                    "dijkstra": "Dijkstra만",
                    "astar":    "A*만",
                    "bfs":      "BFS만",
                }[x],
                horizontal=True,
                key="anim_algo_radio",
            )

            if anim_algo == "dijkstra":
                _d_ref, _a_ref = d_order, []
            elif anim_algo == "astar":
                _d_ref, _a_ref = [], a_order
            elif anim_algo == "bfs":
                _d_ref, _a_ref = b_order, []
            else:
                _d_ref, _a_ref = d_order, a_order

            _max_step = max(len(_d_ref), len(_a_ref))

            if _max_step > 0:
                # 프레임 데이터 빌드 — 각 단계에서 추가할 마커 목록
                _astar_set = set(_a_ref)
                _dijk_set  = set(_d_ref)
                _added: set[int] = set()
                _frames: list[list[dict]] = []

                for _i in range(_max_step):
                    _additions: list[dict] = []

                    if _i < len(_d_ref):
                        _nid = _d_ref[_i]
                        if _nid != 0 and _nid < len(all_nodes) and _nid not in _added:
                            _nd = all_nodes[_nid]
                            _color = "#8B5CF6" if _nid in _astar_set else "#3B82F6"
                            _lbl   = f"{'Dijkstra+A* 공통' if _nid in _astar_set else 'Dijkstra 전용'} #{_i + 1}"
                            _additions.append({"lat": _nd["lat"], "lon": _nd["lon"],
                                               "color": _color, "label": _lbl})
                            _added.add(_nid)

                    if _i < len(_a_ref):
                        _nid = _a_ref[_i]
                        if _nid != 0 and _nid < len(all_nodes) and _nid not in _dijk_set and _nid not in _added:
                            _nd = all_nodes[_nid]
                            _additions.append({"lat": _nd["lat"], "lon": _nd["lon"],
                                               "color": "#F97316", "label": f"A* 전용 #{_i + 1}"})
                            _added.add(_nid)

                    _frames.append(_additions)

                # 목적지 좌표
                _goal_json = "null"
                if goal_node is not None and 0 < goal_node < len(all_nodes):
                    _gn = all_nodes[goal_node]
                    _goal_json = json.dumps({"lat": _gn["lat"], "lon": _gn["lon"]})

                _frames_json = json.dumps(_frames)
                _total       = _max_step

                # Leaflet 기반 순수 JS 애니메이션 — Streamlit rerun 없음 → 깜빡임 없음
                _anim_html = f"""<!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
      *{{margin:0;padding:0;box-sizing:border-box;}}
      body{{background:#1F2937;color:#F9FAFB;font-family:system-ui,sans-serif;}}
      #map{{width:100%;height:340px;}}
      #ctrl{{display:flex;align-items:center;gap:8px;padding:8px 12px;
             background:#111827;flex-wrap:wrap;}}
      .btn{{padding:5px 14px;border:none;border-radius:6px;cursor:pointer;font-size:14px;}}
      #btn-play {{background:#3B82F6;color:#fff;}}
      #btn-pause{{background:#6B7280;color:#fff;display:none;}}
      #btn-stop {{background:#374151;color:#D1D5DB;}}
      #pw{{flex:1;min-width:80px;height:6px;background:#374151;border-radius:3px;}}
      #pb{{height:100%;background:#3B82F6;border-radius:3px;width:0%;transition:width .1s;}}
      #lbl{{font-size:12px;color:#9CA3AF;min-width:75px;text-align:right;}}
      .sw{{display:flex;align-items:center;gap:5px;font-size:12px;color:#9CA3AF;}}
      input[type=range]{{width:70px;accent-color:#3B82F6;}}
    </style></head>
    <body>
    <div id="map"></div>
    <div id="ctrl">
      <button class="btn" id="btn-play"  onclick="play()">▶ 재생</button>
      <button class="btn" id="btn-pause" onclick="pause()">⏸ 일시정지</button>
      <button class="btn" id="btn-stop"  onclick="stop()">⏹ 처음</button>
      <div id="pw"><div id="pb"></div></div>
      <span id="lbl">0 / {_total}</span>
      <div class="sw">속도
        <input type="range" id="sl" min="0.1" max="3" step="0.1" value="0.3">
        <span id="sv">0.3s</span>
      </div>
    </div>
    <script>
    const FRAMES={_frames_json};
    const TOTAL={_total};
    const ULat={user_lat},ULon={user_lon};
    const GOAL={_goal_json};

    const map=L.map('map',{{zoomControl:true}}).setView([ULat,ULon],15);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
      {{attribution:'© OSM',maxZoom:19}}).addTo(map);

    function mkStatic(){{
      L.circleMarker([ULat,ULon],{{radius:9,color:'#3B82F6',fillColor:'#3B82F6',fillOpacity:1,weight:2}})
       .addTo(map).bindTooltip('출발지');
      if(GOAL) L.circleMarker([GOAL.lat,GOAL.lon],
        {{radius:9,color:'#EF4444',fillColor:'#EF4444',fillOpacity:1,weight:2}})
       .addTo(map).bindTooltip('목적지');
    }}
    mkStatic();

    let step=0,timer=null;
    const sl=document.getElementById('sl');
    const sv=document.getElementById('sv');
    sl.addEventListener('input',()=>sv.textContent=sl.value+'s');

    function gi(){{return parseFloat(sl.value)*1000;}}
    function addStep(i){{
      (FRAMES[i]||[]).forEach(p=>
        L.circleMarker([p.lat,p.lon],{{radius:7,color:p.color,fillColor:p.color,
          fillOpacity:.75,weight:1}}).addTo(map).bindTooltip(p.label));
    }}
    function ui(){{
      document.getElementById('lbl').textContent=step+' / '+TOTAL;
      document.getElementById('pb').style.width=(step/TOTAL*100)+'%';
    }}
    function play(){{
      if(timer)return;
      document.getElementById('btn-play').style.display='none';
      document.getElementById('btn-pause').style.display='';
      function tick(){{
        if(step>=TOTAL){{pause();return;}}
        addStep(step++);ui();
        timer=setTimeout(tick,gi());
      }}
      tick();
    }}
    function pause(){{
      clearTimeout(timer);timer=null;
      document.getElementById('btn-play').style.display='';
      document.getElementById('btn-pause').style.display='none';
    }}
    function stop(){{
      pause();
      map.eachLayer(l=>{{if(l instanceof L.CircleMarker)map.removeLayer(l);}});
      mkStatic();step=0;ui();
    }}
    ui();
    </script></body></html>"""

                st.components.v1.html(_anim_html, height=420)

            st.divider()
            st.markdown("#### 그래프 구조 시각화")
            st.caption(
                "노드(시설)와 엣지(연결선)로 구성된 그래프 자료구조를 지도에 표시합니다. "
                "반경 내 시설끼리 엣지로 연결되며, 알고리즘은 이 그래프를 탐색합니다."
            )
            if st.checkbox("엣지 표시", key="show_edges_cb"):
                _edge_map = folium.Map(location=[user_lat, user_lon], zoom_start=15)

                # 엣지 — 회색 얇은 선 (중복 제거)
                _drawn_edges: set[tuple[int, int]] = set()
                _edge_count = 0
                for _src, _nbrs in graph.items():
                    for _dst in _nbrs:
                        _ek = (min(_src, _dst), max(_src, _dst))
                        if _ek in _drawn_edges:
                            continue
                        _drawn_edges.add(_ek)
                        _edge_count += 1
                        if _src < len(all_nodes) and _dst < len(all_nodes):
                            _n1, _n2 = all_nodes[_src], all_nodes[_dst]
                            folium.PolyLine(
                                locations=[[_n1["lat"], _n1["lon"]], [_n2["lat"], _n2["lon"]]],
                                color="#9CA3AF", weight=2.5, opacity=0.7,
                            ).add_to(_edge_map)

                # 노드 — 추천 시설(빨강), 일반 시설(회색), 출발지(파랑)
                _rec_names = {f["name"] for f in top_facilities}
                for _ni, _nd in enumerate(all_nodes):
                    if _ni == 0:
                        folium.CircleMarker(
                            [_nd["lat"], _nd["lon"]], radius=9,
                            color="#3B82F6", fill=True, fill_color="#3B82F6", fill_opacity=1,
                            tooltip="출발지 (노드 0)",
                        ).add_to(_edge_map)
                    elif _nd.get("name") in _rec_names:
                        folium.CircleMarker(
                            [_nd["lat"], _nd["lon"]], radius=7,
                            color="#EF4444", fill=True, fill_color="#EF4444", fill_opacity=0.9,
                            tooltip=f"추천 {_nd.get('name','')} (노드 {_ni})",
                        ).add_to(_edge_map)
                    else:
                        folium.CircleMarker(
                            [_nd["lat"], _nd["lon"]], radius=4,
                            color="#6B7280", fill=True, fill_color="#6B7280", fill_opacity=0.6,
                            tooltip=f"{_nd.get('name','')} (노드 {_ni})",
                        ).add_to(_edge_map)

                st.caption(
                    f"🔵 출발지  🔴 추천 시설  ⚫ 일반 시설  |  "
                    f"노드 {len(all_nodes)}개 · 엣지 {_edge_count}개"
                )
                # st_folium 대신 정적 HTML 렌더링 — 양방향 통신 없으므로 rerun 미발생
                st.components.v1.html(_edge_map._repr_html_(), height=420)

    # ---------------------------------------------------------------------------
    # Greedy Set Cover 단계별 커버리지 시각화
    # ---------------------------------------------------------------------------

    if recommend_mode == "greedy" and top_facilities:
        with st.expander("📊 Greedy Set Cover — 단계별 커버리지", expanded=False):
            st.caption(
                "매 단계에서 아직 커버되지 않은 편의시설 유형을 가장 많이 추가하는 "
                "시설을 탐욕적으로 선택한 과정입니다. "
                "covered set이 갱신될수록 다음 선택 기준이 달라집니다."
            )

            _AMENITY_MAP = {
                "has_toilet":   "🚻 화장실",
                "has_elevator": "🛗 엘리베이터",
                "has_ramp":     "♿ 경사로",
                "has_parking":  "🅿️ 주차",
            }
            _covered: set[str] = set()

            for _step, _fac in enumerate(top_facilities, 1):
                _fac_keys  = {k for k in _AMENITY_MAP if _fac.get(k)}
                _new_keys  = _fac_keys - _covered
                _covered  |= _fac_keys
                _cov_pct   = len(_covered) / len(_AMENITY_MAP)

                _col_no, _col_name, _col_new, _col_bar = st.columns([0.5, 2.5, 2.5, 1.5])
                with _col_no:
                    st.markdown(f"**#{_step}**")
                with _col_name:
                    st.markdown(f"**{_fac['name']}**")
                with _col_new:
                    _new_label = (
                        "  ".join(_AMENITY_MAP[k] for k in _new_keys)
                        if _new_keys else "—  (거리 우선 선택)"
                    )
                    st.markdown(f"🆕 {_new_label}")
                with _col_bar:
                    st.progress(_cov_pct, text=f"{len(_covered)}/{len(_AMENITY_MAP)}")



def _tab3_content() -> None:
    """접근성 정보 탭 — 긴급 화장실 + 지하철 엘리베이터 현황"""
    if not st.session_state.get("user_lat") or st.session_state.norm_facilities is None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("📍 먼저 **사이드바**에서 출발 위치를 검색하고 편의시설 검색을 실행해 주세요.", icon=None)
        return

    user_lat  = st.session_state.user_lat
    user_lon  = st.session_state.user_lon
    user_cfg  = config.USER_TYPES[st.session_state.get("user_type_radio", "일반")]
    gungu     = st.session_state.get("searched_gungu", "")

    # ── 섹션 1: 주변 장애인 화장실 ─────────────────────────────────
    st.markdown("### 🚻 주변 장애인 화장실")
    st.caption("검색한 지역 내 장애인 화장실 보유 시설을 거리순으로 표시합니다.")

    _toilet_facs = sorted(
        [(haversine(user_lat, user_lon, f["lat"], f["lon"]), f)
         for f in st.session_state.norm_facilities if f.get("has_toilet")],
        key=lambda x: x[0],
    )[:5]

    if not _toilet_facs:
        st.warning("검색된 시설 중 장애인 화장실 보유 시설이 없습니다. 검색 지역을 바꿔보세요.")
    else:
        _tm = folium.Map(location=[user_lat, user_lon], zoom_start=15)
        folium.Marker(
            [user_lat, user_lon], tooltip="현재 위치",
            icon=folium.Icon(color="blue", icon="user", prefix="fa"),
        ).add_to(_tm)
        for _ti, (_td, _tf) in enumerate(_toilet_facs):
            folium.Marker(
                [_tf["lat"], _tf["lon"]],
                tooltip=f"{'🥇' if _ti == 0 else f'#{_ti+1}'} {_tf['name']} ({_td:.0f}m)",
                icon=folium.Icon(color="red" if _ti == 0 else "orange", icon="info-sign"),
            ).add_to(_tm)
        _tn_d, _tn_f = _toilet_facs[0]
        try:
            _tr, _ = _cached_walking_route(user_lat, user_lon, _tn_f["lat"], _tn_f["lon"])
            folium.PolyLine(_tr, color="#1565C0", weight=5).add_to(_tm)
        except Exception:
            folium.PolyLine(
                [[user_lat, user_lon], [_tn_f["lat"], _tn_f["lon"]]],
                color="#1565C0", weight=4, dash_array="8",
            ).add_to(_tm)
        st_folium(_tm, width="100%", height=380, key="toilet_map_t3")
        for _ti, (_td, _tf) in enumerate(_toilet_facs):
            _eta = _td / user_cfg["speed_mpm"]
            st.markdown(
                f"**{'🥇' if _ti == 0 else f'#{_ti+1}'}  {_tf['name']}** &nbsp; "
                f"{_td:.0f}m · 도보 약 {_eta:.0f}분  \n"
                f"<small>{_tf.get('address', '')}</small>",
                unsafe_allow_html=True,
            )

    # ── 섹션 2: 지하철 엘리베이터 현황 ──────────────────────────────
    if not config.SEOUL_OPEN_KEY:
        return

    st.divider()
    st.markdown("### 🚇 지하철 엘리베이터 현황 (서울)")
    st.caption("서울교통공사 실시간 엘리베이터 운행 정보입니다. (5분 캐시)")

    _elev_rows = _cached_subway_elevators(config.SEOUL_OPEN_KEY)
    if not _elev_rows:
        st.warning("서울 열린데이터광장 API 응답이 없습니다.")
        return

    # 검색한 지역(gungu)으로 기본 필터 — 사용자가 수정 가능
    _default_q = gungu or ""
    _search_station = st.text_input(
        "역 이름 검색",
        value=_default_q,
        placeholder="예: 강남, 홍대입구",
        key="subway_elev_search_t3",
        help="검색한 지역 이름이 자동 입력됩니다. 직접 수정할 수 있습니다.",
    )

    _filtered = [
        r for r in _elev_rows
        if not _search_station or _search_station in r.get("STATN_NM", "")
    ][:60]

    if not _filtered:
        st.info("검색 결과가 없습니다. 역 이름을 수정해 보세요.")
        return

    _normal = sum(1 for r in _filtered if "정상" in r.get("ELVTR_STTS", ""))
    _broken = len(_filtered) - _normal
    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("🔍 조회 엘리베이터", f"{len(_filtered)}대")
    _c2.metric("✅ 정상 운행", f"{_normal}대")
    _c3.metric("🔴 고장·점검", f"{_broken}대",
               delta=f"-{_broken}" if _broken else None,
               delta_color="inverse")

    if _broken:
        st.error(f"⚠️ 고장·점검 중인 엘리베이터 {_broken}대가 있습니다. 이동 전 확인해 주세요.")

    st.markdown("---")
    for _r in _filtered:
        _stts = _r.get("ELVTR_STTS", "")
        _icon = "✅" if "정상" in _stts else "🔴"
        st.markdown(
            f"{_icon} **{_r.get('STATN_NM', '')}역** — {_r.get('ELVTR_LOCA', '')} ({_stts})"
        )


# ---------------------------------------------------------------------------
# 복지서비스 탭
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_local_welfare(api_key: str, sido: str) -> list[dict]:
    raw = fetch_local_welfare(api_key, sido, num=100)
    return normalize_welfare(raw, source="지자체")


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_central_welfare(api_key: str) -> list[dict]:
    raw = fetch_central_welfare(api_key, num=100)
    return normalize_welfare(raw, source="중앙부처")


def _tab4_content() -> None:
    """복지 서비스 추천 탭"""
    st.markdown("### 🤝 장애인 복지 서비스 추천")
    st.caption("검색한 지역 기반 지자체 + 중앙부처 복지서비스를 통합 조회합니다.")

    if not config.DATA_GO_KR_KEY:
        st.error("공공데이터포털 API 키(`DATA_GO_KR_KEY`)가 필요합니다.")
        return

    sido = st.session_state.get("searched_sido", "")
    gungu = st.session_state.get("searched_gungu", "")

    if not sido:
        st.info("📍 먼저 **편의시설 검색** 탭에서 위치를 검색해 주세요.")
        return

    st.markdown(f"**검색 지역:** {sido} {gungu}".strip())

    # ── 키워드 필터 ───────────────────────────────────────────────
    kw_filter = st.text_input(
        "서비스명 검색",
        placeholder="예: 활동지원, 보조기기, 취업",
        key="welfare_kw",
    )
    source_filter = st.radio(
        "출처",
        ["전체", "지자체", "중앙부처"],
        horizontal=True,
        key="welfare_source",
    )

    # ── 데이터 조회 ───────────────────────────────────────────────
    with st.spinner("복지서비스 조회 중..."):
        local_items   = _cached_local_welfare(config.DATA_GO_KR_KEY, sido)
        central_items = _cached_central_welfare(config.DATA_GO_KR_KEY)

    if source_filter == "지자체":
        items = local_items
    elif source_filter == "중앙부처":
        items = central_items
    else:
        # [자료구조] List: 지자체+중앙부처 합산, ID 기준 중복 제거
        # [알고리즘] 선형탐색으로 중복 wlfareInfoId 제거 (Set 사용)
        seen: set[str] = set()
        items = []
        for it in local_items + central_items:
            uid = it.get("id", "") or it.get("name", "")
            if uid not in seen:
                seen.add(uid)
                items.append(it)

    # 키워드 필터
    if kw_filter.strip():
        kw = kw_filter.strip()
        items = [it for it in items
                 if kw in it.get("name", "") or kw in it.get("summary", "")]

    if not items:
        st.warning(
            "조회된 장애인 복지서비스가 없습니다.\n\n"
            "- 공공데이터포털에서 해당 API 활용 신청이 완료되었는지 확인해 주세요.\n"
            "- 신청 링크: https://www.data.go.kr/data/15108347/openapi.do\n"
            "- 승인 즉시 사용 가능 (자동승인)"
        )
        return

    st.success(f"**{len(items)}개** 서비스 조회됨")

    for it in items:
        with st.expander(
            f"**{it['name']}** "
            f"{'🏛️' if it['source'] == '중앙부처' else '🏢'} {it['source']}"
        ):
            if it.get("summary"):
                st.markdown(f"**개요**  \n{it['summary']}")
            if it.get("target"):
                st.markdown(f"**지원 대상**  \n{it['target']}")
            if it.get("criteria"):
                st.markdown(f"**선정 기준**  \n{it['criteria']}")
            if it.get("apply"):
                st.markdown(f"**신청 방법**  \n{it['apply']}")
            if it.get("contact"):
                st.markdown(f"**문의**  \n{it['contact']}")
            if it.get("theme"):
                st.caption(f"주제: {it['theme']}")


# ---------------------------------------------------------------------------
# 취업정보 탭
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=600)  # 10분 — 실시간 구인
def _cached_jobs(api_key: str, sido: str, category: str) -> list[dict]:
    raw = fetch_disability_jobs(api_key, sido=sido, job_category=category, num=50)
    return normalize_jobs(raw)


_JOB_CATEGORIES = [
    "전체", "사무직", "서비스직", "판매직", "생산직", "전문직",
    "IT·정보통신", "디자인", "교육", "의료·복지", "기타",
]


def _tab5_content() -> None:
    """장애인 취업 정보 탭"""
    st.markdown("### 💼 장애인 구인 현황")
    st.caption("한국장애인고용공단 실시간 구인 정보 (10분 캐시)")

    if not config.DATA_GO_KR_KEY:
        st.error("공공데이터포털 API 키(`DATA_GO_KR_KEY`)가 필요합니다.")
        return

    sido = st.session_state.get("searched_sido", "")

    # ── 필터 ─────────────────────────────────────────────────────
    _fc1, _fc2 = st.columns([2, 1])
    with _fc1:
        sido_input = st.text_input(
            "지역",
            value=sido,
            placeholder="예: 서울, 경기",
            key="job_sido_input",
        )
    with _fc2:
        cat_sel = st.selectbox("직종", _JOB_CATEGORIES, key="job_category_sel")

    search_jobs_btn = st.button("🔍 구인 정보 조회", type="primary", key="job_search_btn")
    if search_jobs_btn:
        st.session_state["job_search_trigger"] = {
            "sido": sido_input,
            "category": "" if cat_sel == "전체" else cat_sel,
        }

    trigger = st.session_state.get("job_search_trigger")
    if not trigger:
        if sido:
            # 편의시설 검색한 지역이 있으면 자동으로 첫 조회
            st.session_state["job_search_trigger"] = {"sido": sido, "category": ""}
            st.rerun()
        else:
            st.info("📍 지역을 입력하거나 편의시설 검색 탭에서 위치를 먼저 검색해 주세요.")
            return

    with st.spinner("구인 정보 조회 중..."):
        jobs = _cached_jobs(
            config.DATA_GO_KR_KEY,
            trigger["sido"],
            trigger["category"],
        )

    if not jobs:
        st.warning(
            "구인 정보가 없습니다.\n\n"
            "- 공공데이터포털에서 해당 API 활용 신청을 완료해 주세요.\n"
            "- 신청 링크: https://www.data.go.kr/data/15117692/openapi.do\n"
            "- 승인 즉시 사용 가능 (자동승인)"
        )
        return

    region_label = trigger["sido"] or "전국"
    cat_label    = trigger["category"] or "전체 직종"
    st.success(f"**{region_label}** · **{cat_label}** 구인 **{len(jobs)}건**")

    for job in jobs:
        deadline = job.get("deadline", "")
        _badge = f"⏰ {deadline}" if deadline else ""
        with st.expander(
            f"**{job.get('company', '회사명 없음')}** — {job.get('job', '직종 미상')} {_badge}"
        ):
            _j1, _j2 = st.columns(2)
            with _j1:
                if job.get("employ"):
                    st.markdown(f"**고용형태** {job['employ']}")
                if job.get("salary"):
                    st.markdown(f"**급여** {job['salary']}")
                if job.get("career"):
                    st.markdown(f"**경력** {job['career']}")
                if job.get("edu"):
                    st.markdown(f"**학력** {job['edu']}")
            with _j2:
                if job.get("count"):
                    st.markdown(f"**채용인원** {job['count']}명")
                if job.get("biz_type"):
                    st.markdown(f"**기업유형** {job['biz_type']}")
                if job.get("contact"):
                    st.markdown(f"**문의** {job['contact']}")
            if job.get("address"):
                st.caption(f"📍 {job['address']}")


# ---------------------------------------------------------------------------
# 탭 구성 — 편의시설 검색 | 길찾기 | 접근성 정보 | 복지서비스 | 취업정보
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ 편의시설 검색", "🧭 길찾기", "♿ 접근성 정보", "🤝 복지 서비스", "💼 취업 정보",
])

with tab1:
    _tab1_content()

with tab2:
    st.markdown("### 🧭 길찾기")
    st.caption("출발지에서 목적지까지 보행 경로와 경로 주변 접근 가능한 편의시설을 안내합니다.")

    if not st.session_state.get("selected_place"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("📍 먼저 **사이드바**에서 출발 위치를 검색하고 선택해 주세요.", icon=None)
    else:
        _sp = st.session_state.selected_place

        # ── 출발지 / 목적지 입력 ────────────────────────────────
        _nav_c1, _nav_c2 = st.columns(2)
        with _nav_c1:
            st.markdown("**🚶 출발지**")
            st.success(f"**{_sp['name']}**\n\n📍 {_sp['address']}")

        with _nav_c2:
            st.markdown("**🏁 목적지**")
            _dest_q = st.text_input(
                "목적지 검색",
                placeholder="장소명 또는 주소 입력 (예: 강남구청)",
                label_visibility="collapsed",
                key="dest_query_nav",
            )
            _dest_search_btn = st.button("🔎 검색", key="dest_search_btn_nav", use_container_width=True)

        if _dest_search_btn:
            if not _dest_q.strip():
                st.warning("목적지를 입력해 주세요.")
            elif not config.KAKAO_REST_KEY:
                st.error("카카오 API 키가 없습니다.")
            else:
                with st.spinner("목적지 검색 중..."):
                    _dr = search_places(_dest_q, config.KAKAO_REST_KEY, max_results=5)
                st.session_state.route_dest_results   = _dr
                st.session_state.route_dest_place     = None
                st.session_state.route_nav_coords     = None
                st.session_state.route_nav_facilities = None
                st.session_state.route_nav_elev_cnt   = None

        if st.session_state.route_dest_results is not None:
            _dr = st.session_state.route_dest_results
            if not _dr:
                st.warning("검색 결과가 없습니다. 다른 검색어를 시도해 보세요.")
            else:
                st.caption("목적지를 선택하세요")
                _sel_dest_idx = st.radio(
                    "목적지 선택",
                    options=range(len(_dr)),
                    format_func=lambda i: (
                        f"{_dr[i]['name']}  "
                        + (f"·  {_dr[i]['category']}  " if _dr[i]["category"] else "")
                        + f"📍 {_dr[i]['address']}"
                    ),
                    label_visibility="collapsed",
                    key="dest_radio_nav",
                )
                st.session_state.route_dest_place = _dr[_sel_dest_idx]

        if st.session_state.route_dest_place:
            _dest = st.session_state.route_dest_place

            st.divider()
            _nav_u_cfg = config.USER_TYPES[user_type]
            st.caption(f"이용자 유형: {_USER_TYPE_LABEL.get(user_type, user_type)}")
            if _nav_u_cfg["accessible_path"]:
                st.info("♿ 경로 주변에서 엘리베이터·경사로가 있는 시설을 우선 표시합니다.")

            _nav_route_btn = st.button(
                "🧭 경로 탐색", type="primary", key="nav_route_btn", use_container_width=True,
            )

            if _nav_route_btn:
                _o_lat, _o_lon = _sp["lat"], _sp["lon"]
                _d_lat, _d_lon = _dest["lat"], _dest["lon"]

                _nav_route = None
                _nav_elev_cnt = 0
                _nav_route_err = ""
                _is_wheelchair = (user_type == "휠체어 사용자")
                with st.spinner("🗺️ 보행 경로 계산 중… (최대 60초)"):
                    try:
                        _nav_route, _nav_elev_cnt = _cached_walking_route(
                            _o_lat, _o_lon, _d_lat, _d_lon,
                            wheelchair=_is_wheelchair,
                        )
                    except Exception as _e:
                        _nav_route_err = str(_e)

                if not _nav_route:
                    if _is_wheelchair:
                        _err_msg = "♿ 계단 없는 접근 경로를 찾을 수 없습니다."
                    else:
                        _err_msg = "경로를 찾을 수 없습니다. 두 지점이 너무 멀거나 도로 데이터가 없을 수 있습니다."
                    if _nav_route_err:
                        _err_msg += f"\n\n오류 상세: {_nav_route_err}"
                    st.error(_err_msg)
                else:
                    st.session_state.route_nav_coords   = _nav_route
                    st.session_state.route_nav_elev_cnt = _nav_elev_cnt

                    # 경로 5개 지점(0·25·50·75·100%)에서 역지오코딩 → 구 중복 제거 후 fetch
                    _n = len(_nav_route)
                    _fetch_pts = [
                        (_o_lat, _o_lon),
                        (_nav_route[max(0, _n // 4)][0],     _nav_route[max(0, _n // 4)][1]),
                        (_nav_route[_n // 2][0],             _nav_route[_n // 2][1]),
                        (_nav_route[min(_n - 1, 3 * _n // 4)][0],
                         _nav_route[min(_n - 1, 3 * _n // 4)][1]),
                        (_d_lat, _d_lon),
                    ]

                    with st.spinner("🏢 경로 주변 편의시설 탐색 중..."):
                        _all_raw: list[dict] = []
                        _seen_ids: set[str]  = set()
                        _seen_regions: set[tuple[str, str]] = set()

                        if config.DATA_GO_KR_KEY:
                            for _plat, _plon in _fetch_pts:
                                try:
                                    _r_sido, _r_gungu = reverse_geocode(
                                        _plat, _plon, config.KAKAO_REST_KEY
                                    )
                                except Exception:
                                    continue
                                if not _r_sido:
                                    continue
                                if (_r_sido, _r_gungu) in _seen_regions:
                                    continue  # 같은 구는 한 번만 fetch
                                _seen_regions.add((_r_sido, _r_gungu))
                                _part = fetch_facilities(
                                    config.DATA_GO_KR_KEY, _r_sido, _r_gungu,
                                    num=500, user_lat=_plat, user_lon=_plon,
                                )
                                for _item in _part:
                                    _fid = _item.get("wfcltId", "")
                                    if _fid and _fid in _seen_ids:
                                        continue
                                    if _fid:
                                        _seen_ids.add(_fid)
                                    _all_raw.append(_item)

                        _r_norm = normalize(_all_raw, kakao_key=config.KAKAO_REST_KEY)

                        # 주거시설(아파트·다세대·연립) 제외 — 전체 데이터의 ~75%를 차지하며
                        # 보행자에게 유용하지 않아 유용한 공공·상업 시설을 묻어버림
                        _RESIDENTIAL = {
                            "아파트", "아파트 부대복리시설", "연립주택",
                            "다세대주택", "기숙사",
                        }
                        _r_norm = [
                            f for f in _r_norm
                            if f.get("fac_type", "") not in _RESIDENTIAL
                        ]

                        # 경로를 50m 간격으로 샘플링 → 경로 전체 균등 커버
                        _NEAR_M   = 400
                        _r_sample = [_nav_route[0]]
                        _r_acc    = 0.0
                        for _ri in range(1, len(_nav_route)):
                            _r_acc += haversine(
                                _nav_route[_ri - 1][0], _nav_route[_ri - 1][1],
                                _nav_route[_ri][0],     _nav_route[_ri][1],
                            )
                            if _r_acc >= 50:
                                _r_sample.append(_nav_route[_ri])
                                _r_acc = 0.0
                        if _nav_route[-1] != _r_sample[-1]:
                            _r_sample.append(_nav_route[-1])

                        def _near_route_fn(f: dict) -> bool:
                            return any(
                                haversine(f["lat"], f["lon"], rp[0], rp[1]) <= _NEAR_M
                                for rp in _r_sample
                            )

                        st.session_state.route_nav_facilities = [
                            f for f in _r_norm if _near_route_fn(f)
                        ]

                    # Overpass API: 경로 주변 보행 인프라 (엘리베이터·보도턱·경사로)
                    with st.spinner("🗺️ 보행 인프라 정보 조회 중… (OSM Overpass)"):
                        # 경로 좌표를 튜플로 변환하여 캐시 키로 사용
                        _route_key = tuple(st.session_state.route_nav_coords)
                        st.session_state.route_nav_infra = _cached_overpass(_route_key)

            # ── 경로 지도 렌더링 ────────────────────────────────
            if st.session_state.route_nav_coords:
                _nav_route    = st.session_state.route_nav_coords
                _nav_facs     = st.session_state.route_nav_facilities or []
                _nav_elev_cnt = st.session_state.route_nav_elev_cnt or 0
                _nav_u_cfg    = config.USER_TYPES[user_type]
                _is_wheelchair = (user_type == "휠체어 사용자")

                # 경로 통계 — 휠체어 모드는 엘리베이터 대기시간(2분/회) 추가
                _ELEVATOR_WAIT_MIN = 2
                _nav_dist_m, _nav_segs = _calc_route_stats(_nav_route)
                _nav_eta = (
                    _nav_dist_m / _nav_u_cfg["speed_mpm"]
                    + (_nav_elev_cnt * _ELEVATOR_WAIT_MIN if _is_wheelchair else 0)
                )
                _nav_dist_lbl = (
                    f"{_nav_dist_m / 1000:.2f} km" if _nav_dist_m >= 1000
                    else f"{_nav_dist_m:.0f} m"
                )
                _nav_acc_cnt = sum(
                    1 for f in _nav_facs if f.get("has_elevator") or f.get("has_ramp")
                )

                _nav_eta_lbl = f"약 {_nav_eta:.0f}분"
                if _is_wheelchair and _nav_elev_cnt:
                    _nav_eta_lbl += f" (🛗×{_nav_elev_cnt})"

                _nm1, _nm2, _nm3, _nm4 = st.columns(4)
                _nm1.metric("📏 총 거리",       _nav_dist_lbl)
                _nm2.metric("⏱️ 예상 소요",      _nav_eta_lbl)
                _nm3.metric("🏢 경로 주변 시설", f"{len(_nav_facs)}개")
                _nm4.metric("♿ 접근 가능",       f"{_nav_acc_cnt}개")

                # Folium 지도
                _nav_mid = _nav_route[len(_nav_route) // 2]
                _nav_m   = folium.Map(location=_nav_mid, zoom_start=15, tiles="CartoDB positron")

                # 경로 폴리라인
                folium.PolyLine(
                    _nav_route, color="#2196F3", weight=5, opacity=0.85,
                    tooltip="보행 경로",
                ).add_to(_nav_m)

                # Overpass 보행 인프라 마커 (DivIcon — 시설 CircleMarker와 시각적 구분)
                _nav_infra = st.session_state.route_nav_infra or []
                for _inf in _nav_infra:
                    _inf_html = (
                        f'<div style="'
                        f'background:{_inf["color"]};color:#fff;'
                        f'border-radius:50%;width:26px;height:26px;'
                        f'display:flex;align-items:center;justify-content:center;'
                        f'font-size:14px;border:2px solid #fff;'
                        f'box-shadow:0 2px 5px rgba(0,0,0,0.4);">'
                        f'{_inf["emoji"]}</div>'
                    )
                    _inf_name = _inf["name"] or _inf["label"]
                    _inf_popup = (
                        f'<b>{_inf["label"]}</b>'
                        + (f'<br>{_inf["name"]}' if _inf["name"] else "")
                        + '<br><small style="opacity:0.6;">OSM 보행 인프라</small>'
                    )
                    folium.Marker(
                        [_inf["lat"], _inf["lon"]],
                        icon=folium.DivIcon(
                            html=_inf_html,
                            icon_size=(26, 26),
                            icon_anchor=(13, 13),
                        ),
                        tooltip=_inf_name,
                        popup=folium.Popup(_inf_popup, max_width=180),
                    ).add_to(_nav_m)

                # 출발지 마커
                folium.Marker(
                    [_sp["lat"], _sp["lon"]],
                    popup=folium.Popup(f"<b>출발:</b> {_sp['name']}", max_width=200),
                    tooltip=f"출발: {_sp['name']}",
                    icon=folium.Icon(color="blue", icon="play", prefix="fa"),
                ).add_to(_nav_m)

                # 목적지 마커
                folium.Marker(
                    [_dest["lat"], _dest["lon"]],
                    popup=folium.Popup(f"<b>목적지:</b> {_dest['name']}", max_width=200),
                    tooltip=f"목적지: {_dest['name']}",
                    icon=folium.Icon(color="red", icon="flag", prefix="fa"),
                ).add_to(_nav_m)

                # 경로 주변 편의시설 마커
                # 마커 색상: 🔵 파랑(엘리베이터+경사로) | 🟣 보라(엘리베이터) | 🟢 초록(경사로) | 🟠 주황(일반)
                for _nf in _nav_facs:
                    _has_elev = bool(_nf.get("has_elevator"))
                    _has_ramp = bool(_nf.get("has_ramp"))
                    if _has_elev and _has_ramp:
                        _m_color   = "#0288D1"
                        _acc_label = "🛗♿ 엘리베이터+경사로"
                    elif _has_elev:
                        _m_color   = "#8E24AA"
                        _acc_label = "🛗 엘리베이터"
                    elif _has_ramp:
                        _m_color   = "#43A047"
                        _acc_label = "♿ 경사로"
                    else:
                        _m_color   = "#FB8C00"
                        _acc_label = ""

                    _bdg_parts: list[str] = []
                    if _nf.get("has_toilet"):
                        _bdg_parts.append("🚻 화장실")
                    if _has_elev:
                        _bdg_parts.append("🛗 엘리베이터")
                    if _has_ramp:
                        _bdg_parts.append("♿ 경사로")
                    if _nf.get("has_parking"):
                        _bdg_parts.append("🅿️ 주차")
                    _popup_html = (
                        f"<b>{_nf['name']}</b>"
                        + (f"<br><span style='color:{_m_color};font-weight:600;'>{_acc_label}</span>" if _acc_label else "")
                        + f"<br><small>{_nf.get('address', '')}</small>"
                        + (("<br>" + "  ".join(_bdg_parts)) if _bdg_parts else "")
                    )
                    _tooltip_text = _nf["name"] + (f"  {_acc_label}" if _acc_label else "")
                    folium.CircleMarker(
                        location=[_nf["lat"], _nf["lon"]],
                        radius=8,
                        color=_m_color,
                        fill=True,
                        fill_color=_m_color,
                        fill_opacity=0.8,
                        popup=folium.Popup(_popup_html, max_width=240),
                        tooltip=_tooltip_text,
                    ).add_to(_nav_m)

                st_folium(_nav_m, width="100%", height=520, key="nav_route_map")

                # 범례
                st.markdown(
                    '<div style="display:flex;gap:16px;font-size:0.82em;margin-top:4px;flex-wrap:wrap;">'
                    '<span>🔵 출발지</span>'
                    '<span>🔴 목적지</span>'
                    '<span><span style="color:#0288D1;font-size:1.3em;">●</span> 엘리베이터+경사로</span>'
                    '<span><span style="color:#8E24AA;font-size:1.3em;">●</span> 엘리베이터(시설)</span>'
                    '<span><span style="color:#43A047;font-size:1.3em;">●</span> 경사로(시설)</span>'
                    '<span><span style="color:#FB8C00;font-size:1.3em;">●</span> 일반 편의시설</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                # Overpass 인프라 범례 (데이터 있을 때만 표시)
                if _nav_infra:
                    _infra_counts = {}
                    for _i in _nav_infra:
                        _infra_counts[_i["feat_type"]] = _infra_counts.get(_i["feat_type"], 0) + 1
                    _infra_legend = "".join(
                        f'<span style="display:inline-flex;align-items:center;gap:4px;">'
                        f'<span style="background:{c};color:#fff;border-radius:50%;'
                        f'width:18px;height:18px;display:flex;align-items:center;'
                        f'justify-content:center;font-size:10px;">{e}</span>'
                        f'<span>{lb} ({_infra_counts.get(k, 0)}곳)</span></span>'
                        for k, (lb, c, e) in [
                            ("elevator",     ("엘리베이터",  "#3949AB", "🛗")),
                            ("kerb_lowered", ("보도턱 낮춤", "#00838F", "♿")),
                            ("ramp",         ("경사로",      "#E65100", "⬆")),
                        ]
                        if _infra_counts.get(k, 0) > 0
                    )
                    st.markdown(
                        f'<div style="font-size:0.78em;margin-top:4px;opacity:0.75;">'
                        f'OSM 보행 인프라: <span style="display:inline-flex;gap:14px;flex-wrap:wrap;">'
                        f'{_infra_legend}</span></div>',
                        unsafe_allow_html=True,
                    )
                elif st.session_state.route_nav_infra is not None:
                    st.caption("OSM 보행 인프라 데이터 없음 (이 구간은 태깅되지 않았습니다)")

                # 시설 목록 (접이식)
                if _nav_facs:
                    with st.expander(
                        f"📋 경로 주변 편의시설 목록 ({len(_nav_facs)}개)", expanded=False
                    ):
                        # 정렬: 엘리베이터+경사로 둘 다 > 엘리베이터만 > 경사로만 > 일반
                        def _acc_rank(f: dict) -> int:
                            e, r = bool(f.get("has_elevator")), bool(f.get("has_ramp"))
                            return 0 if (e and r) else (1 if e else (2 if r else 3))
                        _sorted_facs = sorted(_nav_facs, key=_acc_rank)

                        _r_sample_e = _nav_route[::max(1, len(_nav_route) // 20)]
                        for _nf in _sorted_facs:
                            _he = bool(_nf.get("has_elevator"))
                            _hr = bool(_nf.get("has_ramp"))
                            if _he and _hr:
                                _dot_color = "#0288D1"
                                _acc_tag   = "🛗♿ 엘리베이터+경사로"
                            elif _he:
                                _dot_color = "#8E24AA"
                                _acc_tag   = "🛗 엘리베이터"
                            elif _hr:
                                _dot_color = "#43A047"
                                _acc_tag   = "♿ 경사로"
                            else:
                                _dot_color = "#FB8C00"
                                _acc_tag   = ""

                            _bdg_html = ""
                            if _nf.get("has_toilet"):
                                _bdg_html += '<span class="bdg bdg-t">🚻 화장실</span> '
                            if _he:
                                _bdg_html += '<span class="bdg bdg-e">🛗 엘리베이터</span> '
                            if _hr:
                                _bdg_html += '<span class="bdg bdg-r">♿ 경사로</span> '
                            if _nf.get("has_parking"):
                                _bdg_html += '<span class="bdg bdg-p">🅿️ 주차</span> '
                            _d_to_route = min(
                                haversine(_nf["lat"], _nf["lon"], rp[0], rp[1])
                                for rp in _r_sample_e
                            )
                            _acc_badge_html = (
                                f'<span style="font-size:0.78em;font-weight:600;color:{_dot_color};">'
                                f'{_acc_tag}</span>  ' if _acc_tag else ""
                            )
                            st.markdown(
                                f'<div class="bfn-card" style="margin-bottom:8px;'
                                f'border-left:4px solid {_dot_color};">'
                                f'<div style="font-weight:600;">{_nf["name"]}</div>'
                                f'<div style="margin:2px 0;">{_acc_badge_html}'
                                f'<span class="bfn-addr">📍 {_nf.get("address", "")}</span></div>'
                                f'<div style="margin-top:6px;">{_bdg_html}</div>'
                                f'<div style="font-size:0.78em;opacity:0.55;margin-top:4px;">'
                                f'경로에서 약 {_d_to_route:.0f}m</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

with tab3:
    _tab3_content()

with tab4:
    _tab4_content()

with tab5:
    _tab5_content()
