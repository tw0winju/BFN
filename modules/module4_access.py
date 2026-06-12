# module4_access.py — 팀원 4
# 담당: 교통약자 접근 경로 탐색 + UI 헬퍼
# [자료구조] Queue (collections.deque), Set
# [알고리즘] BFS 제약 탐색 (접근 가능 노드만 통과)
# [알고리즘] 접근성 제약 Dijkstra (휠체어 실거리 최단경로)

import heapq
from collections import deque

import folium
import streamlit as st


def is_accessible(facility: dict) -> bool:
    """접근 가능 노드 판별: 엘리베이터 또는 경사로 보유 시 True

    휠체어 모드에서 중간 경유 노드로 허용되는 기준이다.
    """
    return facility.get("has_elevator", False) or facility.get("has_ramp", False)


def dijkstra_accessible(
    graph: dict,
    nodes: list[dict],
    start: int = 0,
) -> tuple[dict, dict, int, list[int]]:
    """접근 가능 노드만 통과하는 단일 출발 최단거리 (휠체어 모드 전용)

    BFS(최소 홉)와 달리 엣지 가중치(실거리)를 반영하므로 실제 이동 거리를 최소화한다.
    비접근 노드(엘리베이터·경사로 없는 중간 시설)는 릴렉스 대상에서 제외한다.
    start 노드(사용자 위치)와 목적지는 접근성 검사를 생략한다.

    [자료구조] Priority Queue (heapq): 최단거리 노드 우선 처리
    [알고리즘] 접근성 제약 Dijkstra:
        표준 Dijkstra에서 비접근 노드로 향하는 엣지를 건너뜀.
        BFS(최소 홉)와 달리 이동 거리(m) 최소화를 보장한다.
        시간복잡도 O((V + E) log V)

    Returns:
        dist: 각 노드까지의 접근 가능 최단 거리
        prev: 경로 복원용 이전 노드
        visited_count: 처리된 노드 수
        visited_order: 처리 순서 (탐색 시각화용)
    """
    INF = float("inf")
    dist: dict[int, float] = {node: INF for node in graph}
    dist[start] = 0.0
    prev: dict[int, int | None] = {node: None for node in graph}

    # [자료구조] Priority Queue (heapq): (거리, 노드ID) 최소 힙
    pq: list[tuple[float, int]] = [(0.0, start)]
    visited_count = 0
    visited_order: list[int] = []

    while pq:
        cur_dist, u = heapq.heappop(pq)
        if cur_dist > dist[u]:
            continue
        visited_count += 1
        visited_order.append(u)

        # 비접근 시설 노드는 목적지는 될 수 있지만 경유지로 확장하지 않음
        if u > 0 and u != start and not is_accessible(nodes[u]):
            continue

        for v, weight in graph[u].items():
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, prev, visited_count, visited_order


def bfs_accessible_path(
    graph: dict,
    nodes: list[dict],
    start: int,
    goal: int,
) -> tuple[list[int] | None, int, list[int]]:
    """접근 가능 노드만 통과하는 최소 홉 경로 탐색

    휠체어 모드 전용. 엘리베이터 또는 경사로가 없는 중간 노드는
    방문 대상에서 제외한다. start·goal 자체는 접근성 검사를 생략한다.

    **BFS를 선택한 이유 (Dijkstra 대신):**
    Dijkstra는 엣지 가중치(거리)를 최소화하지만, 접근 불가 노드를
    단순히 제외하면 '가중치가 작은 비접근 노드'를 우선 처리하다가
    탐색 방향이 왜곡될 수 있다.
    휠체어 이용자에게 실질적으로 중요한 것은 '경유해야 할 편의시설
    수(홉)를 최소화'하는 것이다 — 각 경유지마다 접근 여부를 확인해야
    하므로 경유 횟수를 줄이는 것이 피로도·실용성 면에서 핵심이다.
    BFS는 접근 가능 노드로만 구성된 제약 그래프에서 최소 홉을 O(V+E)
    에 보장하며, 이 목적에 Dijkstra보다 적합하다.

    [자료구조] Queue (deque): FIFO 큐로 BFS 탐색 순서 관리
    [자료구조] Set: 방문한 노드를 O(1) 에 확인하여 중복 방문 방지
    [알고리즘] BFS 제약 탐색:
        너비 우선 탐색으로 최소 홉 경로를 보장한다.
        단, 접근 불가 노드(휠체어 통과 불가)는 큐에 추가하지 않는다.
        시간복잡도 O(V + E)

    Returns:
        path: 경로 노드 ID 리스트 (없으면 None)
        visited_count: 실제 방문(처리)한 노드 수 (알고리즘 비교용)
        visited_order: 방문 순서 노드 ID 리스트 (탐색 시각화용)
    """
    if start == goal:
        return [start], 1, [start]

    # [자료구조] Queue (deque): BFS 탐색용 FIFO 큐
    queue: deque[tuple[int, list[int]]] = deque()
    queue.append((start, [start]))

    # [자료구조] Set: 방문 노드 집합 (중복 방문 방지, O(1) 조회)
    visited: set[int] = {start}
    visited_order: list[int] = [start]

    while queue:
        current, path = queue.popleft()

        for neighbor in graph.get(current, {}):
            if neighbor in visited:
                continue

            # 목표 노드에 도달하면 즉시 반환 (BFS → 최소 홉 보장)
            if neighbor == goal:
                visited_order.append(neighbor)
                return path + [neighbor], len(visited_order), visited_order

            # node 0 = 사용자 위치(start)이므로 nodes 인덱스는 neighbor-1
            # 중간 노드: 접근 가능 여부 검사 (엘리베이터 or 경사로 필요)
            if neighbor > 0:
                fac = nodes[neighbor - 1]  # nodes[0] = facilities[0]
                if not is_accessible(fac):
                    visited.add(neighbor)
                    continue

            visited.add(neighbor)
            visited_order.append(neighbor)
            queue.append((neighbor, path + [neighbor]))

    return None, len(visited_order), visited_order  # 접근 가능한 경로 없음


# ---------------------------------------------------------------------------
# UI 헬퍼
# ---------------------------------------------------------------------------

def render_map(
    facilities: list[dict],
    recommended: list[dict],
    user_lat: float,
    user_lon: float,
    route_coords: list[tuple[float, float]] | None = None,
    goal_fac: dict | None = None,
) -> folium.Map:
    """Folium 지도 생성

    - 파란 마커: 사용자 위치
    - 빨간 마커: 선택된 목적지 시설 (goal_fac 지정 시) 또는 추천 TOP-N 전체
    - 파란 선: 경로 (route_coords 가 있을 때)
    goal_fac 이 주어지면 해당 시설 마커만 표시하고 나머지 마커는 생략한다.
    """
    m = folium.Map(location=[user_lat, user_lon], zoom_start=15)

    # 사용자 위치 — 파란 마커
    folium.Marker(
        location=[user_lat, user_lon],
        tooltip="내 위치",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    if goal_fac is not None:
        # 선택된 목적지 시설만 표시
        popup_html = (
            f"<b>{goal_fac['name']}</b><br>"
            f"{goal_fac.get('address', '')}<br>"
            f"{'🚻 ' if goal_fac.get('has_toilet') else ''}"
            f"{'🛗 ' if goal_fac.get('has_elevator') else ''}"
            f"{'♿ ' if goal_fac.get('has_ramp') else ''}"
            f"{'🅿️ ' if goal_fac.get('has_parking') else ''}"
        )
        folium.Marker(
            location=[goal_fac["lat"], goal_fac["lon"]],
            tooltip=goal_fac["name"],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)
    else:
        # 목적지 미선택 시: 추천 TOP-N 전체 표시
        recommended_names = {r["name"] for r in recommended}
        for fac in facilities:
            if fac["name"] in recommended_names:
                continue
            folium.CircleMarker(
                location=[fac["lat"], fac["lon"]],
                radius=5,
                color="gray",
                fill=True,
                fill_opacity=0.6,
                tooltip=fac["name"],
            ).add_to(m)
        for rank, fac in enumerate(recommended, start=1):
            popup_html = (
                f"<b>#{rank} {fac['name']}</b><br>"
                f"{fac.get('address', '')}<br>"
                f"점수: {fac.get('score', 0):.4f}<br>"
                f"{'🚻 ' if fac.get('has_toilet') else ''}"
                f"{'🛗 ' if fac.get('has_elevator') else ''}"
                f"{'♿ ' if fac.get('has_ramp') else ''}"
                f"{'🅿️ ' if fac.get('has_parking') else ''}"
            )
            folium.Marker(
                location=[fac["lat"], fac["lon"]],
                tooltip=f"#{rank} {fac['name']}",
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color="red", icon="star", prefix="fa"),
            ).add_to(m)

    # 경로 — 파란 선
    if route_coords:
        folium.PolyLine(
            locations=route_coords,
            color="blue",
            weight=4,
            opacity=0.8,
            tooltip="이동 경로",
        ).add_to(m)

    return m


def render_exploration_map(
    dijk_order: list[int],
    astar_order: list[int],
    nodes: list[dict],
    user_lat: float,
    user_lon: float,
    goal_node: int | None = None,
) -> folium.Map:
    """Dijkstra vs A* 탐색 노드 시각화 지도

    파란 원: Dijkstra 전용 탐색 노드 (A*가 건너뛴 '낭비' 탐색)
    보라 원: 두 알고리즘이 공통으로 탐색한 노드
    주황 원: A* 전용 탐색 노드 (드묾)
    진할수록 먼저 탐색된 노드다.
    """
    m = folium.Map(location=[user_lat, user_lon], zoom_start=15)

    total_d = max(len(dijk_order), 1)
    astar_set = set(astar_order)
    dijk_set  = set(dijk_order)

    # Dijkstra 탐색 노드
    for step, nid in enumerate(dijk_order):
        if nid == 0 or nid >= len(nodes):
            continue
        node    = nodes[nid]
        opacity = 0.2 + 0.65 * (1 - step / total_d)
        color   = "purple" if nid in astar_set else "blue"
        label   = "Dijkstra+A* 공통" if nid in astar_set else "Dijkstra 전용"
        folium.CircleMarker(
            location=[node["lat"], node["lon"]],
            radius=7,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            tooltip=f"{label} #{step + 1}",
        ).add_to(m)

    # A* 전용 탐색 노드 (Dijkstra에 없는 경우)
    for step, nid in enumerate(astar_order):
        if nid == 0 or nid >= len(nodes) or nid in dijk_set:
            continue
        node = nodes[nid]
        folium.CircleMarker(
            location=[node["lat"], node["lon"]],
            radius=5,
            color="orange",
            weight=1,
            fill=True,
            fill_color="orange",
            fill_opacity=0.75,
            tooltip=f"A* 전용 #{step + 1}",
        ).add_to(m)

    # 목적지 마커
    if goal_node is not None and 0 < goal_node < len(nodes):
        gn = nodes[goal_node]
        folium.Marker(
            location=[gn["lat"], gn["lon"]],
            tooltip="목적지",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)

    # 출발지 마커
    folium.Marker(
        location=[user_lat, user_lon],
        tooltip="출발지",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    return m


def render_facility_card(
    facility: dict,
    score: float,
    rank: int,
    reasons: list[str] | None = None,
) -> None:
    """Streamlit 시설 카드 출력 (UX 원칙: 스캔 가능성 · 접근성 · 설명 가능한 결과)

    - 편의시설: 색상 배지(아이콘+텍스트 병행) → 색상만으로 정보 전달 금지 원칙 준수
    - 추천 이유: ✓ 체크리스트 형식으로 명시적으로 표시
    - 순위·거리·이름을 최상단에 배치해 훑어보기 편하게 구성
    """
    # 편의시설 배지: app.py 에서 주입된 .bdg / .bdg-* 클래스 사용
    # → 텍스트는 var(--text-color) 상속, 배경은 반투명 컬러 + 좌측 보더로 구분
    # (접근성: 색+텍스트 병행, 다크/라이트 모드 자동 대응)
    _BADGES = [
        ("has_toilet",   "🚻", "장애인화장실", "bdg-t"),
        ("has_elevator", "🛗", "엘리베이터",   "bdg-e"),
        ("has_ramp",     "♿", "경사로",       "bdg-r"),
        ("has_parking",  "🅿️", "전용주차",     "bdg-p"),
    ]
    badge_html = "".join(
        f'<span class="bdg {cls}">{icon} {label}</span>'
        for key, icon, label, cls in _BADGES
        if facility.get(key)
    ) or '<span style="font-size:0.8em;opacity:0.45;">편의시설 정보 없음</span>'

    # 추천 이유: .bfn-reasons 클래스 → 라이트 #2E7D32 / 다크 #A5D6A7 자동 전환
    reason_html = ""
    if reasons:
        items = "".join(f'<span>✓ {r}</span>' for r in reasons)
        reason_html = f'<div class="bfn-reasons">{items}</div>'

    # 좌측 강조선: TOP3는 파란색, 나머지는 중립 회색
    accent = "#42A5F5" if rank <= 3 else "rgba(128,128,128,0.4)"

    with st.container():
        st.markdown(
            f'<div class="bfn-card" style="border-left:4px solid {accent};">'
            f'  <div style="display:flex;justify-content:space-between;align-items:flex-start;">'
            f'    <div style="flex:1;min-width:0;padding-right:12px;">'
            f'      <p class="bfn-muted" style="margin:0 0 2px 0;font-size:0.78em;font-weight:600;">'
            f'#{rank}위{"  · " + facility["fac_type"] if facility.get("fac_type") else ""}</p>'
            f'      <h4 class="bfn-name" style="margin:0 0 4px 0;font-size:1em;'
            f'          overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{facility["name"]}</h4>'
            f'      <p class="bfn-addr" style="margin:0;">📍 {facility.get("address","주소 없음")}</p>'
            f'    </div>'
            f'    <div style="text-align:right;flex-shrink:0;">'
            f'      <p class="bfn-slabel" style="margin:0;">추천점수</p>'
            f'      <p class="bfn-score" style="margin:0;">{score:.3f}</p>'
            f'    </div>'
            f'  </div>'
            f'  <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:6px;">{badge_html}</div>'
            f'  {reason_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
