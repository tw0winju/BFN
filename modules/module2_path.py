# module2_path.py — 팀원 2
# 담당: 거리 계산 & 최단 경로 탐색
# [자료구조] Graph (dict of dict), Priority Queue (heapq)
# [알고리즘] Haversine, Dijkstra, A*

import concurrent.futures
import heapq
import math

import requests

from config import GRAPH_RADIUS_M

_WALK_TIMEOUT_SEC = 60  # OSMnx 하드 타임아웃 (초)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """위경도 두 점 간 실제 거리(m)

    [알고리즘] Haversine 공식: 구면 좌표 거리 계산
        두 점의 위경도 차이를 이용해 지구 표면 위의 대원 거리를 구한다.
        시간복잡도 O(1)
    """
    R = 6_371_000  # 지구 반지름 (m)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def build_graph(
    facilities: list[dict],
    user_lat: float,
    user_lon: float,
    radius_m: int = GRAPH_RADIUS_M,
) -> dict:
    """반경 기반 시설 네트워크 그래프 생성

    scipy가 설치된 경우 KD-Tree(O(N log N))로 인접 노드를 검색하고,
    없으면 O(N²) 전수 비교로 폴백한다.

    [자료구조] Graph (dict of dict): {node_id: {neighbor_id: distance_m}}
    [알고리즘] KD-Tree 공간 인덱스 (O(N log N)) / O(N²) 폴백
    """
    try:
        from scipy.spatial import KDTree  # noqa: PLC0415
        return _build_graph_kdtree(facilities, user_lat, user_lon, radius_m, KDTree)
    except ImportError:
        return _build_graph_naive(facilities, user_lat, user_lon, radius_m)


def _build_graph_naive(
    facilities: list[dict],
    user_lat: float,
    user_lon: float,
    radius_m: int,
) -> dict:
    """O(N²) 전수 비교 그래프 생성 (KD-Tree 폴백)"""
    nodes = [{"lat": user_lat, "lon": user_lon}] + facilities
    graph: dict[int, dict[int, float]] = {i: {} for i in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dist = haversine(
                nodes[i]["lat"], nodes[i]["lon"],
                nodes[j]["lat"], nodes[j]["lon"],
            )
            if dist <= radius_m:
                graph[i][j] = dist
                graph[j][i] = dist
    return graph


def _build_graph_kdtree(
    facilities: list[dict],
    user_lat: float,
    user_lon: float,
    radius_m: int,
    KDTree,  # scipy.spatial.KDTree (passed to avoid re-import)
) -> dict:
    """KD-Tree 기반 그래프 생성 — O(N log N)

    등장방형 근사(Equirectangular)로 위경도를 미터 단위 평면 좌표로 변환한 뒤
    KDTree.query_pairs 로 반경 내 쌍을 O(N log N) 에 열거한다.
    이후 Haversine 으로 실거리를 검증하여 투영 오차를 제거한다.

    [자료구조] KD-Tree: 공간 분할 이진 트리, 반경 검색 O(N log N)
    [알고리즘] 등장방형 투영 + KD-Tree 범위 검색
    """
    nodes = [{"lat": user_lat, "lon": user_lon}] + facilities
    N = len(nodes)

    # 위경도 → 미터 평면 좌표 (기준점: 사용자 위치)
    R = 6_371_000
    cos_lat = math.cos(math.radians(user_lat))
    xy = [
        (
            (n["lon"] - user_lon) * math.pi / 180 * R * cos_lat,
            (n["lat"] - user_lat) * math.pi / 180 * R,
        )
        for n in nodes
    ]

    # [자료구조] KD-Tree: query_pairs 가 반경 내 (i, j) 쌍을 O(N log N) 반환
    tree = KDTree(xy)
    pairs = tree.query_pairs(radius_m * 1.01)  # 1% 여유: 투영 오차 보정

    graph: dict[int, dict[int, float]] = {i: {} for i in range(N)}
    for i, j in pairs:
        dist = haversine(
            nodes[i]["lat"], nodes[i]["lon"],
            nodes[j]["lat"], nodes[j]["lon"],
        )
        if dist <= radius_m:  # Haversine 정밀 검증
            graph[i][j] = dist
            graph[j][i] = dist

    return graph


def dijkstra(graph: dict, start: int = 0) -> tuple[dict, dict, int, list[int]]:
    """시작 노드에서 전체 노드까지 최단거리 & 이전 노드 반환

    [자료구조] Priority Queue (heapq): 현재까지의 최단거리 기준으로
        다음 방문 노드를 O(log N) 에 선택한다.
    [알고리즘] Dijkstra 최단경로:
        음수 가중치가 없는 그래프에서 단일 출발지 최단경로를 구한다.
        시간복잡도 O((V + E) log V)

    Returns:
        dist: 각 노드까지의 최단 거리
        prev: 경로 복원용 이전 노드
        visited_count: 실제로 처리(확정)된 노드 수 (알고리즘 비교용)
        visited_order: 처리된 노드 ID 순서 (탐색 시각화용)
    """
    INF = float("inf")
    # [자료구조] Dict: node_id → 최단 거리
    dist: dict[int, float] = {node: INF for node in graph}
    dist[start] = 0.0
    # 이전 노드 추적 (경로 복원용)
    prev: dict[int, int | None] = {node: None for node in graph}

    # [자료구조] Priority Queue (heapq): (거리, 노드) 최소 힙
    pq: list[tuple[float, int]] = [(0.0, start)]
    visited_count = 0
    visited_order: list[int] = []

    while pq:
        cur_dist, u = heapq.heappop(pq)
        if cur_dist > dist[u]:
            continue  # 이미 처리된 노드
        visited_count += 1
        visited_order.append(u)
        for v, weight in graph[u].items():
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, prev, visited_count, visited_order


def astar(graph: dict, start: int, goal: int, nodes: list[dict]) -> tuple[list[int] | None, int, list[int]]:
    """A* 최단경로 탐색

    휴리스틱: 목표 노드까지의 Haversine 직선거리 (admissible — 실제 거리 초과 불가).

    [자료구조] Priority Queue (heapq): f = g + h 기준 최소 힙
    [알고리즘] A* 최단경로:
        Dijkstra에 목표 방향 휴리스틱을 추가하여 탐색 범위를 줄인다.
        시간복잡도 O((V + E) log V) (worst case), 실질적으로 Dijkstra보다 빠름

    Returns:
        path: 경로 노드 ID 리스트 (없으면 None)
        visited_count: 처리된 노드 수 (알고리즘 비교용)
        visited_order: 처리된 노드 ID 순서 (탐색 시각화용)
    """
    INF = float("inf")

    def h(node_id: int) -> float:
        return haversine(
            nodes[node_id]["lat"], nodes[node_id]["lon"],
            nodes[goal]["lat"], nodes[goal]["lon"],
        )

    # g[n]: start → n 까지 실제 비용
    g: dict[int, float] = {node: INF for node in graph}
    g[start] = 0.0
    came_from: dict[int, int | None] = {node: None for node in graph}

    # [자료구조] Priority Queue (heapq): (f, node)
    open_set: list[tuple[float, int]] = [(h(start), start)]
    visited_count = 0
    visited_order: list[int] = []

    while open_set:
        _, u = heapq.heappop(open_set)
        visited_count += 1
        visited_order.append(u)

        if u == goal:
            # 경로 복원
            path: list[int] = []
            cur: int | None = goal
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1], visited_count, visited_order

        for v, weight in graph[u].items():
            tentative_g = g[u] + weight
            if tentative_g < g[v]:
                g[v] = tentative_g
                came_from[v] = u
                f = tentative_g + h(v)
                heapq.heappush(open_set, (f, v))

    return None, visited_count, visited_order  # 경로 없음


def reconstruct_path(prev: dict, goal: int) -> list[int] | None:
    """Dijkstra prev 딕셔너리로 start → goal 경로 복원"""
    path: list[int] = []
    cur: int | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    if path and path[0] is not None:
        return path
    return None


def path_to_coords(path: list[int], nodes: list[dict]) -> list[tuple[float, float]]:
    """경로 노드 ID 리스트 → (위도, 경도) 좌표 리스트 (Folium polyline용)"""
    return [(nodes[i]["lat"], nodes[i]["lon"]) for i in path]


# ---------------------------------------------------------------------------
# Overpass API — 보행 접근성 인프라 조회
# 데이터 출처: OpenStreetMap_보행 인프라(엘리베이터·경사로·보도턱):Overpass API (overpass-api.de)
# ---------------------------------------------------------------------------

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# 조회 대상 태그 → (feat_type, 표시명, 색상, 이모지)
_OVERPASS_FEATURE_META: dict[str, tuple[str, str, str]] = {
    "elevator":     ("엘리베이터",    "#3949AB", "🛗"),
    "kerb_lowered": ("보도턱 낮춤",   "#00838F", "♿"),
    "ramp":         ("경사로",        "#E65100", "⬆"),
}


def fetch_overpass_accessibility(
    route_coords: list[tuple[float, float]],
    pad_m: int = 80,
    timeout: int = 15,
) -> list[dict]:
    """경로 주변 보행 접근성 인프라를 OSM Overpass API로 조회

    대상 태그:
        highway=elevator        — 엘리베이터
        kerb=lowered / flush    — 보도턱 낮춤 (차도→인도 경사 처리)
        ramp:wheelchair=yes     — 휠체어 경사로
        ramp=yes                — 일반 경사로

    [알고리즘] Overpass QL 바운딩 박스 조회:
        경로 좌표의 min/max 위경도로 bbox를 계산 후 Overpass 서버에 질의.
        응답 노드를 feat_type 별로 분류하여 반환.

    Returns:
        [{"lat", "lon", "feat_type", "label", "color", "emoji", "name"}, ...]
        실패(타임아웃·네트워크 오류) 시 빈 리스트 반환.
    """
    if not route_coords:
        return []

    # 바운딩 박스 (경로 전체 범위 + pad_m 여유)
    lats = [c[0] for c in route_coords]
    lons = [c[1] for c in route_coords]
    mid_lat = sum(lats) / len(lats)
    R = 6_371_000
    pad_lat = pad_m / R * (180 / math.pi)
    pad_lon = pad_m / (R * math.cos(math.radians(mid_lat))) * (180 / math.pi)
    bbox = f"{min(lats) - pad_lat},{min(lons) - pad_lon},{max(lats) + pad_lat},{max(lons) + pad_lon}"

    query = f"""
[out:json][timeout:{timeout}];
(
  node["highway"="elevator"]({bbox});
  node["kerb"="lowered"]({bbox});
  node["kerb"="flush"]({bbox});
  node["ramp:wheelchair"="yes"]({bbox});
  node["ramp"="yes"]({bbox});
);
out body;
"""
    try:
        resp = requests.post(
            _OVERPASS_URL,
            data={"data": query},
            timeout=timeout + 5,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception:  # noqa: BLE001
        return []

    results: list[dict] = []
    seen: set[int] = set()  # 중복 노드 제거

    for el in elements:
        if el.get("type") != "node" or el["id"] in seen:
            continue
        seen.add(el["id"])

        tags = el.get("tags", {})
        hw   = tags.get("highway", "")
        kerb = tags.get("kerb", "")
        ramp_wc = tags.get("ramp:wheelchair", "")
        ramp    = tags.get("ramp", "")

        if hw == "elevator":
            feat_type = "elevator"
        elif kerb in ("lowered", "flush"):
            feat_type = "kerb_lowered"
        elif ramp_wc == "yes" or ramp == "yes":
            feat_type = "ramp"
        else:
            continue

        label, color, emoji = _OVERPASS_FEATURE_META[feat_type]
        results.append({
            "lat":       el["lat"],
            "lon":       el["lon"],
            "feat_type": feat_type,
            "label":     label,
            "color":     color,
            "emoji":     emoji,
            "name":      tags.get("name", ""),
        })

    return results


# 데이터 출처: OpenStreetMap_보행자 도로망:OSMnx 라이브러리 (openstreetmap.org)
def get_walking_route(
    user_lat: float,
    user_lon: float,
    dest_lat: float,
    dest_lon: float,
    wheelchair: bool = False,
) -> tuple[list[tuple[float, float]], int]:
    """OSMnx 보행자 네트워크 기반 실제 도보 경로 반환

    OpenStreetMap의 보행자(인도) 네트워크를 다운로드하여
    출발지·목적지를 가장 가까운 보행 노드에 매핑한 뒤,
    NetworkX 최단경로로 실제 이동 가능한 경로 좌표를 계산한다.

    [자료구조] Graph (OSMnx/NetworkX MultiDiGraph): 보행자 도로망
    [알고리즘] NetworkX shortest_path (Dijkstra, 엣지 walk_weight 가중치)
        OSMnx 가 반경 내 보행 도로망을 O(N log N) 로 구성한 뒤
        NetworkX Dijkstra 로 최단 보행 경로를 탐색한다.

    Args:
        user_lat, user_lon: 출발지 위경도
        dest_lat, dest_lon: 목적지 위경도
        wheelchair: True 이면 highway=steps 엣지를 그래프에서 완전 제거하고 경사로·wheelchair 태그 경로 우선 선택

    Returns:
        (coords, elevator_count)
        coords: [(위도, 경도), ...] 경로 좌표 리스트
        elevator_count: OSM highway=elevator 노드 통과 횟수 (대기시간 추정용)
    """
    try:
        import networkx as nx  # noqa: PLC0415
        import osmnx as ox  # noqa: PLC0415
    except ImportError:
        raise RuntimeError("osmnx/networkx 패키지가 설치되지 않았습니다.")  # noqa: B904

    def _fetch() -> tuple[list[tuple[float, float]], int]:
        import os as _os  # noqa: PLC0415
        ox.settings.log_console = False
        ox.settings.timeout = 20  # OSM Overpass API 서버 응답 타임아웃
        # 디스크 캐시 활성화 — 동일 지역 재요청 시 즉시 응답
        ox.settings.use_cache = True
        ox.settings.cache_folder = _os.path.join(
            _os.path.dirname(__file__), "..", "data", "osmnx_cache"
        )

        # graph_from_bbox: 출발·도착 좌표의 최소 경계 박스 + 250m 여유
        # graph_from_point(원형) 대비 직선형 경로에서 다운로드 면적 최대 80% 절감
        _PAD_M   = 250
        _mid_lat = (user_lat + dest_lat) / 2
        _deg_lat = _PAD_M / 111_000
        _deg_lon = _PAD_M / (111_000 * math.cos(math.radians(_mid_lat)))
        _bbox = (
            min(user_lon, dest_lon) - _deg_lon,  # west  (left)
            min(user_lat, dest_lat) - _deg_lat,  # south (bottom)
            max(user_lon, dest_lon) + _deg_lon,  # east  (right)
            max(user_lat, dest_lat) + _deg_lat,  # north (top)
        )

        # [자료구조] Graph (OSMnx): network_type="walk" 보행자 전용 도로망
        G = ox.graph_from_bbox(
            _bbox,
            network_type="walk",
            simplify=True,
        )

        if wheelchair:
            # [알고리즘] 휠체어 모드: highway=steps 엣지 완전 제거
            # 패널티로 억제하는 대신 그래프에서 완전히 삭제하여 계단 경로를 원천 차단
            def _is_steps(hw) -> bool:
                return ("steps" in hw) if isinstance(hw, list) else (hw == "steps")
            _step_edges = [
                (u, v, k) for u, v, k, data in G.edges(data=True, keys=True)
                if _is_steps(data.get("highway", ""))
            ]
            G.remove_edges_from(_step_edges)

        # 도로 유형별 패널티 가중치 부여
        _PENALTY: dict[str, float] = {
            "path":      8.0,
            "track":     8.0,
            "steps":     5.0,
            "bridleway": 8.0,
            "footway":   1.2,
            "cycleway":  1.5,
        }
        for _, _, data in G.edges(data=True):
            hw = data.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0]
            base_w = data.get("length", 1) * _PENALTY.get(hw, 1.0)

            if wheelchair:
                # OSM wheelchair 태그: 접근 가능 경로 우선 선택, 불가 경로 패널티
                wc = data.get("wheelchair", "")
                if wc in ("yes", "designated"):
                    base_w *= 0.7
                elif wc == "no":
                    base_w *= 5.0
                # OSM 경사로 태그 반영 (ramp=yes / ramp:wheelchair=yes)
                if data.get("ramp") == "yes" or data.get("ramp:wheelchair") == "yes":
                    base_w *= 0.8

            data["walk_weight"] = base_w

        orig_node = ox.nearest_nodes(G, X=user_lon, Y=user_lat)
        dest_node = ox.nearest_nodes(G, X=dest_lon, Y=dest_lat)

        if orig_node == dest_node:
            return [(user_lat, user_lon), (dest_lat, dest_lon)], 0

        try:
            # [알고리즘] NetworkX Dijkstra: walk_weight(도로 유형 패널티 반영) 기준 최단 보행 경로
            route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="walk_weight")
        except nx.NetworkXNoPath:
            if wheelchair:
                raise RuntimeError(
                    "계단 없이 이동 가능한 경로를 찾을 수 없습니다. "
                    "목적지까지 엘리베이터·경사로가 연결되지 않거나 OSM 데이터가 부족할 수 있습니다."
                ) from None
            raise RuntimeError("두 지점 사이에 연결된 보행 경로가 없습니다.") from None

        # OSM highway=elevator 노드 감지 — 탑승 횟수 → 대기시간 추정에 사용
        elevator_count = sum(
            1 for n in route_nodes
            if G.nodes[n].get("highway") == "elevator"
        )

        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route_nodes]

        if coords and coords[0] != (user_lat, user_lon):
            coords = [(user_lat, user_lon)] + coords
        if coords and coords[-1] != (dest_lat, dest_lon):
            coords = coords + [(dest_lat, dest_lon)]

        return coords, elevator_count

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch)
            return future.result(timeout=_WALK_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError as e:
        raise RuntimeError("OSMnx 경로 계산 시간 초과 (60초)") from e
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(str(e)) from e


def get_walking_route_waypoints(
    waypoint_coords: list[tuple[float, float]],
    wheelchair: bool = False,
) -> tuple[list[tuple[float, float]], int]:
    """알고리즘이 결정한 경유지를 순서대로 통과하는 실제 보행 경로 반환

    OSMnx 그래프를 경유지 전체 bbox로 한 번만 구성하고,
    연속된 경유지 쌍을 순서대로 라우팅하여 이어붙인다.
    get_walking_route(출발→목적지 직행) 와 달리 알고리즘이 선택한
    중간 시설 노드를 실제 경로에 반영한다.

    [자료구조] Graph (OSMnx): 경유지 전체 bbox를 커버하는 보행자 도로망
    [알고리즘] NetworkX Dijkstra × (경유지 수 - 1) 구간:
        단일 그래프 내에서 경유지 쌍마다 최단 경로를 구하고 이어붙임.
        그래프 빌드 1회, 라우팅 N-1회 → 다중 OSMnx 호출 대비 빠름.

    Args:
        waypoint_coords: [(위도, 경도), ...] 출발지 포함 순서대로 (≥ 2)
        wheelchair: True 이면 계단 엣지 제거 + 경사로/wheelchair 태그 반영

    Returns:
        (coords, elevator_count)
    """
    if len(waypoint_coords) < 2:
        return list(waypoint_coords), 0

    def _fetch_wp() -> tuple[list[tuple[float, float]], int]:
        import os as _os  # noqa: PLC0415

        import networkx as nx  # noqa: PLC0415
        import osmnx as ox  # noqa: PLC0415

        ox.settings.log_console = False
        ox.settings.timeout = 20
        ox.settings.use_cache = True
        ox.settings.cache_folder = _os.path.join(
            _os.path.dirname(__file__), "..", "data", "osmnx_cache"
        )

        # 모든 경유지를 포함하는 bbox (패딩 250m)
        lats = [c[0] for c in waypoint_coords]
        lons = [c[1] for c in waypoint_coords]
        _PAD_M  = 250
        _mid_lat = sum(lats) / len(lats)
        _deg_lat = _PAD_M / 111_000
        _deg_lon = _PAD_M / (111_000 * math.cos(math.radians(_mid_lat)))
        _bbox = (
            min(lons) - _deg_lon,
            min(lats) - _deg_lat,
            max(lons) + _deg_lon,
            max(lats) + _deg_lat,
        )

        # [자료구조] Graph (OSMnx): 경유지 전체 범위 보행자 도로망 (1회 빌드)
        G = ox.graph_from_bbox(_bbox, network_type="walk", simplify=True)

        if wheelchair:
            def _is_steps(hw) -> bool:
                return ("steps" in hw) if isinstance(hw, list) else (hw == "steps")
            G.remove_edges_from([
                (u, v, k) for u, v, k, d in G.edges(data=True, keys=True)
                if _is_steps(d.get("highway", ""))
            ])

        _PENALTY: dict[str, float] = {
            "path": 8.0, "track": 8.0, "steps": 5.0,
            "bridleway": 8.0, "footway": 1.2, "cycleway": 1.5,
        }
        for _, _, data in G.edges(data=True):
            hw = data.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0]
            base_w = data.get("length", 1) * _PENALTY.get(hw, 1.0)
            if wheelchair:
                wc = data.get("wheelchair", "")
                if wc in ("yes", "designated"):
                    base_w *= 0.7
                elif wc == "no":
                    base_w *= 5.0
                if data.get("ramp") == "yes" or data.get("ramp:wheelchair") == "yes":
                    base_w *= 0.8
            data["walk_weight"] = base_w

        # 각 경유지를 가장 가까운 OSM 노드에 매핑
        osm_nodes = [
            ox.nearest_nodes(G, X=lon, Y=lat)
            for lat, lon in waypoint_coords
        ]

        # [알고리즘] 구간별 Dijkstra → 이어붙이기
        full_coords: list[tuple[float, float]] = []
        elevator_count = 0

        for i in range(len(osm_nodes) - 1):
            src, dst = osm_nodes[i], osm_nodes[i + 1]
            if src == dst:
                if not full_coords:
                    full_coords.append((G.nodes[src]["y"], G.nodes[src]["x"]))
                continue
            try:
                seg_nodes = nx.shortest_path(G, src, dst, weight="walk_weight")
            except nx.NetworkXNoPath:
                if wheelchair:
                    raise RuntimeError(
                        "계단 없이 이동 가능한 경로를 찾을 수 없습니다. "
                        "목적지까지 엘리베이터·경사로가 연결되지 않거나 OSM 데이터가 부족할 수 있습니다."
                    ) from None
                raise RuntimeError("두 지점 사이에 연결된 보행 경로가 없습니다.") from None

            elevator_count += sum(
                1 for n in seg_nodes if G.nodes[n].get("highway") == "elevator"
            )
            seg_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in seg_nodes]
            # 첫 구간이면 전체 추가, 이후엔 중복 첫 좌표 제거
            full_coords.extend(seg_coords if i == 0 else seg_coords[1:])

        # 시작·끝 정확한 좌표 보정
        if full_coords:
            if full_coords[0] != waypoint_coords[0]:
                full_coords = [waypoint_coords[0]] + full_coords
            if full_coords[-1] != waypoint_coords[-1]:
                full_coords = full_coords + [waypoint_coords[-1]]

        return full_coords, elevator_count

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch_wp)
            return future.result(timeout=_WALK_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError as e:
        raise RuntimeError("OSMnx 경로 계산 시간 초과 (60초)") from e
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(str(e)) from e
