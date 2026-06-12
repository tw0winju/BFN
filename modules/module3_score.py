# module3_score.py — 팀원 3
# 담당: 편의시설 점수화 추천
# [자료구조] Heap (heapq), Dict, Set
# [알고리즘] Greedy Set Cover 추천, Heap Sort (Top-N)

import heapq

from config import TOP_N


def calc_score(facility: dict, distance_m: float, weights: dict) -> float:
    """단일 시설 추천 점수 계산

    점수 산식:
        score = 거리점수 × w_dist
                + has_toilet × w_toilet
                + has_elevator × w_elev
                + has_ramp × w_ramp
                + has_parking × w_park
        거리점수 = 1 / (1 + 거리_km)  → 가까울수록 높음

    [알고리즘] Greedy 가중 점수화:
        각 항목의 기여도를 가중합으로 즉시(greedy) 계산한다.
        전역 최적을 보장하지 않지만 계산이 빠르고 직관적이다.
        시간복잡도 O(1)
    """
    dist_km = distance_m / 1000.0
    dist_score = 1.0 / (1.0 + dist_km)

    score = (
        dist_score             * weights.get("dist",   0.5)
        + int(facility["has_toilet"])   * weights.get("toilet", 0.2)
        + int(facility["has_elevator"]) * weights.get("elev",   0.1)
        + int(facility["has_ramp"])     * weights.get("ramp",   0.1)
        + int(facility["has_parking"])  * weights.get("park",   0.1)
    )
    return score


def rank_facilities(
    facilities: list[dict],
    distances: dict,          # {node_id: distance_m}  (Dijkstra 결과)
    weights: dict,
) -> list[tuple[float, dict]]:
    """전체 시설 점수 계산 후 (score, facility) 리스트 반환

    node_id는 build_graph 기준 1-based (node 0 = 사용자).

    [자료구조] Dict: node_id → score 매핑 (중간 계산 결과 저장)
    """
    # [자료구조] Dict: 각 노드의 점수를 저장
    score_map: dict[int, float] = {}

    scored: list[tuple[float, dict]] = []
    for idx, facility in enumerate(facilities):
        node_id = idx + 1  # node 0 = 사용자
        dist_m = distances.get(node_id, float("inf"))
        if dist_m == float("inf"):
            # 도달 불가능한 시설은 목록에서 제외 (0점으로 추천하면 오해 유발)
            score_map[node_id] = 0.0
            continue
        s = calc_score(facility, dist_m, weights)
        score_map[node_id] = s
        # node_id를 facility에 임시 주입하여 app.py에서 goal_node 추적에 사용
        tagged = dict(facility)
        tagged["_node_id"] = node_id
        scored.append((s, tagged))

    return scored


def get_top_n(scored: list[tuple[float, dict]], n: int = TOP_N) -> list[dict]:
    """점수 상위 N개 추출

    [자료구조] Heap (heapq): 최대 힙 시뮬레이션으로 상위 N개를 효율적으로 선택
    [알고리즘] Heap Sort (Top-N 선택):
        전체 정렬(O(M log M)) 대신 heapq.nlargest를 사용해
        O(M log N) 에 상위 N개를 선택한다.
    """
    # [자료구조] Heap (heapq.nlargest): 최대 N개 힙 기반 선택
    top = heapq.nlargest(n, scored, key=lambda x: x[0])
    # 결과에 점수 필드 추가 후 반환
    result: list[dict] = []
    for score, facility in top:
        fac = dict(facility)
        fac["score"] = round(score, 4)
        result.append(fac)
    return result


# ---------------------------------------------------------------------------
# Greedy Set Cover 추천
# ---------------------------------------------------------------------------

_AMENITY_KEYS = ["has_toilet", "has_elevator", "has_ramp", "has_parking"]


def greedy_coverage_recommend(
    facilities: list[dict],
    distances: dict,
    weights: dict,
    n: int = TOP_N,
) -> list[dict]:
    """Greedy Set Cover 기반 편의시설 추천

    각 선택 단계에서 '아직 제공되지 않은 편의시설 유형'을 가장 많이
    새롭게 커버하는 시설을 탐욕적으로 선택한다.

    [자료구조] Set: covered — 이미 선택된 편의시설 유형 집합 (O(1) 갱신·조회)
    [알고리즘] Greedy Set Cover:
        매 단계에서 marginal_value = 새 커버 유형 수 × w_access
                                    + 거리점수 × w_dist
        를 최대화하는 시설을 선택한다.
        이전 선택 결과(covered)가 다음 선택의 평가 기준을 바꾸므로
        단순 가중합과 달리 진정한 순차 탐욕 결정이다.
        Set Cover (1 - 1/e) 근사비를 달성한다.
        시간복잡도 O(n × M) — n: 선택 횟수, M: 후보 시설 수
    """
    w_dist   = weights.get("dist", 0.5)
    w_access = 1.0 - w_dist

    # 도달 가능한 시설만 후보 (node_id 1-based)
    candidates: list[tuple[int, dict]] = [
        (idx + 1, fac)
        for idx, fac in enumerate(facilities)
        if distances.get(idx + 1, float("inf")) < float("inf")
    ]

    # [자료구조] Set: 이미 커버된 편의시설 유형 집합
    covered: set[str] = set()
    selected: list[dict] = []

    for _ in range(n):
        if not candidates:
            break

        # 현재 covered 스냅샷을 기본값으로 캡처 (루프 변수 클로저 문제 방지)
        snap = frozenset(covered)

        def _marginal(item: tuple[int, dict], _snap: frozenset = snap) -> float:
            node_id, fac = item
            dist_m    = distances[node_id]
            dist_score = 1.0 / (1.0 + dist_m / 1000.0)
            new_cov   = len({k for k in _AMENITY_KEYS if fac.get(k)} - _snap)
            return new_cov * w_access + dist_score * w_dist

        best = max(candidates, key=_marginal)
        node_id, fac = best

        # [자료구조] Set: 선택된 시설의 편의시설 유형을 covered에 추가
        covered |= {k for k in _AMENITY_KEYS if fac.get(k)}

        result = dict(fac)
        result["_node_id"] = node_id
        result["score"]    = round(_marginal(best), 4)
        selected.append(result)
        candidates.remove(best)

    return selected
