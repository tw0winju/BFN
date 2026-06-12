"""
파이프라인 통합 테스트 — 각 단계에서 계산된 값이 실제로 다음 단계로 전달되는지 검증

실제 함수 시그니처:
  dijkstra(graph, start)            → (dist, prev, visited_count, visited_order)
  astar(graph, start, goal, nodes)  → (path|None, visited_count, visited_order)
  bfs_accessible_path(graph, nodes, start, goal) → (path|None, visited_count, visited_order)
  rank_facilities(facilities, distances, weights) → list[(score, facility)]  (도달불가 제외)
  get_top_n(scored, n)              → list[dict]  (score 필드 추가된 facility dict)
"""

import sys

sys.path.insert(0, "/Users/lsj/Study/BFN")

PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  ✅ {name}")
        PASS += 1
    else:
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))
        FAIL += 1

# ──────────────────────────────────────────────
# 공통 픽스처 (정제된 시설 데이터)
# ──────────────────────────────────────────────
facilities = [
    {"name": "시설A", "lat": 37.450, "lon": 127.130,
     "has_toilet": True,  "has_elevator": True,  "has_ramp": True,  "has_parking": False},
    {"name": "시설B", "lat": 37.452, "lon": 127.132,
     "has_toilet": False, "has_elevator": True,  "has_ramp": False, "has_parking": True},
    {"name": "시설C", "lat": 37.454, "lon": 127.134,
     "has_toilet": True,  "has_elevator": False, "has_ramp": True,  "has_parking": False},
    {"name": "시설D", "lat": 37.460, "lon": 127.140,
     "has_toilet": False, "has_elevator": False, "has_ramp": False, "has_parking": False},
]
user_lat, user_lon = 37.449, 127.129
# build_graph 내부에서 노드 0 = 사용자로 prepend함
# astar/bfs_accessible_path 의 nodes 파라미터는 [user_node] + facilities
all_nodes = [{"lat": user_lat, "lon": user_lon, "name": "사용자",
              "has_elevator": False, "has_ramp": False}] + facilities

# ──────────────────────────────────────────────
print("\n[Stage 1] Module1 — normalize / filter / Trie")
# ──────────────────────────────────────────────
from modules.module1_data import build_region_trie, filter_facilities, normalize  # noqa: E402

# 실제 API 응답 필드명으로 raw 구성
raw = [
    {
        "faclNm":  "서울시설",
        "lcMnad":  "서울특별시 강남구 테헤란로 1",
        "faclLat": "37.500", "faclLng": "127.036",
        "evalInfo": "장애인사용가능화장실, 승강기",
        "salStaDivCd": "Y", "wfcltId": "T001", "faclTyCd": "종합병원",
    },
    {
        "faclNm":  "성남시설",
        "lcMnad":  "경기도 성남시 수정구 태평로 10",
        "faclLat": "37.452", "faclLng": "127.137",
        "evalInfo": "주출입구 접근로",
        "salStaDivCd": "Y", "wfcltId": "T002", "faclTyCd": "도서관",
    },
    {
        "faclNm":  "폐업시설",
        "lcMnad":  "서울특별시 종로구 종로1",
        "faclLat": "37.570", "faclLng": "126.980",
        "evalInfo": "",
        "salStaDivCd": "N",   # 폐업 → normalize에서 제외
        "wfcltId": "T003", "faclTyCd": "도서관",
    },
]
normalized = normalize(raw)
check("normalize 반환 타입 list", isinstance(normalized, list))
check("normalize 폐업 시설 제외 → 2개", len(normalized) == 2, f"반환 수={len(normalized)}")
check("normalize 필드 has_toilet 존재", "has_toilet" in normalized[0] if normalized else False)
check("normalize has_elevator True (evalInfo에 '승강기')",
      normalized[0]["has_elevator"] is True)
check("normalize has_ramp True (evalInfo에 '주출입구 접근로')",
      normalized[1]["has_ramp"] is True)
check("normalize has_toilet False (evalInfo에 없음)",
      normalized[1]["has_toilet"] is False)

filtered = filter_facilities(normalized, {"need_toilet": True})
check("filter need_toilet=True → 화장실 있는 시설만",
      all(f["has_toilet"] for f in filtered) and len(filtered) > 0,
      f"결과={[f['name'] for f in filtered]}")

# Trie 검증 — 주소 문자열 리스트로 빌드
addresses = ["서울특별시 강남구 테헤란로 1", "경기도 성남시 수정구 태평로 10"]
trie = build_region_trie(addresses)
results_seoul = trie.prefix_search("서울")
check("Trie '서울' 접두사 검색 결과 존재",
      len(results_seoul) > 0, f"결과={results_seoul}")
check("Trie '서울' 결과에 '서울특별시' 포함",
      any("서울" in r for r in results_seoul), f"결과={results_seoul}")

results_gy = trie.prefix_search("경기")
check("Trie '경기' 접두사 검색 결과 존재",
      len(results_gy) > 0, f"결과={results_gy}")

# ──────────────────────────────────────────────
print("\n[Stage 2] Module2 — build_graph / Dijkstra / A*")
# ──────────────────────────────────────────────
from modules.module2_path import astar, build_graph, dijkstra, haversine  # noqa: E402

# Haversine 검증
d_AB = haversine(facilities[0]["lat"], facilities[0]["lon"],
                 facilities[1]["lat"], facilities[1]["lon"])
check("Haversine A→B 양수", d_AB > 0, f"{d_AB:.1f}m")
check("Haversine A→B 합리적 거리 (50~500m)", 50 < d_AB < 500, f"{d_AB:.1f}m")

# 그래프 구성 (facilities만 전달, user는 내부에서 node 0으로 추가)
G = build_graph(facilities, user_lat, user_lon, radius_m=500)
check("build_graph 노드 수 == 1+시설수", len(G) == 1 + len(facilities),
      f"{len(G)} vs {1+len(facilities)}")
check("사용자 노드(0)가 그래프에 존재", 0 in G)
check("그래프에 엣지 존재", sum(len(v) for v in G.values()) > 0)

# Dijkstra — 튜플 언패킹
dist, prev, v_count, v_order = dijkstra(G, start=0)
check("dijkstra dist 반환 타입 dict", isinstance(dist, dict))
check("dijkstra prev 반환 타입 dict", isinstance(prev, dict))
check("dijkstra visited_count 정수", isinstance(v_count, int) and v_count > 0)
check("dijkstra 출발지 거리 == 0", dist.get(0) == 0, f"dist[0]={dist.get(0)}")
check("dijkstra 시설A 거리 > 0", dist.get(1, 0) > 0, f"dist[1]={dist.get(1):.1f}m")
check("dijkstra 거리 단조증가(A≤B≤C≤D)",
      dist.get(1,0) <= dist.get(2,0) <= dist.get(3,0) <= dist.get(4,0),
      f"A={dist.get(1):.1f} B={dist.get(2):.1f} C={dist.get(3):.1f} D={dist.get(4):.1f}")

hav_to_A = haversine(user_lat, user_lon, facilities[0]["lat"], facilities[0]["lon"])
check("Dijkstra 거리 ≥ Haversine 직선거리(A)",
      dist.get(1, 0) >= hav_to_A * 0.99,
      f"dijkstra={dist.get(1):.1f} haversine={hav_to_A:.1f}")

# A* — 튜플 언패킹
path, a_vcount, a_vorder = astar(G, start=0, goal=3, nodes=all_nodes)
check("astar 반환 path가 list or None", path is None or isinstance(path, list))
check("astar visited_count 정수", isinstance(a_vcount, int) and a_vcount > 0)
if path is not None:
    check("astar 시작=0", path[0] == 0, f"path={path}")
    check("astar 끝=3",   path[-1] == 3, f"path={path}")
    check("astar 경로 노드가 그래프에 존재", all(n in G for n in path), f"path={path}")

# ──────────────────────────────────────────────
print("\n[Stage 3] Module3 — score / rank / top-N")
# ──────────────────────────────────────────────
from config import USER_TYPES  # noqa: E402
from modules.module3_score import calc_score, get_top_n, rank_facilities  # noqa: E402

weights    = USER_TYPES["일반"]["weights"]
wc_weights = USER_TYPES["휠체어 사용자"]["weights"]

score_A = calc_score(facilities[0], 100, weights)
score_D = calc_score(facilities[3], 100, weights)
check("calc_score 반환 float", isinstance(score_A, float))
check("시설A(화장실+엘베+경사로) > 시설D(없음) 동일거리",
      score_A > score_D, f"A={score_A:.4f} D={score_D:.4f}")

elev_ramp_w_normal = weights["elev"]    + weights["ramp"]
elev_ramp_w_wc     = wc_weights["elev"] + wc_weights["ramp"]
check("휠체어 가중치: 엘베+경사로 합이 일반보다 높음",
      elev_ramp_w_wc > elev_ramp_w_normal,
      f"wc={elev_ramp_w_wc:.2f} normal={elev_ramp_w_normal:.2f}")

# rank_facilities — Dijkstra dist 연동 (1-based node id)
dist_map = {i+1: dist.get(i+1, float("inf")) for i in range(len(facilities))}
ranked = rank_facilities(facilities, dist_map, weights)
check("rank_facilities 반환 list", isinstance(ranked, list))
check("rank_facilities 각 항목이 (score, facility) 튜플",
      all(isinstance(r, tuple) and len(r) == 2 for r in ranked) if ranked else True,
      f"sample={ranked[0] if ranked else None}")
check("rank_facilities score > 0 (도달 가능 시설)",
      all(s > 0 for s, _ in ranked))
check("rank_facilities facility에 'name' 필드 존재",
      all("name" in fac for _, fac in ranked))

# get_top_n — list[dict] 반환 (score 필드 포함)
top2 = get_top_n(ranked, n=2)
check("get_top_n 반환 list", isinstance(top2, list))
check("get_top_n 반환 수 == 2", len(top2) == 2)
check("top2 각 항목이 dict (score 필드 포함)",
      all(isinstance(f, dict) and "score" in f for f in top2))
check("top2[0].score ≥ top2[1].score",
      top2[0]["score"] >= top2[1]["score"],
      f"top0={top2[0]['score']} top1={top2[1]['score']}")
check("top2[0] 편의시설이 시설D보다 풍부 (D는 모두 False)",
      any(top2[0].get(k) for k in ["has_toilet","has_elevator","has_ramp","has_parking"]))

# ──────────────────────────────────────────────
print("\n[Stage 4] Module4 — is_accessible / BFS")
# ──────────────────────────────────────────────
from modules.module4_access import bfs_accessible_path, is_accessible  # noqa: E402

check("is_accessible A(엘베+경사로)=True",  is_accessible(facilities[0]) is True)
check("is_accessible B(엘베만)=True",       is_accessible(facilities[1]) is True)
check("is_accessible C(경사로만)=True",     is_accessible(facilities[2]) is True)
check("is_accessible D(없음)=False",        is_accessible(facilities[3]) is False)

# BFS — (path|None, visited_count, visited_order) 언패킹
bfs_path, b_vcount, b_vorder = bfs_accessible_path(G, all_nodes, start=0, goal=3)
check("bfs_accessible_path path가 list or None",
      bfs_path is None or isinstance(bfs_path, list))
check("bfs visited_count 정수", isinstance(b_vcount, int) and b_vcount > 0)
if bfs_path is not None:
    intermediate = bfs_path[1:-1]
    check("BFS 중간 노드 모두 접근가능",
          all(is_accessible(all_nodes[n]) for n in intermediate),
          f"intermediate={intermediate}")
    check("BFS 시작=0 끝=3",
          bfs_path[0] == 0 and bfs_path[-1] == 3, f"path={bfs_path}")

bfs_to_D, *_ = bfs_accessible_path(G, all_nodes, start=0, goal=4)
check("BFS goal=D 반환 list or None",
      bfs_to_D is None or isinstance(bfs_to_D, list))

# ──────────────────────────────────────────────
print("\n[Stage 5] 파이프라인 연결 검증")
# ──────────────────────────────────────────────

# Dijkstra 거리 → rank_facilities → get_top_n 체인 검증
dist_full, *_ = dijkstra(G, start=0)
dist_map_full  = {i+1: dist_full.get(i+1, float("inf")) for i in range(len(facilities))}
ranked_full = rank_facilities(facilities, dist_map_full, weights)
top3 = get_top_n(ranked_full, n=3)

# top3 점수가 전체 중 상위 3개인지
all_scores_sorted = sorted([s for s, _ in ranked_full], reverse=True)
top3_scores = sorted([f["score"] for f in top3], reverse=True)
check("top3 점수 = 전체 점수 상위 3개 (heapq.nlargest 검증)",
      top3_scores == [round(s, 4) for s in all_scores_sorted[:3]],
      f"top3={top3_scores} vs full_top3={[round(s,4) for s in all_scores_sorted[:3]]}")

# A* 경로 길이가 Dijkstra 거리와 일치하는지
path3, *_ = astar(G, start=0, goal=3, nodes=all_nodes)
if path3:
    path_len = sum(G[path3[i]][path3[i+1]] for i in range(len(path3)-1))
    dijk_3   = dist_full.get(3, float("inf"))
    check("A* 경로 길이 ≈ Dijkstra 거리 (5% 이내)",
          abs(path_len - dijk_3) / max(dijk_3, 1) < 0.05,
          f"A*={path_len:.1f}m dijkstra={dijk_3:.1f}m diff={abs(path_len-dijk_3):.1f}m")

# BFS와 A*가 동일 목적지에 도달하는지
goal_node = 3
bfs_res, *_ = bfs_accessible_path(G, all_nodes, start=0, goal=goal_node)
ast_res, *_ = astar(G, start=0, goal=goal_node, nodes=all_nodes)
check("BFS와 A* 모두 동일 목적지 도달",
      (bfs_res is None or bfs_res[-1] == goal_node) and
      (ast_res[-1] == goal_node if ast_res else False),
      f"bfs={bfs_res[-1] if bfs_res else None} astar={ast_res[-1] if ast_res else None}")

# Greedy Set Cover 추천이 rank+top_n과 일관된지
from modules.module3_score import greedy_coverage_recommend  # noqa: E402

greedy_top = greedy_coverage_recommend(facilities, dist_map_full, weights, n=3)
check("greedy_coverage_recommend 반환 list[dict]",
      isinstance(greedy_top, list) and all(isinstance(f, dict) for f in greedy_top))
check("greedy top3이 3개 이하", len(greedy_top) <= 3)
check("greedy 각 시설에 score 필드", all("score" in f for f in greedy_top))

# ──────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"결과: {PASS}개 통과 / {PASS+FAIL}개 총 검사 ({100*PASS//(PASS+FAIL)}%)")
if FAIL:
    print(f"실패: {FAIL}개")
print('='*50)
