# module1_data.py — 팀원 1
# 담당: 데이터 수집 · 좌표 변환 · 필터링 · 지역명 자동완성
# [자료구조] List, Dict, Trie
# [알고리즘] 선형탐색, 조건 필터링, 접두사 탐색 (Prefix Search / DFS)

import concurrent.futures
import json
import os
import time
import xml.etree.ElementTree as ET

import requests

_EVAL_WORKERS   = 10  # evalInfo 병렬 요청 스레드 수 (쿼터 보호)
_DETAIL_LIMIT   = 50  # 상세 API 호출 최대 시설 수 (거리 가까운 순 우선)

# 데이터 출처: 한국사회보장정보원_장애인편의시설 현황 (공공데이터포털)
FACILITY_LIST_URL = (
    "https://apis.data.go.kr/B554287/DisabledPersonConvenientFacility/getDisConvFaclList"
)
FACILITY_DETAIL_URL = (
    "https://apis.data.go.kr/B554287/DisabledPersonConvenientFacility"
    "/getFacInfoOpenApiJpEvalInfoList"
)
# 데이터 출처: 카카오 로컬 API (카카오 개발자센터) — 주소→좌표 변환, 키워드 장소 검색
KAKAO_GEOCODE_URL  = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_KEYWORD_URL  = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_REGION_URL   = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cache.json")
CACHE_TTL_SEC = 60 * 60 * 24  # 24시간

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 거리(m) — module2 의존 없이 독립 구현"""
    import math
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# evalInfo 문자열에서 편의시설 보유 여부를 판별하는 키워드
_TOILET_KW   = "장애인사용가능화장실"
_ELEVATOR_KW = "승강기"
_RAMP_KW1    = "주출입구 접근로"
_RAMP_KW2    = "주출입구 높이차이 제거"
_PARKING_KW  = "장애인전용주차구역"


# ---------------------------------------------------------------------------
# 캐시 헬퍼
# ---------------------------------------------------------------------------

_mem_cache: dict = {}
_mem_cache_mtime: float = -1.0

def _load_cache() -> dict:
    global _mem_cache, _mem_cache_mtime
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        mtime = os.path.getmtime(CACHE_PATH)
        if mtime == _mem_cache_mtime:   # 파일 미변경 → 메모리 캐시 반환
            return _mem_cache
        with open(CACHE_PATH, encoding="utf-8") as f:
            _mem_cache = json.load(f)
        _mem_cache_mtime = mtime
        return _mem_cache
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# API 1: 시설 목록 조회
# ---------------------------------------------------------------------------

def fetch_facilities(
    api_key: str,
    sido: str,
    gungu: str = "",
    num: int = 50,
    facl_ty_cd: str = "",
    progress_cb=None,
    user_lat: float = 0.0,
    user_lon: float = 0.0,
    detail_limit: int = _DETAIL_LIMIT,
    priority_types: list | None = None,
) -> list[dict]:
    """장애인편의시설 목록 + 편의시설 상세(evalInfo) 조회

    priority_types 가 주어지면 해당 타입코드로 먼저 API 호출(병렬)하여 결과 앞에 배치,
    나머지 num 을 일반 조회로 보충한다. 카테고리 선택 시 해당 시설이 500개 한도 밖으로
    밀리는 문제를 방지한다.

    목록 조회 후 시설별로 getFacInfoOpenApiJpEvalInfoList를 추가 호출하여
    has_toilet, has_elevator, has_ramp, has_parking 정보를 병합한다.
    상세 결과는 wfcltId 단위로 캐시하여 중복 호출을 방지한다.

    user_lat/user_lon 제공 시 가까운 detail_limit개만 상세 API 호출 (쿼터 절약).
    나머지 시설은 evalInfo="" 로 처리되어 점수 산정 시 편의시설 점수 0점.

    [자료구조] List[Dict]: 시설 목록 저장
    [알고리즘] 선형탐색: 페이지 결과를 순회하며 누적
    """
    cache = _load_cache()

    # ── Step 1: 목록 조회 ────────────────────────────────────────────────
    # priority_types 가 있으면 해당 타입코드별로 병렬 호출 후 앞에 배치,
    # 나머지 슬롯을 일반(facl_ty_cd) 조회로 채운다.
    if priority_types:
        seen_ids: set[str] = set()
        priority_raw: list[dict] = []

        def _fetch_type(ty: str) -> list[dict]:
            _key = f"list_{sido}_{gungu}_{num}_{ty}"
            _e = cache.get(_key)
            if _e and time.time() - _e.get("ts", 0) < CACHE_TTL_SEC:
                return _e["data"]
            result = _fetch_list(api_key, sido, gungu, num, ty)
            cache[_key] = {"ts": time.time(), "data": result}
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(priority_types)) as ex:
            for part in ex.map(_fetch_type, priority_types):
                for item in part:
                    wid = item.get("wfcltId", "")
                    if wid not in seen_ids:
                        seen_ids.add(wid)
                        priority_raw.append(item)

        # 일반 결과로 나머지 슬롯 보충
        if len(priority_raw) < num:
            general_key = f"list_{sido}_{gungu}_{num}_{facl_ty_cd}"
            ge = cache.get(general_key)
            if ge and time.time() - ge.get("ts", 0) < CACHE_TTL_SEC:
                general_raw = ge["data"]
            else:
                general_raw = _fetch_list(api_key, sido, gungu, num, facl_ty_cd)
                cache[general_key] = {"ts": time.time(), "data": general_raw}
            for item in general_raw:
                wid = item.get("wfcltId", "")
                if wid not in seen_ids:
                    seen_ids.add(wid)
                    priority_raw.append(item)
                    if len(priority_raw) >= num:
                        break

        _save_cache(cache)
        raw_list = priority_raw[:num]
    else:
        list_cache_key = f"list_{sido}_{gungu}_{num}_{facl_ty_cd}"
        entry = cache.get(list_cache_key)
        if entry and time.time() - entry.get("ts", 0) < CACHE_TTL_SEC:
            raw_list = entry["data"]
        else:
            raw_list = _fetch_list(api_key, sido, gungu, num, facl_ty_cd)
            cache[list_cache_key] = {"ts": time.time(), "data": raw_list}
            _save_cache(cache)

    if not raw_list:
        return []

    # ── Step 2: 시설별 편의시설 상세 조회 (evalInfo) ─────────────────────
    # 1차 패스: 캐시 히트 항목은 즉시 처리, 미스 항목은 병렬 대상으로 분류
    total = len(raw_list)
    to_fetch: list[tuple[int, dict, str]] = []  # (index, item, wfclt_id)
    done_count = 0

    for i, item in enumerate(raw_list):
        wfclt_id = item.get("wfcltId", "")
        if not wfclt_id:
            item["evalInfo"] = ""
            done_count += 1
            if progress_cb:
                progress_cb(done_count, total)
            continue

        detail_key = f"detail_{wfclt_id}"
        detail_entry = cache.get(detail_key)
        if detail_entry and time.time() - detail_entry.get("ts", 0) < CACHE_TTL_SEC:
            # 캐시 히트 — API 호출 없이 즉시 적용
            item["evalInfo"] = detail_entry["data"]
            done_count += 1
            if progress_cb:
                progress_cb(done_count, total)
        else:
            to_fetch.append((i, item, wfclt_id))

    # ── 거리 기반 우선순위: 가까운 detail_limit개만 API 호출 ─────────────
    # user_lat/lon이 있으면 캐시 미스 목록을 거리 순으로 정렬하고
    # detail_limit 초과분은 evalInfo="" 로 즉시 처리 (API 호출 생략)
    if user_lat and user_lon and len(to_fetch) > detail_limit:
        def _dist_key(args: tuple) -> float:
            _, item, _ = args
            try:
                return _haversine_m(
                    user_lat, user_lon,
                    float(item.get("faclLat") or 0),
                    float(item.get("faclLng") or 0),
                )
            except (ValueError, TypeError):
                return float("inf")

        to_fetch.sort(key=_dist_key)

        # detail_limit 초과 시설: evalInfo="" 로 건너뜀
        for _, item, _ in to_fetch[detail_limit:]:
            item["evalInfo"] = ""
            done_count += 1
            if progress_cb:
                progress_cb(done_count, total)
        to_fetch = to_fetch[:detail_limit]

    # 2차 패스: 캐시 미스 항목을 ThreadPoolExecutor로 병렬 호출
    # [알고리즘] 병렬 선형탐색: 순차 O(N)에서 O(N/W)로 단축 (W=스레드 수)
    if to_fetch:
        new_cache: dict = {}

        def _fetch_one(args: tuple) -> tuple:
            idx, item, wfclt_id = args
            return idx, wfclt_id, _fetch_eval_info(api_key, wfclt_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=_EVAL_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, args): args for args in to_fetch}
            for future in concurrent.futures.as_completed(futures):
                idx, wfclt_id, eval_info = future.result()
                raw_list[idx]["evalInfo"] = eval_info
                # 빈 결과(API 실패·쿼터 초과)는 캐시 저장 안 함 → 다음 검색 시 재시도
                if eval_info:
                    new_cache[f"detail_{wfclt_id}"] = {"ts": time.time(), "data": eval_info}
                done_count += 1
                if progress_cb:
                    progress_cb(done_count, total)

        cache.update(new_cache)

    _save_cache(cache)
    return raw_list


def _fetch_list(api_key: str, sido: str, gungu: str, num: int, facl_ty_cd: str = "") -> list[dict]:
    """목록 API 페이지네이션 호출 → raw dict 리스트"""
    results: list[dict] = []
    page = 1
    page_size = min(num, 1000)

    while len(results) < num:
        params = {
            "serviceKey": api_key,
            "pageNo": page,
            "numOfRows": page_size,
        }
        if sido:
            params["siDoNm"] = sido
        if gungu:
            params["cggNm"] = gungu
        if facl_ty_cd:
            params["faclTyCd"] = facl_ty_cd

        try:
            resp = requests.get(FACILITY_LIST_URL, params=params, timeout=10)
            resp.raise_for_status()
            items = _parse_xml_list(resp.text)
            if not items:
                break
            for item in items:
                results.append(item)
                if len(results) >= num:
                    break
            if len(items) < page_size:
                break
            page += 1
        except requests.RequestException as e:
            print(f"[module1] 목록 API 오류 (page={page}): {e}")
            break
        except Exception as e:
            print(f"[module1] 목록 파싱 오류 (page={page}): {e}")
            break

    return results


def _parse_xml_list(xml_text: str) -> list[dict]:
    """목록 API XML 응답 파싱"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    items: list[dict] = []
    for node in root.findall(".//servList"):
        item = {child.tag: (child.text or "") for child in node}
        items.append(item)
    return items


def _fetch_eval_info(api_key: str, wfclt_id: str) -> str:
    """상세 API 호출 → evalInfo 문자열 반환"""
    params = {"serviceKey": api_key, "wfcltId": wfclt_id}
    try:
        resp = requests.get(FACILITY_DETAIL_URL, params=params, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        node = root.find(".//evalInfo")
        return node.text or "" if node is not None else ""
    except requests.RequestException as e:
        print(f"[module1] evalInfo API 오류 (wfcltId={wfclt_id}): {e}")
        return ""
    except ET.ParseError as e:
        print(f"[module1] evalInfo XML 파싱 오류 (wfcltId={wfclt_id}): {e}")
        return ""


# ---------------------------------------------------------------------------
# 카카오 Geocoding
# ---------------------------------------------------------------------------

def search_places(keyword: str, kakao_key: str, max_results: int = 10) -> list[dict]:
    """카카오 키워드·주소 통합 검색 → 선택 가능한 장소 목록 반환

    키워드 검색을 먼저 시도하고, 결과가 없으면 주소 검색으로 fallback한다.
    "강남대로15" 같은 도로명 주소도 검색 가능.

    반환 항목: {name, address, lat, lon, category}
    """
    if not keyword.strip() or not kakao_key:
        return []
    headers = {"Authorization": f"KakaoAK {kakao_key}"}

    def _parse_keyword(docs: list) -> list[dict]:
        return [
            {
                "name":     doc["place_name"],
                "address":  doc.get("road_address_name") or doc.get("address_name", ""),
                "lat":      float(doc["y"]),
                "lon":      float(doc["x"]),
                "category": doc.get("category_name", "").split(" > ")[-1],
            }
            for doc in docs
        ]

    def _parse_address(docs: list) -> list[dict]:
        results = []
        for doc in docs:
            road = doc.get("road_address") or {}
            addr = doc.get("address") or {}
            lat = float(doc.get("y", 0))
            lon = float(doc.get("x", 0))
            if not lat or not lon:
                continue
            name = road.get("address_name") or addr.get("address_name", "")
            results.append({
                "name":     name,
                "address":  name,
                "lat":      lat,
                "lon":      lon,
                "category": "주소",
            })
        return results

    try:
        # 1차: 키워드 검색
        resp = requests.get(
            KAKAO_KEYWORD_URL,
            headers=headers,
            params={"query": keyword, "size": max_results},
            timeout=5,
        )
        resp.raise_for_status()
        results = _parse_keyword(resp.json().get("documents", []))

        # 2차: 결과 없으면 주소 검색으로 fallback
        if not results:
            resp2 = requests.get(
                KAKAO_GEOCODE_URL,
                headers=headers,
                params={"query": keyword, "size": max_results},
                timeout=5,
            )
            resp2.raise_for_status()
            results = _parse_address(resp2.json().get("documents", []))

        return results
    except requests.RequestException as e:
        print(f"[module1] 장소 검색 오류: {e}")
        return []


def geocode(address: str, kakao_key: str) -> tuple[float, float] | None:
    """카카오 로컬 API로 주소/키워드 → (위도, 경도) 변환

    주소 검색을 먼저 시도하고 결과가 없으면 키워드 검색으로 폴백한다.
    """
    headers = {"Authorization": f"KakaoAK {kakao_key}"}

    for url in [KAKAO_GEOCODE_URL, KAKAO_KEYWORD_URL]:
        try:
            resp = requests.get(url, headers=headers, params={"query": address}, timeout=5)
            docs = resp.json().get("documents", [])
            if docs:
                return float(docs[0]["y"]), float(docs[0]["x"])  # (lat, lon)
        except Exception:
            continue

    return None


def reverse_geocode(lat: float, lon: float, kakao_key: str) -> tuple[str, str]:
    """좌표 → (시도, 시군구) 변환 (카카오 역지오코딩)

    region_input 이 비어있을 때 출발지 좌표로 시도/구를 자동 감지하는 데 사용한다.
    실패 시 빈 문자열 튜플 반환.
    """
    headers = {"Authorization": f"KakaoAK {kakao_key}"}
    try:
        resp = requests.get(
            KAKAO_REGION_URL,
            headers=headers,
            params={"x": lon, "y": lat},
            timeout=5,
        )
        docs = resp.json().get("documents", [])
        # region_type "B" = 법정동 코드 기준, 시도/시군구 포함
        for doc in docs:
            sido  = doc.get("region_1depth_name", "")
            gungu = doc.get("region_2depth_name", "")
            if sido:
                return sido, gungu
    except Exception:
        pass
    return "", ""


# ---------------------------------------------------------------------------
# 정제 (normalize)
# ---------------------------------------------------------------------------

_GEOCODE_LIMIT_PER_RUN = 30   # 실행당 최대 신규 geocode 호출 수 (쿼터 보호)
_GEOCODE_INTERVAL_SEC = 0.05  # 호출 간 최소 간격 (초)


def normalize(raw: list[dict], kakao_key: str = "") -> list[dict]:
    """raw 데이터를 표준 스키마로 정제

    faclLat/faclLng 가 없거나 0인 항목은 kakao_key 가 있으면
    lcMnad(주소) 로 geocode 를 시도한다. geocode 결과는 cache.json 에 저장한다.
    캐시 미스 시 API 호출은 _GEOCODE_LIMIT_PER_RUN 회로 제한하여 쿼터 소진을 방지한다.

    [자료구조] List[Dict]: 정제된 시설 목록
    [알고리즘] 선형탐색: raw 목록을 순회하며 필드 매핑
    """
    cache = _load_cache() if kakao_key else {}
    cache_dirty = False
    geocode_calls = 0  # 이번 실행에서 신규 API 호출 횟수

    normalized: list[dict] = []
    for item in raw:
        # 위도: faclLat / 경도: faclLng (API 문서 기준)
        lat_str = item.get("faclLat", "")
        lon_str = item.get("faclLng", "")
        lat, lon = 0.0, 0.0
        if lat_str and lon_str:
            try:
                lat, lon = float(lat_str), float(lon_str)
            except ValueError:
                pass

        # 좌표가 없거나 0,0이면 주소로 geocode 보완
        if (lat == 0.0 or lon == 0.0) and kakao_key:
            address = (item.get("lcMnad") or "").strip()
            if address:
                geo_key = f"geo_{address}"
                geo_entry = cache.get(geo_key)
                if geo_entry and time.time() - geo_entry.get("ts", 0) < CACHE_TTL_SEC:
                    # 캐시 히트 — API 호출 없음
                    coord = tuple(geo_entry["data"]) if geo_entry["data"] else None
                elif geocode_calls < _GEOCODE_LIMIT_PER_RUN:
                    # 캐시 미스 — 쿼터 한도 내에서만 API 호출
                    time.sleep(_GEOCODE_INTERVAL_SEC)
                    coord = geocode(address, kakao_key)
                    geocode_calls += 1
                    cache[geo_key] = {"ts": time.time(), "data": list(coord) if coord else None}
                    cache_dirty = True
                else:
                    coord = None  # 한도 초과 — 이번 실행에서는 건너뜀
                if coord:
                    lat, lon = coord[0], coord[1]

        if lat == 0.0 or lon == 0.0:
            continue

        # 폐업 시설 제외
        if item.get("salStaDivCd", "Y") == "N":
            continue

        # evalInfo: "승강기, 장애인사용가능화장실, 주출입구 접근로, ..."
        eval_info = item.get("evalInfo", "")

        normalized.append({
            "name":         item.get("faclNm") or "이름 없음",
            "address":      item.get("lcMnad") or "",
            "lat":          lat,
            "lon":          lon,
            "wfclt_id":     item.get("wfcltId", ""),
            "fac_type":     item.get("faclTyCd") or "",
            "has_toilet":   _TOILET_KW   in eval_info,
            "has_elevator": _ELEVATOR_KW in eval_info,
            "has_ramp":     _RAMP_KW1    in eval_info or _RAMP_KW2 in eval_info,
            "has_parking":  _PARKING_KW  in eval_info,
        })

    if cache_dirty:
        _save_cache(cache)

    return normalized


# ---------------------------------------------------------------------------
# 필터링
# ---------------------------------------------------------------------------

def filter_facilities(facilities: list[dict], filters: dict) -> list[dict]:
    """조건 필터링

    [자료구조] List: 조건을 충족한 시설만 담는 결과 리스트
    [알고리즘] 선형탐색 + 조건 필터링: O(n)
    """
    need_toilet   = filters.get("need_toilet",   False)
    need_elevator = filters.get("need_elevator",  False)
    need_ramp     = filters.get("need_ramp",      False)
    need_parking  = filters.get("need_parking",   False)

    result: list[dict] = []
    for facility in facilities:
        if need_toilet and not facility["has_toilet"]:
            continue
        if need_elevator and not facility["has_elevator"]:
            continue
        if need_ramp and not facility["has_ramp"]:
            continue
        if need_parking and not facility["has_parking"]:
            continue
        result.append(facility)

    return result


# ---------------------------------------------------------------------------
# 지역 문자열 파싱 헬퍼
# ---------------------------------------------------------------------------

def parse_region(region: str) -> tuple[str, str]:
    """'서울특별시 중구' → ('서울특별시', '중구')"""
    parts = region.strip().split(maxsplit=1)
    sido  = parts[0] if parts else ""
    gungu = parts[1] if len(parts) > 1 else ""
    return sido, gungu


# ---------------------------------------------------------------------------
# Trie — 지역명 자동완성
# ---------------------------------------------------------------------------

class _TrieNode:
    """Trie 내부 노드: 자식 맵과 단어 끝 표시"""
    __slots__ = ("children", "is_end")

    def __init__(self) -> None:
        # [자료구조] Dict: 문자 → 자식 노드 매핑 (O(1) 탐색)
        self.children: dict[str, _TrieNode] = {}
        self.is_end: bool = False


class Trie:
    """[자료구조] Trie (접두사 트리):
        문자열 집합을 공통 접두사 기준으로 압축 저장하는 트리.
        삽입·탐색 모두 O(L) — L은 문자열 길이.

    [알고리즘] 접두사 탐색 (Prefix Search + DFS):
        1) 접두사 문자를 따라 트리를 내려간다 — O(L)
        2) 도달한 노드에서 DFS로 모든 완성 문자열을 수집한다 — O(K)
           K = 결과 수
        자동완성·오타 제안·사전 구현에 표준적으로 쓰이는 알고리즘이다.
    """

    def __init__(self) -> None:
        self._root = _TrieNode()

    def insert(self, word: str) -> None:
        """문자열을 Trie에 삽입 — O(L)"""
        node = self._root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        node.is_end = True

    def prefix_search(self, prefix: str, max_results: int = 8) -> list[str]:
        """접두사로 시작하는 모든 문자열 반환 — O(L + K)

        [알고리즘] 접두사 탐색: 접두사 노드까지 이동 후 DFS로 수집
        """
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        results: list[str] = []
        self._dfs(node, prefix, results, max_results)
        return results

    def _dfs(self, node: _TrieNode, path: str, results: list[str], max_results: int) -> None:
        """DFS로 현재 노드 이하의 완성 문자열을 results에 누적"""
        if len(results) >= max_results:
            return
        if node.is_end:
            results.append(path)
        for ch, child in node.children.items():
            self._dfs(child, path + ch, results, max_results)


def build_region_trie(addresses: list[str]) -> Trie:
    """시설 주소 목록에서 시도·시군구 단위 Trie 생성

    주소 문자열의 첫 두 토큰(시도, 시군구)만 추출하여 삽입한다.
    중복 지역명은 Set으로 걸러 삽입 횟수를 최소화한다.

    [자료구조] Trie: 지역명 자동완성용 접두사 트리
    [자료구조] Set: 중복 지역명 O(1) 제거
    """
    trie = Trie()
    # [자료구조] Set: 이미 삽입한 지역명 중복 방지
    seen: set[str] = set()
    for addr in addresses:
        parts = addr.strip().split()
        if not parts:
            continue
        sido = parts[0]
        if sido and sido not in seen:
            trie.insert(sido)
            seen.add(sido)
        if len(parts) >= 2:
            sido_gungu = f"{parts[0]} {parts[1]}"
            if sido_gungu not in seen:
                trie.insert(sido_gungu)
                seen.add(sido_gungu)
    return trie
