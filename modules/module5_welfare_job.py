# module5_welfare_job.py
# 복지서비스 및 장애인 취업 데이터 수집
# [자료구조] List, Dict
# [알고리즘] 선형탐색, 조건 필터링

import xml.etree.ElementTree as ET

import requests

# ── 데이터 출처 ──────────────────────────────────────────────────────────────
# 한국사회보장정보원_지자체복지서비스  (공공데이터포털 15108347)
_LOCAL_WELFARE_URL   = "https://apis.data.go.kr/B460010/wlfareinfo/wlfareInfoList"
# 한국사회보장정보원_중앙부처복지서비스 (공공데이터포털 15090532)
_CENTRAL_WELFARE_URL = "https://apis.data.go.kr/B460010/govwlfareinfo/wlfareInfoList"
# 한국장애인고용공단_장애인구인실시간현황 (공공데이터포털 15117692)
_JOB_URL             = "https://apis.data.go.kr/B552474/DisabledPersonJobInfo/getJobList"

# 시도명 → 복지로 API 시도코드
_SIDO_CODE: dict[str, str] = {
    "서울": "11", "부산": "26", "대구": "27", "인천": "28",
    "광주": "29", "대전": "30", "울산": "31", "세종": "36",
    "경기": "41", "강원": "51", "충북": "43", "충남": "44",
    "전북": "52", "전남": "46", "경북": "47", "경남": "48", "제주": "50",
}


def _sido_code(sido: str) -> str:
    for k, v in _SIDO_CODE.items():
        if k in sido:
            return v
    return "11"


def _parse_response(resp: requests.Response) -> list[dict]:
    """JSON 우선 파싱, 실패 시 XML, 둘 다 실패 시 빈 리스트 반환."""
    # JSON 시도
    try:
        body = resp.json()
        # 복지로 응답 구조: {"wlfareinfo": {"list": {"item": [...]}}}
        for root_key in ("wlfareinfo", "response"):
            node = body.get(root_key, {})
            # response → body → items → item (공공데이터 표준 구조)
            if root_key == "response":
                items = node.get("body", {}).get("items", {}).get("item", [])
            else:
                items = node.get("list", {}).get("item", [])
            if items:
                return [items] if isinstance(items, dict) else items
        return []
    except Exception:
        pass

    # XML 폴백
    try:
        root = ET.fromstring(resp.text)
        items = []
        for item in root.iter("item"):
            d = {child.tag: (child.text or "").strip() for child in item}
            if d:
                items.append(d)
        return items
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 복지서비스
# ─────────────────────────────────────────────────────────────────────────────

def fetch_local_welfare(api_key: str, sido: str, num: int = 100) -> list[dict]:
    """지자체 복지서비스 목록 — 시도 필터 + 장애인 관련 client-side 필터
    [알고리즘] 조건 필터링: 서비스명/요약에 '장애' 포함 항목만 추출
    """
    params = {
        "serviceKey": api_key,
        "pageNo":     1,
        "numOfRows":  num,
        "siDoCd":     _sido_code(sido),
        "_type":      "json",
    }
    try:
        resp = requests.get(_LOCAL_WELFARE_URL, params=params, timeout=10)
        return _parse_response(resp)
    except Exception:
        return []


def fetch_central_welfare(api_key: str, num: int = 100) -> list[dict]:
    """중앙부처 복지서비스 목록 — 전국 단위 (지역 필터 없음)"""
    params = {
        "serviceKey": api_key,
        "pageNo":     1,
        "numOfRows":  num,
        "_type":      "json",
    }
    try:
        resp = requests.get(_CENTRAL_WELFARE_URL, params=params, timeout=10)
        return _parse_response(resp)
    except Exception:
        return []


# 복지서비스에서 '장애' 관련 항목을 필터링하는 키워드
_DISABILITY_KW = ("장애", "휠체어", "배리어프리", "접근성", "재활", "보조기기")


def normalize_welfare(raw: list[dict], source: str = "") -> list[dict]:
    """복지서비스 raw → 표준 스키마 변환 + 장애 관련 필터링
    [자료구조] List[Dict]: 표준화된 복지서비스 목록
    [알고리즘] 선형탐색: 키워드 포함 여부 확인
    """
    result: list[dict] = []
    for it in raw:
        name    = (it.get("wlfareInfoNm") or it.get("servNm") or "").strip()
        summary = (it.get("wlfareInfoOutlCn") or it.get("servDgst") or "").strip()
        target  = (it.get("wlfareTrgtNm") or it.get("tgtrDvNm") or "").strip()

        # [알고리즘] 조건 필터링: 장애 관련 서비스만 포함
        combined = name + summary + target
        if not any(kw in combined for kw in _DISABILITY_KW):
            continue

        result.append({
            "id":       it.get("wlfareInfoId") or it.get("servId") or "",
            "name":     name or "서비스명 없음",
            "summary":  summary,
            "target":   target,
            "theme":    (it.get("intrsThemaNm") or it.get("lifeNm") or "").strip(),
            "apply":    (it.get("aplyMthdCn") or it.get("aplyMthd") or "").strip(),
            "criteria": (it.get("slctCritCn") or it.get("slctCrit") or "").strip(),
            "contact":  (it.get("wlfareInfoCtadr") or it.get("rprsOrgnNm") or "").strip(),
            "source":   source,
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 장애인 취업
# ─────────────────────────────────────────────────────────────────────────────

def fetch_disability_jobs(
    api_key: str,
    sido: str = "",
    job_category: str = "",
    num: int = 50,
) -> list[dict]:
    """장애인 구인 실시간 현황 조회
    [자료구조] List[Dict]: 구인 목록
    """
    params: dict = {
        "serviceKey": api_key,
        "pageNo":     1,
        "numOfRows":  num,
        "_type":      "json",
    }
    # 시도명 필터 (API 지원 파라미터명 — 실제 명세 확인 후 조정 가능)
    if sido:
        params["SIDO_NM"] = sido
    if job_category:
        params["JOB_CD"] = job_category

    try:
        resp = requests.get(_JOB_URL, params=params, timeout=10)
        return _parse_response(resp)
    except Exception:
        return []


def normalize_jobs(raw: list[dict]) -> list[dict]:
    """구인 raw → 표준 스키마 변환
    [자료구조] List[Dict]: 표준화된 구인 목록
    """
    def _pick(it: dict, *keys: str) -> str:
        """여러 후보 키 중 첫 번째 비어있지 않은 값 반환 (API 필드명 불확실성 대응)"""
        for k in keys:
            v = str(it.get(k, "")).strip()
            if v and v != "None":
                return v
        return ""

    result: list[dict] = []
    for it in raw:
        result.append({
            "company":  _pick(it, "COMPANY_NM", "corpNm", "entNm", "CORP_NM"),
            "job":      _pick(it, "JOB_CATEGORY_NM", "jobCategoryNm", "jobNm", "JOB_NM"),
            "employ":   _pick(it, "EMPLOYMENT_TYPE_NM", "empTypNm", "EMPLOY_TYPE_NM"),
            "salary":   (
                _pick(it, "SALARY", "salaryAmt", "SALARY_AMT")
                + " " +
                _pick(it, "SALARY_TYPE_NM", "salaryTypNm", "SALARY_TYPE")
            ).strip(),
            "deadline": _pick(it, "DEAD_LINE_DATE", "deadLineDt", "DEADLINE_DT"),
            "address":  _pick(it, "WORK_PLACE_ADDR", "workplaceAddr", "WORK_ADDR", "ADDR"),
            "career":   _pick(it, "CAREER_PERIOD_NM", "careerPeriod", "CAREER_NM"),
            "edu":      _pick(it, "EDU_REQ_NM", "eduReqNm", "EDU_NM"),
            "count":    _pick(it, "HIRE_COUNT", "hireCnt", "HIRE_CNT") or "1",
            "biz_type": _pick(it, "COMPANY_TYPE_NM", "companyTypNm", "CORP_TYPE_NM"),
            "contact":  _pick(it, "CONTACT", "tel", "TEL"),
        })
    return [j for j in result if j["company"] or j["job"]]
