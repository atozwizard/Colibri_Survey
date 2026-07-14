#!/usr/bin/env python3
"""서베이 전 문서를 '요약 없이' 하나의 종합 보고서(master.md)로 결합.
- 각 docs/*.md 를 전문 그대로 포함(헤딩만 한 단계 강등해 부/장 아래로 중첩).
- 코드펜스(```) 내부의 '#' 은 헤딩으로 오인하지 않도록 보호.
- pandoc 으로 docx 변환(별도 명령).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "report" / "master.md"
IMG = ROOT / "report" / "img"

# mermaid CLI: env MMDC 우선, 없으면 스크래치 설치본, 없으면 PATH.
MMDC = os.environ.get("MMDC") or "/tmp/mermaid-tools/node_modules/.bin/mmdc"
if not Path(MMDC).exists():
    MMDC = shutil.which("mmdc") or MMDC
PPTR = "/tmp/mermaid-tools/pptr.json"
RENDER = os.environ.get("RENDER_MERMAID", "1") == "1"

MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL)

# 부(Part) 구성: (제목, [문서 stem prefix])
PARTS = [
    ("제1부 · 개요와 아키텍처", ["00-overview", "10-colibri-architecture"]),
    ("제2부 · 핵심 기술", ["20-moe-streaming", "21-mla-kv-compression", "22-speculative-decoding"]),
    ("제3부 · 로컬 실행과 정확성 검증", ["30-local-run-notes", "31-engine-verification"]),
    ("제4부 · 논리 분석: 장단점·자원", ["40-analysis-tradeoffs", "50-resource-requirements"]),
    ("제5부 · 타 모델 적용", ["60-applying-to-other-models", "61-apply-gpt-oss-20b", "62-apply-gemma4"]),
    ("제6부 · 경영·기술 통합 브리프", ["70-executive-brief"]),
    ("제7부 · 실적용: ThinkFlow(H100) & 변환·실측",
     ["80-olmoe-and-h100-recommendations", "81-thinkflow-upgrade-design",
      "82-gpt-oss-mxfp4-to-int4-converter", "83-olmoe-streaming-measurement",
      "84-thinkflow-swap-checklist", "85-dual-model-crossval-resources"]),
    ("제8부 · 라이선스·상태·마감·참고문헌",
     ["91-licensing-compliance", "90-status-and-closeout", "99-references"]),
]

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})(\s)")


def demote(md: str, by: int = 1) -> str:
    """코드펜스 밖의 헤딩만 by 단계 강등."""
    out, in_fence = [], False
    for line in md.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence:
            m = HEADING.match(line)
            if m:
                line = "#" * min(len(m.group(1)) + by, 6) + m.group(2) + line[m.end():]
        out.append(line)
    return "\n".join(out)


def find_doc(stem: str) -> Path:
    p = DOCS / f"{stem}.md"
    if not p.is_file():
        raise SystemExit(f"문서 없음: {p}")
    return p


def render_mermaid(md: str, stem: str) -> str:
    """```mermaid 블록을 png로 렌더해 이미지 링크로 치환. 실패 시 원본 유지."""
    if not RENDER:
        return md
    IMG.mkdir(parents=True, exist_ok=True)
    idx = 0

    def repl(m: re.Match) -> str:
        nonlocal idx
        idx += 1
        src = m.group(1)
        base = f"{stem}-{idx}"
        mmd = IMG / f"{base}.mmd"
        png = IMG / f"{base}.png"
        mmd.write_text(src, encoding="utf-8")
        cmd = [MMDC, "-i", str(mmd), "-o", str(png), "-b", "white", "-s", "2"]
        if Path(PPTR).exists():
            cmd += ["-p", PPTR]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except Exception as e:  # 실패하면 원본 코드블록 유지
            print(f"  ! mermaid 렌더 실패({base}): {e}")
            return m.group(0)
        # 페이지 폭(약 6.2in)에 맞게 이미지 폭 제약. -s2 고해상도라 192dpi로 환산.
        w_in = 6.2
        try:
            raw = png.read_bytes()[:24]
            if raw[12:16] == b"IHDR":
                px = int.from_bytes(raw[16:20], "big")
                w_in = min(6.2, round(px / 192, 2))
        except Exception:
            pass
        return f"![{stem} 다이어그램 {idx}](img/{base}.png){{width={w_in}in}}"

    return MERMAID_BLOCK.sub(repl, md)


FRONT = """---
title: "colibrì 추론 엔진 종합 서베이 보고서"
subtitle: "GLM-5.2 744B MoE 디스크 스트리밍 엔진 · 분석 · 로컬 검증 · ThinkFlow(H100) 실적용"
author: "Colibri_Survey · atozwizard/Colibri_Survey"
date: "2026-07-14"
lang: ko
toc-title: "목차"
---

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 보고서 서문 · 독해 안내

본 보고서는 colibrì(이하 colibri) 추론 엔진 서베이의 **모든 산출 문서를 요약 없이 전문 그대로** 하나로 묶은 종합본이다. 각 부(Part)는 원 문서를 그대로 포함하며, 내용의 삭제·축약은 없다. 부 서두의 짧은 안내문만 새로 추가되었다.

- **대상**: colibri = GLM-5.2 744B MoE를 int4로 양자화하고 expert를 디스크에서 스트리밍하여, GPU 없이 또는 소용량 하드웨어에서 초대형 MoE를 구동하는 C 추론 엔진.
- **방법**: 벤더링된 원본 소스(`external/colibri`) 직접 분석 + 로컬(Apple Silicon) 빌드·실행 + tiny GLM oracle token-exact 검증 + 실서버(ThinkFlow, H100 PCIe 80GB) 적용 설계.
- **독해**: 제1~2부(구조·핵심기술) → 제3부(검증) → 제4~6부(분석·종합) → 제7부(실적용) → 제8부(상태·참고) 순서를 권장한다.
- **표기**: 코드/설정/다이어그램(mermaid)은 원문 그대로 코드 형식으로 포함한다. 다이어그램은 mermaid 소스로 표기되며, 렌더링 도구(mermaid)로 그림화할 수 있다.

## 문서 상호 관계 지도

```
개요(00) ─ 아키텍처(10)
                 └ 핵심기술: MoE스트리밍(20)·MLA(21)·speculative(22)
로컬실행(30) ─ 정확성검증(31: C테스트+oracle 32/32)
분석: tradeoff(40)·자원(50) ─ 종합브리프(70)
타모델: 일반(60)·gpt-oss-20b(61)·gemma4(62)
실적용: OLMoE위상·H100추천(80) ─ ThinkFlow설계(81)·런북(84)
        변환기(82: MXFP4→int4) ─ 스트리밍실측(83)
상태·마감(90) ─ 참고문헌(99)
```
"""


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = [FRONT]
    for title, stems in PARTS:
        buf.append(f"\n\n# {title}\n")
        for stem in stems:
            md = find_doc(stem).read_text(encoding="utf-8")
            md = render_mermaid(md, stem)          # mermaid → 이미지
            buf.append("\n\n" + demote(md, by=1) + "\n")
    OUT.write_text("\n".join(buf), encoding="utf-8")
    n = OUT.read_text(encoding="utf-8").count("\n")
    print(f"master.md 작성: {OUT}  ({n} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
