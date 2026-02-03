# xpuruntime에 기여하기

xpuruntime은 **오픈 소스·비영리**로 운영되며, 커뮤니티 기여를 환영합니다.

## 어떻게 기여할 수 있나요?

- **버그 리포트**: [Issues](https://github.com/devyulbae/xpu_runtime/issues)에서 Bug report 템플릿으로 이슈를 열어 주세요.
- **기능 제안**: Feature request 템플릿으로 아이디어를 공유해 주세요.
- **코드/문서 기여**: 아래 개발 환경 설정 후 PR을 보내 주세요. `good first issue` / `help wanted` 라벨이 붙은 이슈부터 도전해 보시면 좋습니다.

## 개발 환경 설정

1. **저장소 클론**
   ```bash
   git clone https://github.com/devyulbae/xpu_runtime.git
   cd xpu_runtime
   ```

2. **Python 환경 (필수)**
   ```bash
   py -3.14 -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/macOS: source venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **테스트 실행**
   ```bash
   pytest tests/python -v
   ```

4. **C++/CUDA 빌드 (선택)**  
   CUDA Toolkit + CMake가 있는 환경에서만 필요합니다.  
   자세한 내용은 [docs/plans/07_build_packaging.md](docs/plans/07_build_packaging.md)를 참고하세요.

## Pull Request 절차

1. 이 저장소를 fork한 뒤, 본인 fork에서 브랜치를 만듭니다.
2. 변경 후 `pytest tests/python -v`로 테스트가 통과하는지 확인합니다.
3. PR을 열 때 **어떤 이슈를 다루는지**, **변경 요약**을 적어 주세요.
4. 리뷰 반영 후 maintainer가 머지합니다.

## 코드 스타일

- **Python**: [Ruff](https://docs.astral.sh/ruff/), [Black](https://black.readthedocs.io/) 스타일을 따릅니다. (`pip install -e ".[dev]"` 시 포함)
- **C++**: 프로젝트 내 기존 스타일(들여쓰기, 네이밍)을 유지해 주세요.

## 행동 강령

참여 시 서로 존중하는 커뮤니티를 위해, 상대방을 비하하거나 차별하는 언행은 자제해 주세요.  
문제가 있을 경우 이슈나 maintainer에게 연락해 주시면 됩니다.

---

질문이 있으면 [Issues](https://github.com/devyulbae/xpu_runtime/issues)에 이슈를 열어 주세요.
