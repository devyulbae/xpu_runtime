# TASK_014: PyPI 배포

> Phase 5: Release

---

## 개요

PyPI에 xpuruntime 패키지를 배포한다.

## 목표

- PyPI 계정/토큰 설정
- 자동 배포 워크플로우
- 버전 관리 자동화
- 릴리스 노트 생성

## 선행 작업

- TASK_013: CI/CD 설정

## 작업 내용

### 1. PyPI 토큰 설정

```bash
# GitHub Secrets에 추가:
# - PYPI_API_TOKEN: PyPI API 토큰
# - TEST_PYPI_API_TOKEN: TestPyPI API 토큰 (테스트용)
```

### 2. 릴리스 워크플로우

```yaml
# .github/workflows/release.yml
name: Release

on:
  release:
    types: [published]

jobs:
  build:
    uses: ./.github/workflows/build-wheels.yml
  
  publish-testpypi:
    needs: build
    runs-on: ubuntu-latest
    environment: testpypi
    steps:
      - name: Download wheels
        uses: actions/download-artifact@v3
        with:
          name: wheels
          path: dist/
      
      - name: Publish to TestPyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
          password: ${{ secrets.TEST_PYPI_API_TOKEN }}
  
  publish-pypi:
    needs: publish-testpypi
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - name: Download wheels
        uses: actions/download-artifact@v3
        with:
          name: wheels
          path: dist/
      
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

### 3. 버전 관리

```python
# src/python/xpuruntime/__init__.py
__version__ = "0.1.0"

# pyproject.toml에서 dynamic version 사용 가능
# [project]
# dynamic = ["version"]
# [tool.setuptools.dynamic]
# version = {attr = "xpuruntime.__version__"}
```

### 4. CHANGELOG

```markdown
# Changelog

## [0.1.0] - 2024-XX-XX

### Added
- Initial release
- InferenceSession with TensorRT and ONNX Runtime support
- KernelPolicy for explicit kernel selection
- PyTorch extension for training integration
- DeviceManager, MemoryManager, StreamManager

### Changed
- N/A

### Fixed
- N/A
```

### 5. 릴리스 체크리스트

```markdown
# Release Checklist

## Pre-release
- [ ] 모든 테스트 통과
- [ ] CHANGELOG 업데이트
- [ ] 버전 번호 업데이트 (pyproject.toml, __init__.py, CMakeLists.txt)
- [ ] README 최신화
- [ ] 문서 빌드 확인

## Release
- [ ] GitHub Release 생성
- [ ] 태그 생성 (v0.1.0)
- [ ] CI/CD 워크플로우 완료 확인
- [ ] TestPyPI 배포 확인
- [ ] PyPI 배포 확인

## Post-release
- [ ] pip install xpuruntime 테스트
- [ ] 릴리스 공지 (GitHub Discussions, 소셜 미디어)
- [ ] 다음 버전 개발 시작
```

### 6. 설치 테스트

```bash
# TestPyPI에서 테스트
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            xpuruntime

# PyPI에서 설치
pip install xpuruntime

# 버전 확인
python -c "import xpuruntime; print(xpuruntime.__version__)"
```

## 완료 조건

- [ ] PyPI 계정 및 토큰 설정 완료
- [ ] TestPyPI 배포 성공
- [ ] PyPI 배포 성공
- [ ] `pip install xpuruntime` 동작 확인
- [ ] 버전 자동 관리 동작

## 예상 소요 시간

4-6시간

## 관련 문서

- [07_build_packaging.md](../plans/07_build_packaging.md)
