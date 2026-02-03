# Issue 초안: Flash Attention 2 Windows 지원 / 폴백

> GitHub에서 새 Issue 열 때 아래 내용을 복사해 사용하세요.

---

## Title

**[Enhancement] Flash Attention 2 Windows 지원 또는 플랫폼별 폴백 정책**

---

## Body

### 배경

- HF / PyTorch로 모델을 가져다 쓸 때 Flash Attention 2가 이미 포함된 경우가 많음 (e.g. `attn_implementation="flash_attention_2"`, SDPA flash backend).
- Flash Attention 2는 Windows에서 소스 빌드 시 MSVC + 템플릿 이슈(예: `Headdim` C2975, `kBlockM` 참조)로 실패하는 경우가 많고, 그 결과 해당 경로를 쓰는 모델이 Windows에서 동작하지 않음.
- 참고: [Dao-AILab/flash-attention#595](https://github.com/Dao-AILab/flash-attention/issues/595) 등에 Windows 빌드 우회 방법이 제안되어 있음.

### 제안

1. **플랫폼 인식 + 폴백 (단기)**  
   - Dispatcher / KernelRegistry 연동: Windows 환경에서는 Flash Attention 커널을 **선택하지 않거나** 선호도에서 제외하고, cuDNN attention / eager 등 **동작하는 구현으로 자동 폴백**.
   - 문서화: “Flash Attention: Linux 권장, Windows는 폴백 사용” 등 명시.

2. **Flash Attention 2 Windows 빌드 지원 (중장기, 선택)**  
   - [flash-attention#595](https://github.com/Dao-AILab/flash-attention/issues/595) 스타일 패치를 적용한 포크 또는 vendor 도입 검토.
   - 또는 Windows용 기성 wheel/포크를 의존성으로 두고, 우리 런타임에서는 해당 구현체를 커널로 등록하는 방식 검토.

### 기대 효과

- HF/torch 기반 모델을 Windows에서 쓸 때, Flash Attention 경로 대신 동작하는 attention 백엔드로 넘어가서 런타임이 깨지지 않도록 함.
- (중장기) Windows에서도 Flash Attention 2 경로를 쓸 수 있게 되면, 동일 모델·설정을 OS 구분 없이 사용 가능.

### 라벨 제안

`enhancement`, `help wanted`
