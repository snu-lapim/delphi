
```markdown
# Introduction

Delphi was the home of a temple to Phoebus Apollo, which famously had the inscription, 'Know Thyself.' This library lets language models know themselves through automated interpretability.

This library provides utilities for generating and scoring text explanations of sparse autoencoder (SAE) and transcoder features. The explainer and scorer models can be run locally or accessed using API calls via OpenRouter.

The branch used for the article [Automatically Interpreting Millions of Features in Large Language Models](https://arxiv.org/pdf/2410.13928) is the legacy branch [article_version](https://github.com/EleutherAI/delphi/tree/article_version), that branch contains the scripts to reproduce our experiments. Note that we're still actively improving the codebase and that the newest version on the main branch could require slightly different usage.

# Installation

Install this library as a local editable installation. Run the following commands:

```bash
# 기본 패키지 설치
pip install -e .

# 추가 필수 패키지 설치
pip install lxt
pip install plotly
```

# 프로젝트 개요 및 실행 방법

## 전체 프로세스

이 프로젝트의 실행 흐름은 다음과 같습니다:

1. **e2e.py → __main__.py**: 테스트 스크립트인 e2e.py가 __main__.py를 호출하는 구조입니다.

2. **Activation Caching**: __main__.py에서는 우선 특정 레이어의 activation을 캐싱합니다. 여기서 activation이란 특정 레이어의 출력뿐 아니라, 그 출력을 SAE(Sparse Autoencoder)로 feature extract한 결과도 포함됩니다.

3. **Explainer**: 캐싱된 결과는 Explainer에 입력됩니다. Explainer는 특정 SAE feature의 의미를 설명하기 위해 더 성능이 좋은 LLM을 사용합니다. 기존에는 각 토큰마다 해당 feature가 얼마나 activate 되었는지 예시를 모아 LLM에게 feature의 의미를 서술하게 했습니다. 우리의 새로운 접근법은 오직 한 토큰의 activation과 그 토큰을 누가 만들었는지에 대한 기여도를 LLM에 전달합니다. e2e.py의 run_cfg 옵션 중 `apply_attnlrp=True`로 설정하면 이 방식이 적용됩니다.

4. **Scorer**: Explainer가 생성한 feature 개념(예: "feature_0: word relevant to cat")을 바탕으로, Scorer는 모르는 문장이 입력되었을 때 특정 토큰에서 실제로 feature가 activate 되었는지 예측하고 그 정확도를 평가합니다.

## 실행 방법

가장 간단한 실행 방법은 e2e.py 스크립트를 실행하는 것입니다:

```bash
python -m tests.e2e
```

## 팀원들이 해야 할 일

1. **프롬프트 수정**: prompts.py 파일의 프롬프트를 우리의 새로운 방법(attnlrp)에 맞게 수정해야 합니다.

2. **런타임 개선**: 현재 explainer_provider가 'openrouter'로 설정되어 있어 latency가 길어 실험이 오래 걸립니다. 이를 'offline'으로 변경하고 적절한 로컬 모델을 찾아 런타임을 개선해야 합니다. 단, 사용할 모델은 CoT(Chain of Thought) 추론이 가능한 모델이어야 합니다.

3. **실험 실행**: 수정된 코드로 실험을 실행해 보고 결과를 분석합니다.
