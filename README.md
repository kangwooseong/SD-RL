# SD-RL

ViSpec 기반 강화학습 적용: ViSpec의 고정된 draft 깊이와 검증 후보 수를 PPO 정책으로 동적으로 조절해, Target 모델의 출력 분포를 유지하면서 추론 처리량을 높이는 것이 목표입니다.

- 기반 코드: [ViSpec](https://github.com/KangJialiang/ViSpec) (NeurIPS 2025, Apache-2.0, 커밋 `49a33ff`). 원본 설명은 `README_ViSpec.md`에 있습니다.
- 참고 방법: [Learning to Draft](https://github.com/zhzihao/Learning-to-Draft) (ICLR 2026)의 depth/size 두 정책 구조를 따릅니다.

> **상태: 아직 GPU에서 실행해 보지 않은 코드입니다.** 문법 검사만 통과했습니다. 본 학습 전에 아래 "실험 시작 전 체크리스트"를 순서대로 진행하세요.

## 구조

ViSpec과 Target VLM은 고정하고 작은 MLP 정책 두 개만 학습합니다.

| 구성 | 내용 |
|---|---|
| depth 정책 | 트리를 한 단계 확장할 때마다 계속(1) / 중단(0) 결정 |
| size 정책 | Target에 검증을 맡길 draft 토큰 수 결정 (기본 선택지 10~60) |
| 상태 | draft 토큰의 누적 점수, top-1 확률, 마진, 엔트로피 / 현재 깊이 / context 길이, 이미지 토큰 비율 / 최근 수락 이력 / ViSpec의 전역 시각 특징 `g_v` |
| 학습 | PPO (`vispec/rl/ppo.py`, 추가 의존성 없음) |

정책은 트리 모양만 바꾸고 수락/거부는 그대로 Target이 판정하므로, 출력 분포는 정책과 무관하게 보존됩니다. 그래서 보상에는 품질 항이 없습니다.

### ViSpec 원본에서 바뀐 파일

| 파일 | 변경 |
|---|---|
| `vispec/model/cnets_ours.py` (`topK_genrate`) | 확장 루프 안에 depth 결정, 후보 선택 직전에 size 결정 훅 추가. 트리가 얕을 때 후보 수가 노드 수를 넘지 않게 상한 처리 |
| `vispec/model/spec_model_ours.py` (`specgenerate`, `from_pretrained`) | 사이클별 시간 측정과 보상 훅 추가. 환경변수 `VISPEC_RL_POLICY`로 정책 연결 |
| `vispec/rl/` (신규) | `controller.py`(상태·행동·보상), `cost_model.py`(시간 비용 모델), `data.py`(학습·검증 프롬프트), `ppo.py`, `train_ppo.py` |

수정한 줄에는 `[RL]` 주석이 있습니다. 정책을 연결하지 않으면 동작은 원본과 같습니다.

## 보상 설계

한 사이클 = draft 트리 생성 → Target 검증 → 수락.

**기본 (`--reward-mode gain`)**

```
r = 0.25 × ( 생성 토큰 수 − rate_ref × (draft 시간 + 검증 시간) )
```

- `생성 토큰 수` = 수락된 draft 토큰 수 + 1 (Target이 검증하며 직접 내는 토큰)
- `rate_ref` = 기준 처리량(토큰/초). 학습 시작 시 **ViSpec 고정 설정의 처리량**으로 두고, PPO 업데이트마다 한 번씩 최근 rollout의 처리량 쪽으로 갱신합니다(`--rate-ema`). rollout 도중에는 바뀌지 않습니다.
- r > 0 이면 기준보다 빠른 사이클입니다. 수렴하면 전체 토큰 수 / 전체 시간을 최대화합니다.
- 보상을 사이클 간에 더할 수 있으므로, 한 번의 생성 안에서 사이클을 넘어 return을 누적합니다 (GAE, `--gamma 0.97`, `--gae-lambda 0.9`). 사이클 내부 결정 사이의 할인은 1입니다.

**비교용 (`--reward-mode throughput`)**

```
r = 0.01 × 생성 토큰 수 / (draft 시간 + 검증 시간)
```

Learning to Draft 방식의 사이클별 처리량입니다. 사이클 하나가 하나의 에피소드입니다.

**시간 (`--time-source`)**

- `measured` (기본): CUDA 동기화 후 실측한 시간. 정책 추론 시간도 draft 시간에 포함됩니다.
- `model`: 보정 단계에서 맞춘 선형 비용 모델의 예측값. 보상 잡음이 없어지고 학습 중 동기화도 필요 없습니다. 보정 로그의 `r2_draft`, `r2_verify`가 낮으면 쓰지 마세요.

공통: 프롬프트 인코딩이 섞이는 첫 사이클과, 검증되지 못한 마지막 draft는 학습에서 제외합니다. 최종 속도 수치는 항상 실측으로 보고해야 합니다.

## 설치

```bash
pip install -r requirements.txt
```

## 실험 시작 전 체크리스트

본 학습에 들어가기 전에 아래 순서대로 진행합니다. 앞 단계가 통과하지 않으면 다음 단계의 숫자는 믿을 수 없습니다. 통과 기준 중 수치(%)는 권장값이며 팀에서 조정할 수 있습니다.

공통 변수

```bash
BASE=llava-hf/llava-v1.6-vicuna-7b-hf
SPEC=JLKang/ViSpec-llava-v1.6-vicuna-7b-hf
COMMON="--base-model-path $BASE --spec-model-path $SPEC --model-id test"
VISPEC="--num-q 2 --top-k 8 --use-ours=True"
```

결과 파일(`<bench-name>/test-temperature-0.0.jsonl`)에는 문항별 `new_tokens`, `wall_time`, `acceptance_length`, 생성 텍스트(`turns`)가 들어 있습니다. 처리량은 문항별 `new_tokens / wall_time`의 평균이고, 배율은 `vispec/evaluation/speed.py`와 같은 방식(ViSpec 평균 ÷ baseline 평균)으로 계산합니다. `speed.py`는 폴더 이름이 코드에 고정돼 있으므로, 아래 폴더 이름을 쓰거나 스크립트의 목록을 수정해서 사용합니다.

### 0. 측정 조건 고정

- 모든 측정을 같은 GPU 한 장에서 하고, 측정 중에는 그 GPU에 다른 작업을 올리지 않습니다.
- GPU 모델, 드라이버, PyTorch/transformers 버전을 기록해 둡니다(`requirements.txt` 기준).
- 같은 설정을 3회 이상 반복해 평균과 표준편차를 구합니다. 이 표준편차가 이후 단계의 "오차 범위"입니다.

통과 기준: 같은 설정 반복 측정의 처리량 편차가 2~3% 이내.

### 1. ViSpec 재현

```bash
# Target 단독 (baseline)
python -m vispec.evaluation.gen_baseline_answer_coco_caption $COMMON \
  --bench-name baseline_results/coco_caption_test/baseline_llava/ --temperature 0

# ViSpec 기본 설정 (깊이 3, 후보 30)
python -m vispec.evaluation.gen_spec_answer_coco_caption $COMMON $VISPEC \
  --bench-name results/coco_caption_test/llava_2q/ --depth 3 --total-token 30 --temperature 0
```

`--temperature 1`로도 한 번씩 실행합니다. 전체 벤치마크는 `baseline.sh`, `exp.sh`로 돌립니다.

통과 기준: 배율이 ViSpec 논문의 같은 모델·데이터셋 수치와 비슷한 수준. 크게 다르면 환경 문제부터 해결합니다. 이 결과가 이후 모든 비교의 기준선입니다.

### 2. RL 훅이 원본 동작을 바꾸지 않는지 확인

```bash
# (a) 원본 코드: "Import ViSpec" 커밋으로 이동해 1번의 ViSpec 명령을 실행
git checkout <Import ViSpec 커밋>   # 결과 폴더: results/coco_caption_test/orig/
git checkout rl-draft-controller

# (b) 이 브랜치, 환경변수 없음: 1번 결과를 그대로 사용

# (c) 컨트롤러만 연결, 고정값 사용
VISPEC_RL_POLICY=fixed python -m vispec.evaluation.gen_spec_answer_coco_caption $COMMON $VISPEC \
  --bench-name results/coco_caption_test/hook_fixed/ --depth 3 --total-token 30 --temperature 0
```

통과 기준: (a), (b), (c)의 생성 텍스트가 temperature 0에서 모두 동일하고, 처리량 차이가 0번의 오차 범위 이내. 다르면 훅 코드에 문제가 있는 것이므로 학습을 시작하지 않습니다.

### 3. 시간 분해 확인

```bash
VISPEC_RL_POLICY=fixed VISPEC_RL_PROFILE=profile_vispec.json \
python -m vispec.evaluation.gen_spec_answer_coco_caption $COMMON $VISPEC \
  --bench-name results/coco_caption_test/profile/ --depth 3 --total-token 30 --temperature 0
```

`profile_vispec.json`에 사이클당 draft 시간, 검증 시간, 사이클당 토큰 수가 기록됩니다. 매 사이클 동기화하므로 이 실행의 속도는 대표 수치로 쓰지 않습니다.

통과 기준: 값이 상식적일 것(검증 시간이 draft 시간보다 큼, 사이클당 토큰 수가 1번 결과의 평균 수락 길이 + 1과 비슷함). draft 시간 비중이 매우 작으면 depth 정책으로 얻을 이득이 작다는 뜻이므로 4번 결과와 함께 판단합니다.

### 4. 고정값 sweep (개선 여지 확인, 진행 여부 판단)

```bash
for d in 1 2 3 4 5 6; do for n in 10 20 30 40 50 60; do
python -m vispec.evaluation.gen_spec_answer_coco_caption $COMMON $VISPEC \
  --bench-name results/coco_caption_test/sweep_d${d}_n${n}/ --depth $d --total-token $n --temperature 0
done; done
```

조합마다 모델을 다시 올리므로 시간이 오래 걸립니다. 처음에는 격자를 줄여서(예: 깊이 2·3·4·6, 후보 20·30·40·60) 돌리고, 성격이 다른 데이터셋 하나(예: ScienceQA 또는 TextVQA)에서도 반복합니다.

확인할 것

- 데이터셋별 최적 고정값과, 그것이 ViSpec 기본값(3, 30)보다 얼마나 빠른지
- 최적 조합이 데이터셋마다 다른지
- 가능하면 문항별 최적 조합을 골랐을 때의 처리량(동적 조절의 상한 추정)

통과 기준(권장): 최적 고정값이 데이터셋마다 다르거나, 문항별 최적의 이득이 기본값 대비 5% 이상이면 진행합니다. 기본값이 모든 곳에서 최적이고 이득이 오차 범위 수준이면 RL로 얻을 것이 거의 없으므로 방향을 재검토합니다.

이 결과로 학습 옵션도 정합니다. 최적 조합이 격자의 가장자리에 있으면 `--max-depth`, `--size-choices` 범위를 넓힙니다.

### 5. 비교군 확정

- **최적 고정값**: 4번에서 찾은 데이터셋별 최적 조합. "잘 튜닝한 ViSpec"에 해당하며 반드시 이겨야 하는 대상입니다.
- **규칙 기반 동적 방식**: 확신도 임계값으로 draft를 멈추는 방식 등. 이 저장소에는 아직 구현되어 있지 않으므로 별도 구현이 필요합니다.
- **시각 특징 없는 정책**: `--no-visual`로 학습한 정책.

통과 기준: 본 실험 전에 비교군 목록과 각 비교군의 설정이 문서로 확정되어 있을 것.

### 6. 학습 스모크 테스트

```bash
python -m vispec.rl.train_ppo --base-model-path $BASE --spec-model-path $SPEC \
  --dataset scienceqa --train both --save-dir rl_ckpt/smoke \
  --total-cycles 8000 --rollout 1024 --val-num 10 --val-every 2
```

통과 기준

- 에러 없이 끝나고 `rl_ckpt/smoke/train_log.jsonl`, `policy_last.pt`가 생성됨
- 보정 로그의 `vispec_rate_measured`가 3번에서 구한 처리량과 비슷함
- `r2_draft`, `r2_verify`가 대략 0.8 이상(`--time-source model`을 쓸 계획일 때만 필요)
- `action_dist`가 처음부터 한 행동으로 쏠리지 않고, `entropy`가 서서히 줄어듦
- `val_speedup_vs_vispec`가 기록됨(이 단계에서는 1보다 작아도 됨)

### 7. 무손실 확인

```bash
VISPEC_RL_POLICY=rl_ckpt/smoke/policy_last.pt \
python -m vispec.evaluation.gen_spec_answer_coco_caption $COMMON $VISPEC \
  --bench-name results/coco_caption_test/rl_smoke/ --depth 3 --total-token 30 --temperature 0
```

1번 baseline 결과와 문항별 생성 텍스트(`choices[0].turns`)를 비교합니다.

통과 기준: 텍스트가 일치함. 트리 모양이 달라지면 반정밀도 연산 오차로 드물게 달라질 수 있으므로 불일치 건수를 기록하고, 1번의 ViSpec 대 baseline 불일치 건수와 비슷한 수준인지 확인합니다. 불일치가 훨씬 많으면 훅 코드를 다시 점검합니다.

### 8. 본 학습 시작

0~7이 모두 통과하면 "학습" 항목의 순서대로 진행합니다. 0번과 4번의 결과(오차 범위, sweep 표)는 논문의 실험 설정과 ablation에 그대로 쓰이므로 보관합니다.

## 데이터

정책 학습에는 프롬프트(이미지 + 질문)만 필요합니다. 보상이 속도에서만 나오므로 정답 라벨은 쓰지 않습니다. 모든 세트는 평가 벤치마크가 쓰는 문항과 겹치지 않게 나눠져 있습니다(`vispec/rl/data.py`).

| 이름 | 학습 | 검증 | 벤치마크가 쓰는 부분 | 준비 |
|---|---|---|---|---|
| `llava_pretrain` | seed-42 셔플의 68,000번 이후 | 셔플 끝 500개 | 벤치마크 아님. ViSpec draft 모델은 0~67,999번으로 학습 | `--llava-pretrain-dir`: `blip_laion_cc_sbu_558k.json`과 이미지 (Hugging Face `liuhaotian/LLaVA-Pretrain`) |
| `scienceqa` | train split (이미지 있는 문항) | val split | test split | 자동 다운로드 |
| `textvqa` | `TextVQA_0.5.1_train.json` | 같은 목록의 끝 500개 | `TextVQA_0.5.1_val.json` | `--textvqa-dir`: train json과 `train_images/` |
| `coco_caption` | seed-42 셔플의 100번 이후 | 셔플 끝 500개 | 0~99번 | 자동 다운로드 |

- `--dataset`에 쉼표로 여러 개를 주면 섞어서 학습합니다. 세트마다 `--train-num`개(기본 2000)를 씁니다.
- 권장 구성은 `llava_pretrain,scienceqa,textvqa`입니다. COCO를 빼 두면 COCO 캡션 벤치마크가 "학습에 쓰지 않은 분포"로 남습니다. GQA, MME, MM-Vet, SEED-Bench, VizWiz는 어느 경우에도 학습에 쓰이지 않습니다.
- **검증**: `--val-every`번(기본 10) 업데이트마다 검증 프롬프트(`--val-num`, 세트당 기본 30개)에서 실측 처리량을 재고 ViSpec 고정 설정 대비 배율(`val_speedup_vs_vispec`)을 로그에 남깁니다. 가장 빠른 체크포인트는 `policy_best.pt`로 저장됩니다. 검증 세트는 `--val-dataset`으로 따로 지정할 수 있고, `--val-num 0`이면 끕니다.

## 학습

학습을 시작하면 먼저 보정 단계가 돕니다(`--calib-prompts`, 기본 30개): ViSpec 고정 설정으로 기준 처리량을 재고, 무작위 깊이/후보 수로 비용 모델을 맞춥니다.

```bash
BASE=llava-hf/llava-v1.6-vicuna-7b-hf
SPEC=JLKang/ViSpec-llava-v1.6-vicuna-7b-hf
DATA="--dataset llava_pretrain,scienceqa,textvqa --llava-pretrain-dir /data/LLaVA-Pretrain --textvqa-dir /data/textvqa"

# 1) size 정책 (깊이는 무작위)
python -m vispec.rl.train_ppo --base-model-path $BASE --spec-model-path $SPEC $DATA \
  --train size --depth-mode random --save-dir rl_ckpt/s1 --total-cycles 100000

# 2) depth 정책 (size 정책은 고정)
python -m vispec.rl.train_ppo --base-model-path $BASE --spec-model-path $SPEC $DATA \
  --train depth --resume rl_ckpt/s1/policy_last.pt --save-dir rl_ckpt/s2 --total-cycles 100000

# 3) 두 정책 함께 미세조정
python -m vispec.rl.train_ppo --base-model-path $BASE --spec-model-path $SPEC $DATA \
  --train both --resume rl_ckpt/s2/policy_last.pt --save-dir rl_ckpt/s3 --total-cycles 50000 --lr 1e-4
```

데이터를 아직 받지 않았다면 `--dataset coco_caption`(기본값)이나 `--dataset scienceqa`만으로도 바로 돌려볼 수 있습니다.

주요 옵션

| 옵션 | 용도 |
|---|---|
| `--temperature 1.0` | sampling 조건 학습 |
| `--no-visual` | 시각 특징 제거 ablation |
| `--reward-mode throughput` | 보상 설계 ablation |
| `--time-source model` | 비용 모델 기반 보상 |
| `--max-depth`, `--size-choices` | 행동 범위 |
| `--skip-calib` | `--resume` 체크포인트의 비용 모델과 기준 처리량 재사용 (같은 GPU·모델일 때만) |

로그는 `<save-dir>/train_log.jsonl`에 남습니다. 다른 학습 세트는 `vispec/rl/data.py`에 로더를 추가하면 됩니다.

## 평가

기존 평가 스크립트를 그대로 쓰고 환경변수만 지정합니다.

```bash
VISPEC_RL_POLICY=rl_ckpt/s3/policy_last.pt \
python -m vispec.evaluation.gen_spec_answer_coco_caption --base-model-path $BASE --spec-model-path $SPEC \
  --model-id rl --bench-name results/coco/rl --num-q 2 --depth 3 --top-k 8 --total-token 30 \
  --use-ours=True --temperature 0
```

| 환경변수 | 용도 |
|---|---|
| `VISPEC_RL_POLICY=<ckpt.pt>` | 학습된 정책 사용 (greedy 행동) |
| `VISPEC_RL_DEPTH_MODE=fixed` / `VISPEC_RL_SIZE_MODE=fixed` | size만 / depth만 쓰는 ablation. 고정된 쪽은 `--depth`, `--total-token` 값을 사용 |
| `VISPEC_RL_PROFILE=out.json` | 사이클별 draft/검증 시간, 평균 깊이, 평균 후보 수 기록. 매 사이클 동기화하므로 대표 속도 수치 측정에는 쓰지 않습니다 |
| `VISPEC_RL_POLICY=fixed` + `VISPEC_RL_PROFILE=...` | ViSpec 원본 설정의 시간 분해 |

`--top-k`는 학습 때와 같아야 합니다(기본 8).

## 라이선스

ViSpec 코드는 Apache-2.0(`LICENSE`)입니다. 수정한 파일에는 파일 상단과 `[RL]` 주석으로 변경 사실을 표기했습니다.
