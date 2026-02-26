# Trace Replay Visualizer

프레임 단위로 알고리즘 실행 과정을 재생하는 기능입니다.

## 목적

- 발표/리뷰 시 알고리즘 내부 상태 변화를 단계별로 설명
- 단순 결과값이 아니라, 탐색/최적화의 **의사결정 과정**을 시각화

## 구성 요소

- `visualization/trace_generators.py`
  - `DijkstraTraceGenerator`: 그래프 탐색 과정을 프레임으로 기록
  - `BellmanFordTraceGenerator`: 음수 가중치 안전 그래프 탐색 과정을 기록
  - `KBestTraceGenerator`: 후보 경로 생성/랭킹/재평가 과정을 기록
  - `GreedySplitTraceGenerator`: 청크 단위 할당 과정을 기록
  - `ConvexSplitTraceGenerator`: gradient 최적화 과정을 기록
  - `KKTTraceGenerator`: 분할 최적화(이분탐색) 과정을 프레임으로 기록
- `visualization/generate_traces.py`
  - 샘플 트레이스 JSON 생성 + GIF/MP4 자동 export
- `visualization/export_media.py`
  - trace JSON을 발표용 GIF/MP4로 렌더링
- `visualization/replay_viewer.html`
  - 웹 플레이어 (Prev/Next/Play/Pause/속도/슬라이더)
  - 레이아웃 선택(`Flow`, `Force`, `Circular`)
  - 엣지 표시 모드(`Active Only`, `Best Path Only`, `All`)
  - 프레임 간 diff 하이라이트(바뀐 distance/queue 항목만)

## 사용 방법

```bash
cd routing-algorithms

# 1) 트레이스 생성
python3 visualization/generate_traces.py --token-in ETH --token-out USDC --amount 250

# 2) 웹 서버 실행
python3 -m http.server 8000
```

브라우저 접속:

- `http://localhost:8000/visualization/replay_viewer.html`

옵션:

- `--skip-media`: GIF/MP4 export를 건너뛰고 JSON만 생성
- `--media-fps 3`: media fps 조정
- `--media-layout flow`: export 그래프 레이아웃 선택 (`flow`/`circular`)
- `--bellman-frames 420`: Bellman-Ford trace 최대 프레임 수 조정

## 출력 파일

- `output/traces/dijkstra_trace.json`
- `output/traces/bellman_ford_trace.json`
- `output/traces/kbest_trace.json`
- `output/traces/greedy_split_trace.json`
- `output/traces/convex_split_trace.json`
- `output/traces/kkt_split_trace.json`
- `output/traces/index.json`
- `output/media/*.gif`
- `output/media/*.mp4` (ffmpeg 환경일 때)

## 프레임 데이터 요약

- 공통
  - `step`, `event`, `description`
- Dijkstra/Bellman-Ford
  - `current_node`, `visited`, `distances`, `priority_queue`, `edge_highlight`
- K-Best
  - `candidate_path`, `spot_score`, `candidate_amount_out`, `best_path_so_far`
- Greedy/Convex/KKT
  - `per_pool_results`, `allocations`, `total_amount_out`
- KKT
  - `lambda_low`, `lambda_high`, `lambda_mid`, `allocation_sum_at_mid`
  - `per_pool_results`, `final_allocations`
