# Market Making under Regime-Switching Heston
## 수정된 코드 (논문 충실 버전)

### 주요 수정사항

## Step 1: Data Preprocessing

### 기존 문제점:
```python
variance_5min['Variance'] *= 78 * 252  # Annualize
```
- 연율화가 Heston 모델의 instantaneous variance V_t와 맞지 않을 수 있음

### 수정:
- Raw realized variance 유지 (HMM이 상대적 레벨로 regime 구분)
- 필요시 별도로 annualization 옵션 제공

---

## Step 2: Regime Identification

### 기존 문제점:
```python
dt = 5 / (60 * 24)  # 24시간 기준 → 잘못됨!
```

### 수정:
```python
TRADING_HOURS = 6.5
INTERVALS_PER_DAY = int(TRADING_HOURS * 60 / INTERVAL_MINUTES)  # 78
dt = 1.0 / INTERVALS_PER_DAY  # ≈ 0.0128 trading days per interval
```

### CIR 파라미터 추정 개선:
- Bounds 확장 (데이터 기반 동적 설정)
- Multiple starting points로 robustness 향상
- Feller condition 명시적 확인

### Transition rates 계산:
```python
# 논문 Eq. 1: Q = [[-λ_HL, λ_HL], [λ_LH, -λ_LH]]
# P_ii(dt) ≈ exp(-λ_ij * dt)
lambda_LH = -np.log(trans_mat[0, 0]) / dt  # Low → High
lambda_HL = -np.log(trans_mat[1, 1]) / dt  # High → Low
```

---

## Step 3: Intensity Estimation (가장 중요!)

### 기존 문제점:
```python
# 기존: spread의 histogram을 fitting → 개념적 오류
bins = np.linspace(spread_vec.min(), np.quantile(spread_vec, 0.95), 11)
counts, bin_edges = np.histogram(spread_vec, bins=bins)
```

### 수정 (논문 Eq. 13-14에 충실):
```python
# 논문: Λ^a_i(δ) = A^a_i * exp(-η^a_i * δ)
# δ = half-spread at execution

# Buy trade (hit ask): δ = trade_price - mid_price
# Sell trade (hit bid): δ = mid_price - trade_price
trades['Delta'] = np.where(
    trades['Side'] == 'buy',
    trades['Price'] - trades['Mid'],
    trades['Mid'] - trades['Price']
)
```

### Intensity 추정 방법:
1. δ 구간별로 거래 수 집계
2. Bin width로 정규화하여 intensity 계산
3. Exponential decay 함수 fitting

---

## 사용법

### 폴더 구조:
```
project/
├── data/
│   ├── MSFT_quotes_combined.csv
│   └── MSFT_trades_combined.csv
├── scripts/
│   ├── step1_preprocessing.py
│   ├── step2_regime_identification.py
│   └── step3_intensity_estimation.py
└── output/
    ├── csv/
    ├── plots/
    └── parameters/
```

### 실행 순서:
```bash
cd scripts
python step1_preprocessing.py
python step2_regime_identification.py
python step3_intensity_estimation.py
```

---

## 예상 출력 (정상적인 경우)

### Step 2 (Regime Identification):
```
Transition rates (per trading day):
  λ_LH (Low→High): 0.5 ~ 5.0
  λ_HL (High→Low): 1.0 ~ 10.0

Expected regime duration:
  Low regime:  0.2 ~ 2.0 days (30 min ~ 13 hours)
  High regime: 0.1 ~ 1.0 days (40 min ~ 6.5 hours)

Heston parameters:
  κ (mean-reversion): 1 ~ 20
  θ (long-run var): 데이터 variance의 평균 근처
  ξ (vol-of-vol): 0.1 ~ 2.0
  Feller ratio (2κθ/ξ²) > 1 ← 중요!
```

### Step 3 (Intensity):
```
Low regime - buy:  A = 1000~10000, η = 10~100
Low regime - sell: A = 1000~10000, η = 10~100
High regime - buy: A = 2000~20000, η = 5~50
High regime - sell: A = 2000~20000, η = 5~50
```

High regime에서 A가 더 크고 (더 많은 거래), η가 더 작으면 (가격 민감도 낮음) 합리적임.

---

## 논문 수식 참조

### Definition 4.1 (Regime Process):
$$Q = \begin{pmatrix} -\lambda_{HL} & \lambda_{HL} \\ \lambda_{LH} & -\lambda_{LH} \end{pmatrix}$$

### Eq. 6 (Heston Dynamics):
$$dV_t = \kappa_{X_t}(\theta_{X_t} - V_t)dt + \xi\sqrt{V_t}dW^V_t$$

### Eq. 13-14 (Intensity):
$$\Lambda^a_i(\delta) = A^a_i \exp(-\eta^a_i \delta)$$
$$\Lambda^b_i(\delta) = A^b_i \exp(-\eta^b_i \delta)$$

### Assumption 4.7 (Feller Condition):
$$2\kappa_i\theta_i > \xi^2$$