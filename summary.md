# Two-Stage Sparse von Mises-Fisher Mixture Models for Clustering L2-Normalized Text Embeddings

**연구 미팅 자료 (최종)**

---

## 1. 핵심 아이디어

L2 정규화된 텍스트 임베딩을 단위 초구면 위의 방향성 자료로 보고, $L_1$-penalized vMF mixture로 군집 판별 좌표를 선택한 뒤, **원래 sphere $S^{d-1}$ 위의 sparse-vMF submodel**에서 unpenalized refit을 수행한다.

### 1.1 파이프라인

$$d_i \xrightarrow{\phi(\cdot)} \mathbf{x}_i \in \mathbb{R}^d \xrightarrow{L_2\text{-norm}} \mathbf{z}_i = \frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_2} \in S^{d-1}$$

$$\text{Stage 1: } \hat{S}_\lambda \leftarrow L_1\text{-penalized vMF mixture}$$

$$\text{Stage 2: } \hat{\Theta}_{\hat{S}}^{\text{refit}} \leftarrow \text{unpenalized sparse-vMF submodel on original } S^{d-1}$$

### 1.2 Novelty

- "Sparse vMF mixture 자체"가 아님 (Rossi & Barbaro 등 선행 존재)
- **Meynet식 Lasso-MLE 원칙을 sparse directional mixture로 확장**
- **원래 sphere 위의 sparse-vMF submodel refit 정식화** — 모든 $(K, \lambda)$ 모델이 같은 sample space 위에서 정의되어 BIC/ICL을 통한 nested likelihood 비교 가능

---

## 2. vMF 혼합모형

### 2.1 vMF density

$$f(\mathbf{z} \mid \mu, \kappa) = c_d(\kappa) \exp(\kappa \mu^\top \mathbf{z}), \quad \mathbf{z}, \mu \in S^{d-1}, \; \kappa \geq 0$$

$$c_d(\kappa) = \frac{\kappa^{d/2-1}}{(2\pi)^{d/2} I_{d/2-1}(\kappa)}$$

### 2.2 Mixture

$$p(\mathbf{z}_i \mid \Theta) = \sum_{h=1}^K \pi_h c_d(\kappa_h) \exp(\kappa_h \mu_h^\top \mathbf{z}_i)$$

**제약**:
$$\sum_{h=1}^K \pi_h = 1, \quad \|\mu_h\|_2 = 1, \quad 0 \leq \kappa_h \leq \kappa_{\max}$$

### 2.3 Natural parameter

$$\eta_h = \kappa_h \mu_h, \quad f(\mathbf{z}_i \mid \eta_h) = c_d(\|\eta_h\|_2) \exp(\eta_h^\top \mathbf{z}_i)$$

$$p(\mathbf{z}_i \mid \pi, \eta) = \sum_{h=1}^K \pi_h c_d(\|\eta_h\|_2) \exp(\eta_h^\top \mathbf{z}_i)$$

---

## 3. Stage 1: Penalized vMF Mixture

### 3.1 Option B (Main): Natural Parameter Penalty

**가중 평균** (algorithm 안정성을 위해 fixed weight로 처리):

$$\bar{\eta}_j = \sum_{h=1}^K w_h \eta_{hj}, \quad w_h = \pi_h^{(t)} \text{ 또는 } N_h^{(t)}/n$$

**Penalty**:

$$P_B(\eta) = \sum_{j=1}^d \left[\sum_{h=1}^K w_h (\eta_{hj} - \bar{\eta}_j)^2\right]^{1/2}$$

**Penalized log-likelihood**:

$$\mathcal{L}_{\lambda_n}^B(\pi, \eta) = \ell_n(\pi, \eta) - n \lambda_n P_B(\eta)$$

$$\ell_n(\pi, \eta) = \sum_{i=1}^n \log \left[\sum_{h=1}^K \pi_h c_d(\|\eta_h\|_2) \exp(\eta_h^\top \mathbf{z}_i)\right]$$

### 3.2 Option A (Ablation): Mean Direction Penalty

$$\bar{\mu}_j = \sum_{h=1}^K w_h \mu_{hj}$$

$$P_A(\mu) = \sum_{j=1}^d \left[\sum_{h=1}^K w_h (\mu_{hj} - \bar{\mu}_j)^2\right]^{1/2}$$

### 3.3 두 옵션의 통계적 의미 차이

vMF log-density의 선형 판별항은:

$$\eta_h^\top \mathbf{z}_i = \kappa_h \mu_h^\top \mathbf{z}_i$$

- **Option B**: $\eta_{hj} = \kappa_h \mu_{hj}$가 component별로 같은 차원은 판별에 기여하지 않음 → $\kappa_h$ 차이 자동 반영
- **Option A**: $\mu_{1j} = \cdots = \mu_{Kj}$여도 $\kappa_h \mu_{hj}$가 다르면 판별 정보가 남을 수 있어 부정확할 수 있음

→ **Option B를 main으로**, Option A는 ablation.

### 3.4 E-step (Option B 기준)

$$\tau_{ih} = \frac{\pi_h c_d(\|\eta_h\|_2) \exp(\eta_h^\top \mathbf{z}_i)}{\sum_{\ell=1}^K \pi_\ell c_d(\|\eta_\ell\|_2) \exp(\eta_\ell^\top \mathbf{z}_i)}, \quad N_h = \sum_{i=1}^n \tau_{ih}$$

### 3.5 M-step

> Cluster-contrast group penalty와 mixture likelihood 결합으로 **closed-form 부재**.
> ECM / MM / proximal-gradient 기반 **generalized EM**으로 구현.

**Option B의 이점**: $\eta_h \in \mathbb{R}^d$가 단위 노름 제약 없음 → manifold optimization 불필요. $\|\eta_h\|_2 \leq \kappa_{\max}$ 제약만 유지.

### 3.6 Degeneracy Safeguard

$$\|\eta_h\|_2 \leq \kappa_{\max}, \quad N_h \geq N_{\min}$$

$\|\eta_h\|_2 < \epsilon$인 component는 uniform/degenerate component로 처리.

### 3.7 Active Set

**Option B (main)**:
$$\hat{S}_\lambda = \left\{j : \left[\sum_{h=1}^K w_h (\hat{\eta}_{hj} - \hat{\bar{\eta}}_j)^2\right]^{1/2} > \epsilon\right\}$$

**Option A (ablation)**:
$$\hat{S}_\lambda^A = \left\{j : \left[\sum_{h=1}^K w_h (\hat{\mu}_{hj} - \hat{\bar{\mu}}_j)^2\right]^{1/2} > \epsilon\right\}$$

---

## 4. Stage 2: Sparse-vMF Submodel Refit on Original Sphere

### 4.1 Main: Sparse-Submodel Refit

**핵심**: 데이터 $\mathbf{z}_i$는 $S^{d-1}$ 위에 그대로 둠. 비활성 좌표에서 mean direction을 0으로 강제.

$$\mu_{h, \hat{S}^c} = \mathbf{0}, \quad \|\mu_{h, \hat{S}}\|_2 = 1$$

$\mu_h$는 부분구면 $\{\mu \in S^{d-1} : \mu_{\hat{S}^c} = \mathbf{0}\}$ 위에 있음 ($S^{d_\lambda - 1}$과 isometric, $d_\lambda = |\hat{S}_\lambda|$).

**Density는 원래 sphere의 정규화 상수** $c_d(\kappa_h)$를 사용 (NOT $c_{d_\lambda}$):

$$p(\mathbf{z}_i \mid \tilde{\Theta}_{\hat{S}}) = \sum_{h=1}^K \tilde{\pi}_h c_d(\tilde{\kappa}_h) \exp\left(\tilde{\kappa}_h \tilde{\mu}_{h,\hat{S}}^\top \mathbf{z}_{i,\hat{S}}\right)$$

### 4.2 Refit EM

**E-step**:
$$\tilde{\tau}_{ih} = \frac{\tilde{\pi}_h c_d(\tilde{\kappa}_h) \exp(\tilde{\kappa}_h \tilde{\mu}_{h,\hat{S}}^\top \mathbf{z}_{i,\hat{S}})}{\sum_{\ell=1}^K \tilde{\pi}_\ell c_d(\tilde{\kappa}_\ell) \exp(\tilde{\kappa}_\ell \tilde{\mu}_{\ell,\hat{S}}^\top \mathbf{z}_{i,\hat{S}})}$$

**Resultant**:
$$\mathbf{r}_{h,\hat{S}} = \sum_{i=1}^n \tilde{\tau}_{ih} \mathbf{z}_{i,\hat{S}}, \quad N_h = \sum_{i=1}^n \tilde{\tau}_{ih}$$

**Mean direction**:
$$\hat{\tilde{\mu}}_{h,\hat{S}} = \frac{\mathbf{r}_{h,\hat{S}}}{\|\mathbf{r}_{h,\hat{S}}\|_2}, \quad \hat{\tilde{\mu}}_{h,\hat{S}^c} = \mathbf{0}$$

**Resultant length**:
$$\bar{R}_{h,\hat{S}} = \frac{\|\mathbf{r}_{h,\hat{S}}\|_2}{N_h}$$

**Concentration update** (dimension은 $d_\lambda$가 아닌 원래 $d$):
$$A_d(\hat{\tilde{\kappa}}_h) = \bar{R}_{h,\hat{S}}, \quad \hat{\tilde{\kappa}}_h \approx \frac{d \bar{R}_{h,\hat{S}} - \bar{R}_{h,\hat{S}}^3}{1 - \bar{R}_{h,\hat{S}}^2}$$

**Mixing weight**:
$$\hat{\tilde{\pi}}_h = \frac{N_h}{n}$$

### 4.3 Variant: Selected-Sphere Refit (Practical Only)

$$\tilde{\mathbf{z}}_i = \frac{P_{\hat{S}} \mathbf{z}_i}{\|P_{\hat{S}} \mathbf{z}_i\|_2} \in S^{d_\lambda - 1}$$

**Admissibility**: $\min_i \|P_{\hat{S}} \mathbf{z}_i\|_2 > \epsilon_0$ 위배 시 해당 $\lambda$ 제외.

이 variant는 $c_{d_\lambda}(\tilde{\kappa}_h)$ 사용. $\lambda$마다 sample space 변경 → BIC 비교는 heuristic.

→ **Sensitivity analysis로만 제시**, model selection은 main version으로.

---

## 5. Model Selection: BIC

### 5.1 자유도 (Main sparse-submodel 기준)

$\mu_h$의 자유도는 $d_\lambda - 1$ (부분구면 $S^{d_\lambda - 1}$ 위):

**Component-specific $\kappa_h$**:
$$\nu(K, \lambda) = \underbrace{(K-1)}_{\pi} + \underbrace{K(d_\lambda - 1)}_{\mu_h} + \underbrace{K}_{\kappa_h} = K d_\lambda + K - 1$$

**Common $\kappa$**:
$$\nu_{\text{common}}(K, \lambda) = (K-1) + K(d_\lambda - 1) + 1 = K d_\lambda$$

### 5.2 BIC

$$\text{BIC}(K, \lambda) = -2 \ell_n(\hat{\tilde{\Theta}}_{K,\lambda}^{\text{submodel-refit}}) + \nu(K, \lambda) \log n$$

**Refit log-likelihood** (원래 sphere 위에서):
$$\ell_n(\hat{\tilde{\Theta}}) = \sum_{i=1}^n \log \left[\sum_{h=1}^K \hat{\tilde{\pi}}_h c_d(\hat{\tilde{\kappa}}_h) \exp\left(\hat{\tilde{\kappa}}_h \hat{\tilde{\mu}}_{h,\hat{S}}^\top \mathbf{z}_{i,\hat{S}}\right)\right]$$

### 5.3 톤 조정

- BIC/ICL은 **practical model selection criterion**
- 고차원 random model collection에 대한 이론적 정당화 (slope heuristic 등)는 future work
- Main version은 모든 $\lambda$ 모델이 $S^{d-1}$ 위에서 정의되므로 **nested likelihood 비교가 정당화됨** (selected-sphere variant 대비 강점)

---

## 6. Stability Selection (Meinshausen–Bühlmann)

Subsampling 또는 bootstrap 반복 $b = 1, \ldots, B$:

$$\hat{\Pi}_j = \frac{1}{B} \sum_{b=1}^B \mathbf{1}\{j \in \hat{S}_\lambda^{(b)}\}$$

**Stable support**:
$$\hat{S}_{\text{stable}} = \{j : \hat{\Pi}_j \geq \pi_{\text{thr}}\}, \quad \pi_{\text{thr}} \in [0.6, 0.9]$$

Meinshausen–Bühlmann의 framework는 finite-sample false selection control도 제공 → 단순 heuristic 이상의 정당성.

---

## 7. 해석가능성

| 도구 | 내용 | 강도 |
|---|---|---|
| $\hat{\kappa}_h$ | cluster thematic tightness | **강함** |
| Cluster 대표 문서 | $\hat{\tilde{\mu}}_h$에 nearest top-k 문서 | **강함** |
| Cluster summary | 대표 문서로부터 키워드/요약 | 중간 |
| Active set $\hat{S}_\lambda$ | dimension screening 결과 | **해석 불가** |

### 7.1 명시 문장

> 선택된 dense embedding 좌표 자체는 의미적으로 해석되지 않으며, **regularization 및 dimension screening 결과**일 뿐이다. 군집 해석은 $\kappa_h$, 대표 문서, cluster summary로 제공한다.

### 7.2 편향에 대한 표현

- "refit은 LASSO penalty에 의한 shrinkage bias를 완화한다"
- "refit은 unbiased estimator를 제공한다"

(support가 random이고 selection error가 있어 전체 unbiased는 주장 불가)

---

## 8. 통계적 기여

### 8.1 강하게 주장 (첫 논문 핵심)

1. **Meynet식 Lasso-MLE 원칙을 sparse directional mixture로 확장**
2. **원래 sphere 위의 sparse-vMF submodel refit 정식화** — $(K, \lambda)$ 모델이 공통 sample space에서 정의되어 nested likelihood 비교 가능
3. **Cluster-contrast penalty의 두 형태 비교** ($\mu$ vs. $\eta$); Option B가 component별 $\kappa_h$ 차이를 반영하므로 더 자연스러움
4. **Stability selection + BIC/ICL 결합 model selection 절차**

### 8.2 Future Target

- Penalized EM monotone ascent
- Selection consistency (vMF Bessel 함수, label switching, unit norm 제약)
- Random model collection 하의 BIC 이론적 정당화 (slope heuristic)

---

## 9. 실험 계획

### 9.1 시뮬레이션

- Active set $S^*$ 회복률 (precision, recall, $F_1$)
- 군집화 정확도 (ARI, NMI)
- $\kappa_h$ 추정 정확도
- 변동 요인: $(d, n, K, |S^*|, \kappa^*)$, cluster 분리도

### 9.2 실데이터

- 텍스트 임베딩 2–3종 (Sentence-BERT, E5)
- 다양한 도메인 (뉴스, 리뷰, 학술 abstract 등)

### 9.3 핵심 검증 — **네 가지 모두 나와야 논문 성립**

1. **LASSO-vMF + sparse-submodel refit > LASSO-vMF** (refit 효과)
2. **LASSO-vMF + sparse-submodel refit > LASSO-GMM + refit** (vMF geometry 효과)
3. **LASSO-vMF + sparse-submodel refit > dense vMF + threshold + refit** (LASSO path의 필요성)
4. **$\hat{S}_\lambda$가 bootstrap/subsampling에서 안정적** ($\hat{\Pi}_j$ 분포로 검증)

### 9.4 전체 Baseline

- spherical k-means
- movMF (Banerjee, dense)
- sparse k-means (Witten–Tibshirani)
- Lasso-GMM + refit (Meynet)
- dense vMF + threshold + refit
- 기존 sparse vMF mixture (Rossi & Barbaro 등)
- BERTopic
- (옵션) DEC

### 9.5 Ablation

- Penalty 형태: Option A ($\mu$) vs. Option B ($\eta$)
- Refit 형태: sparse-submodel (main) vs. selected-sphere (variant)
- Refit 유무
- $\lambda$ 선택: BIC vs. ICL vs. CV
- Stability selection 임계값 $\pi_{\text{thr}}$
- Penalty weight $w_h$ 선택 ($\pi_h^{(t)}$ vs. uniform vs. $N_h^{(t)}/n$)

---

## 10. 방법론 위치

| Method | Probabilistic model | Directional likelihood | Variable selection | $\kappa_h$ estimation | Two-stage refit |
|---|:---:|:---:|:---:|:---:|:---:|
| spherical k-means | ✗ | ✗ | ✗ | ✗ | – |
| movMF (Banerjee) | ✓ | ✓ | ✗ | ✓ | – |
| sparse k-means (W–T) | ✗ | ✗ | ✓ | ✗ | ✗ |
| Lasso-GMM + refit (Meynet) | ✓ | ✗ | ✓ | ✗ | ✓ |
| Sparse vMF (existing) | ✓ | ✓ | ✓ | ✓ | ✗ |
| **Proposed** | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 11. 미해결 / 미팅 논의 항목

1. **Option A vs. Option B 최종 결정** — Option B 권장
2. **M-step의 정확한 형태** — ECM / MM / proximal-gradient 중 선택
3. **Penalty weight $w_h$ 처리** — $\pi_h^{(t)}$ 고정 vs. uniform vs. $N_h/n$
4. **Sparse-submodel refit EM의 정확한 derivation** — 부분구면 위에서의 update
5. **$\lambda_n$ scaling 최종 결정** — $\ell_n - n\lambda_n P$ vs. $\frac{1}{n}\ell_n - \lambda_n P$
6. **Hyperparameter 설정** — $\kappa_{\max}, N_{\min}, \epsilon_0, \pi_{\text{thr}}$
7. **$K$ 결정** — BIC 그리드 vs. nested model selection

---

## 12. 핵심 기여 문장

### 12.1 한국어

> 본 연구는 L2 정규화된 텍스트 임베딩을 초구면 위의 방향성 자료로 보고, vMF 혼합모형의 자연모수 $\eta_h = \kappa_h \mu_h$에 cluster-contrast sparsity penalty를 결합하여 군집 판별에 필요한 좌표 집합을 screening한다. 이후 **원래 sphere $S^{d-1}$ 위에서 비활성 좌표의 mean-direction 성분을 0으로 강제한 sparse-vMF submodel**로 unpenalized refit을 수행하여 LASSO shrinkage에 따른 방향 추정 편향을 완화한다. 이 정식화는 모든 $\lambda$ 모델이 공통 sample space 위에 정의되도록 하여 BIC/ICL을 통한 nested likelihood 비교를 가능하게 한다. 선택된 좌표 자체는 의미적으로 해석하지 않으며, 군집 해석은 $\kappa_h$, 대표 문서, cluster summary로 제공한다.

### 12.2 영문 (Contribution Statement)

> We extend the Lasso-MLE principle of Meynet (2013) to sparse directional mixtures by combining an $L_1$-penalized vMF mixture (with cluster-contrast penalty on the natural parameter $\eta_h = \kappa_h \mu_h$) for support selection with an **unpenalized sparse-vMF submodel refit on the original sphere $S^{d-1}$**, where mean-direction components on the inactive coordinates are constrained to zero. This preserves a common sample space across regularization paths, enabling principled BIC/ICL-based model selection.

---

## 13. 다음 단계

1. **Sparse-submodel refit EM의 정확한 M-step derivation** (부분구면 위에서)
2. **Stage 1 Option B의 generalized EM 구현 방식 결정** (ECM/MM/proximal)
3. **Toy simulation** ($d = 50, n = 500, K = 3, |S^*| = 10$) — 작동 검증부터
4. **실데이터 적용 전 pilot study**

---

## 발표 시 강조할 세 가지

1. **Main refit은 selected-sphere가 아니라 original sphere $S^{d-1}$ 위의 sparse-submodel refit**
2. **Option B (natural parameter penalty)를 main으로** — $\eta_h = \kappa_h \mu_h$가 log-density의 실제 선형 판별항
3. **Active set은 semantic interpretation이 아님** — 해석은 $\kappa_h$, 대표 문서, cluster summary로 제공
