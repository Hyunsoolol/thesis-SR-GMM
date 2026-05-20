# Sparse vMF Mixture with Two-Stage Refit: 연구미팅 정리

> **요약**: 본 연구는 L2 정규화된 텍스트 임베딩의 군집화를 위해, vMF mixture의 natural parameter $\eta_h = \kappa_h \mu_h$에 **cluster-contrast sparsity penalty**를 부여하여 판별 좌표를 선택하고, 선택된 support 위에서 **unpenalized sparse-vMF submodel refit**을 수행하는 2단계 절차를 제안한다.

---

## 1. 배경

### 1.1 vMF Mixture Model

고차원 텍스트 임베딩은 L2 정규화 후 단위 초구면 $\mathbb{S}^{d-1}$ 위의 방향성 자료로 볼 수 있다. 이때 Euclidean distance보다 cosine similarity가 자연스러우며, 이를 확률모형으로 정식화한 것이 **von Mises–Fisher (vMF) mixture model**이다. Banerjee et al. 은 vMF mixture가 spherical $k$-means의 확률모형적 일반화임을 보였다.

vMF density:

$$ f(z \mid \mu, \kappa) = c_d(\kappa),\exp(\kappa,\mu^\top z), \qquad z, \mu \in \mathbb{S}^{d-1},\ \kappa \geq 0. $$

Mixture model:

$$ p(z_i \mid \Theta) = \sum_{h=1}^{K} \pi_h, c_d(\kappa_h), \exp(\kappa_h, \mu_h^\top z_i). $$

### 1.2 선행연구: Rossi & Barbaro (2022)

Rossi & Barbaro는 기존 vMF mixture에 $L_1$ penalty를 부여해 **sparse prototype**을 만드는 방법을 제안하였다. 목적은 cluster를 설명하는 핵심 단어 좌표를 남겨 해석력을 높이는 것이다.

목적함수:

$$ \mathcal{L}_p(\Theta \mid X) = \mathcal{L}(\Theta \mid X) - \beta \sum_{h=1}^{K} |\mu_h|_1. $$

기여: $L_1$-penalized likelihood, modified EM, path-following, AIC/BIC/EBIC/RIC 기반 model selection.

**한계점**:

- Penalized estimator를 **최종 결과**로 사용하므로 $L_1$ shrinkage bias가 남는다.
- Penalty가 $\mu_h$에 직접 걸려, 실제 vMF log-density의 판별항인 $\kappa_h \mu_h$ 구조를 충분히 반영하지 못한다.

---

## 2. 연구 핵심 아이디어

L2 정규화된 텍스트 임베딩을 $\mathbb{S}^{d-1}$ 위의 방향성 자료로 보고:

1. **Stage 1**: $L_1$-penalized vMF mixture로 군집 판별 좌표 선택
2. **Stage 2**: 선택된 좌표에서 unpenalized sparse-vMF submodel refit

즉, sparse vMF 자체가 최종 모형이 아니라 **support selection + refit** 의 2단계 구조가 핵심이다.

### Pipeline

$$ d_i ;\xrightarrow{\phi(\cdot)}; x_i \in \mathbb{R}^d ;\xrightarrow{L_2\text{-norm}}; z_i = \frac{x_i}{|x_i|_2} \in \mathbb{S}^{d-1} $$

$$ \text{Stage 1: } \widehat{S}_\lambda \leftarrow L_1\text{-penalized vMF mixture} $$

$$ \text{Stage 2: } \widehat{\Theta}^{\text{refit}}_{\widehat{S}} \leftarrow \text{unpenalized sparse-vMF submodel on } \mathbb{S}^{d-1} $$

---

## 3. 선행연구와의 차별점

| 구분                  | Rossi & Barbaro (2022)      | 본 연구                                                           |
| ------------------- | --------------------------- | -------------------------------------------------------------- |
| **핵심 목표**           | sparse prototype 생성         | 군집 판별 좌표 선택 후 refit                                            |
| **Penalty 대상**      | $\mu_h$ 자체                  | natural parameter $\eta_h = \kappa_h \mu_h$ 의 cluster contrast |
| **Sparsity 의미**     | cluster별 prototype sparsity | cluster 간 차이를 만드는 active set                                   |
| **최종 추정량**          | penalized estimator         | unpenalized refit estimator                                    |
| **Bias 처리**         | $L_1$ shrinkage bias 잔존     | refit으로 shrinkage bias 완화                                      |
| **해석 방식**           | sparse word coordinate 해석   | 대표 문서·summary 기반 해석                                            |
| **Model selection** | path-following + IC         | refit likelihood 기반 BIC/ICL                                    |

가장 중요한 차이점 세 가지:

1. **Penalty 대상**: 기존은 $\mu_h$를 sparse하게 만들지만, 본 연구는 실제 log-density의 선형 판별항인 $\eta_h = \kappa_h \mu_h$ 에 penalty를 둔다.
2. **추정 절차**: 기존은 penalized sparse vMF를 최종 모형으로 쓰지만, 본 연구는 이를 **screening 단계**로만 사용하고 선택된 support에서 penalty 없이 다시 추정한다.
3. **모형 정의 공간**: 본 연구의 refit은 selected coordinates만 정규화한 sphere가 아니라, **원래 $\mathbb{S}^{d-1}$ 위에서** 비활성 좌표의 mean direction을 0으로 강제하는 sparse submodel이다.

---

## 4. 모형 및 수식

### 4.1 Reparametrization

Natural parameter:

$$ \eta_h = \kappa_h, \mu_h $$

로 두면 vMF mixture는

$$ p(z_i \mid \pi, \eta) = \sum_{h=1}^{K} \pi_h, c_d(|\eta_h|_2), \exp(\eta_h^\top z_i), $$

제약: $\sum_h \pi_h = 1,\ |\mu_h|_2 = 1,\ 0 \leq \kappa_h \leq \kappa_{\max}$.

### 4.2 Stage 1 — Cluster-Contrast Penalty

좌표 $j$에 대한 weighted mean:

$$ \bar{\eta}_j = \sum_{h=1}^{K} w_h, \eta_{hj}. $$

Cluster-contrast penalty:

$$ P_B(\eta) = \sum_{j=1}^{d} \left[ \sum_{h=1}^{K} w_h,(\eta_{hj} - \bar{\eta}_j)^2 \right]^{1/2}. $$

Penalized log-likelihood:

$$ \mathcal{L}^{B}_{\lambda_n}(\pi, \eta) = \ell_n(\pi, \eta) - n,\lambda_n, P_B(\eta). $$

이 penalty는 좌표 $j$에서 cluster들이 서로 다를 때만 해당 좌표를 active로 남긴다. Active set:

$$ \widehat{S}_\lambda = \{ j : \{ \sum_{h=1}^{K} w_h,(\widehat{\eta}_{hj} - \widehat{\bar{\eta}}_j)^2 \}^{1/2} > \epsilon \}. $$

### 4.3 Stage 2 — Sparse-vMF Submodel Refit

선택된 support $\widehat{S}$ 에서 비활성 좌표를 0으로 강제:

$$ \mu_{h,\widehat{S}^c} = 0, \qquad |\mu_{h,\widehat{S}}|_2 = 1. $$

> **핵심 포인트**: density를 selected sphere가 아니라 **원래 sphere $\mathbb{S}^{d-1}$** 위에서 정의한다. 따라서 정규화 상수는 $c_{d_\lambda}$ 가 아니라 $c_d$ 를 사용한다.

$$ p(z_i \mid \widetilde{\Theta}_{\widehat{S}}) = \sum_{h=1}^{K} \widetilde{\pi}_h, c_d(\widetilde{\kappa}_h), \exp!\left( \widetilde{\kappa}_h, \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}} \right). $$

#### Refit EM Updates

**E-step (responsibility)**:

$$ \widetilde{\tau}_{ih} = \frac{\widetilde{\pi}_h, c_d(\widetilde{\kappa}_h), \exp!\left( \widetilde{\kappa}_h, \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}} \right)}{\sum_{\ell=1}^{K} \widetilde{\pi}_\ell, c_d(\widetilde{\kappa}_\ell), \exp!\left( \widetilde{\kappa}_\ell, \widetilde{\mu}_{\ell,\widehat{S}}^\top z_{i,\widehat{S}} \right)}. $$

**M-step (sufficient statistics)**:

$$ r_{h,\widehat{S}} = \sum_{i=1}^{n} \widetilde{\tau}_{ih}, z_{i,\widehat{S}}, \qquad N_h = \sum_{i=1}^{n} \widetilde{\tau}_{ih}. $$

**Mean direction update**:

$$ \widehat{\widetilde{\mu}}_{h,\widehat{S}} = \frac{r_{h,\widehat{S}}}{|r_{h,\widehat{S}}|_2}, \qquad \widehat{\widetilde{\mu}}_{h,\widehat{S}^c} = 0. $$

**Concentration update**:

$$ A_d(\widehat{\widetilde{\kappa}}_h) = \frac{|r_{h,\widehat{S}}|_2}{N_h}. $$

근사식 (Banerjee approximation):

$$ \widehat{\widetilde{\kappa}}_h \approx \frac{\bar{R}_{h,\widehat{S}},d - \bar{R}_{h,\widehat{S}}^3}{1 - \bar{R}_{h,\widehat{S}}^2}, \qquad \bar{R}_{h,\widehat{S}} = \frac{|r_{h,\widehat{S}}|_2}{N_h}. $$

---

## 5. Model Selection

모든 $(K, \lambda)$ 후보 모형이 **같은 sample space $\mathbb{S}^{d-1}$** 위에서 정의되므로 refit log-likelihood 기반 BIC 비교가 가능하다.

**자유도** (component-specific $\kappa_h$):

$$ \nu(K, \lambda) = (K-1) + K(d_\lambda - 1) + K = K d_\lambda + K - 1. $$

**자유도** (common $\kappa$):

$$ \nu_{\text{common}}(K, \lambda) = K d_\lambda. $$

**BIC**:

$$ \text{BIC}(K, \lambda) = -2,\ell_n(\widehat{\widetilde{\Theta}}^{\text{refit}}_{K,\lambda}) + \nu(K,\lambda),\log n. $$

> BIC/ICL은 이론적으로 완결된 oracle criterion이라기보다 현 단계에서는 **practical model selection criterion** 으로 제시하는 것이 안전하다. (이론적 정당화는 후속 과제)

---

## 6. 연구 기여 요약

본 연구의 기여는 "sparse vMF mixture를 새로 제안한다"가 아니다. 그것은 이미 Rossi & Barbaro의 선행연구가 있다. 본 연구의 기여는 다음과 같다.

1. vMF mixture의 natural parameter $\eta_h = \kappa_h \mu_h$ 에 **cluster-contrast sparsity penalty**를 도입한다.
2. Penalized estimator를 최종 모형으로 쓰지 않고, **support selection 후 unpenalized sparse-vMF submodel refit**을 수행한다.
3. Refit을 **원래 sphere $\mathbb{S}^{d-1}$ 위에서 정식화**하여, 서로 다른 $\lambda$ 모델이 같은 sample space에서 비교되도록 한다.
4. LLM/text embedding 맥락에서는 active coordinate 자체를 해석하지 않고, $\kappa_h$, 대표 문서, cluster summary를 통해 cluster를 해석한다.

---

## 7. 발표용 한 문단 요약

> 기존 sparse vMF mixture는 $L_1$ penalty로 cluster prototype $\mu_h$ 자체를 sparse하게 만들고, 그 penalized estimator를 최종 결과로 사용한다. 본 연구는 이를 **support screening 단계**로 사용하고, 실제 vMF 판별항인 $\eta_h = \kappa_h \mu_h$ 에 **cluster-contrast penalty**를 부여한 뒤, 선택된 support에서 **원래 sphere 위의 sparse-vMF submodel을 unpenalized refit** 한다. 따라서 핵심 차별점은 sparse prototype 자체가 아니라, directional mixture에서의 **two-stage Lasso–MLE refit 구조와 shrinkage bias 완화**이다.

---

## 8. 향후 논의/확인 사항 (미팅 안건)

- [ ] Cluster-contrast penalty $P_B(\eta)$ 의 식별성(identifiability)과 weight $w_h$ 선택 기준
- [ ] Refit 단계에서 $\kappa_h$ 추정 시 component-specific vs. common 선택 기준
- [ ] BIC/ICL의 정당화: 같은 sample space 위 비교의 충분성 vs. 추가 보정 필요성
- [ ] Active set threshold $\epsilon$ 의 자료기반 결정 방법
- [ ] 실험 setup: synthetic data, benchmark text embedding (SBERT, OpenAI embeddings 등) 비교군
