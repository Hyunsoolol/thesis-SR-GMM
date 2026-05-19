## 1. 연구 주제명

**Summary-Regularized Gaussian Mixture Models for Interpretable Text Clustering**

- **약칭:** SR-GMM
    
- **추정 알고리즘:** SR-EM (Summary-Regularized EM)
    

**핵심 아이디어는 다음이다.**

Text $\to$ LLM embedding $\to$ Gaussian mixture clustering $\to$ summary prototype $\to$ summary-regularized EM $\to$ interpretable clusters.

기존 Summaries as Centroids는 k-means의 numeric centroid를 textual summary embedding으로 주기적으로 대체하는 방식이다. 즉 hard assignment와 k-means objective를 유지하면서 centroid만 summary-derived centroid로 바꾼다. 본 연구는 이를 GMM으로 단순 확장하는 것이 아니라, summary embedding을 Gaussian mixture component mean에 대한 Mahalanobis regularization target으로 넣는다.

---

## 2. 기존 연구와의 차별점

- **기존 summary-as-centroid 방식:**
    
    $$ \mu_j \leftarrow \phi\{f(S_j)\} $$
    
    즉, cluster $j$에 속한 문서들을 요약하고, 그 요약문을 다시 embedding하여 centroid로 직접 사용한다.
    
- **제안 방법:**
    
    $\mu_j$를 $c_j = \phi{f(S_j)}$로 완전히 대체하지 않고, $\mu_j$가 data-driven mean과 summary prototype 사이의 compromise가 되도록 추정한다.
    

즉, 제안법은 다음을 연결하는 one-parameter family이다.

- $\rho = 0 \implies$ standard GMM
    
- $0 < \rho < \infty \implies$ summary-regularized GMM
    
- $\rho \to \infty \implies$ summary-replacement GMM analogue
    

여기서 $\rho \to \infty$의 극한은 k-LLMmeans와 유사하지만 동일하지 않다. k-LLMmeans는 hard assignment와 k-means objective를 쓰는 반면, SR-GMM은 soft responsibility와 Gaussian likelihood를 유지한다.

---

## 3. 데이터와 embedding

문서 집합을 $D = \{d_1, \dots, d_n\}$ 라고 둔다. 고정된 text encoder 또는 LLM embedding model을 $\phi: \mathcal{T} \to \mathbb{R}^p$ 라고 하면,

$$ x_i = \phi(d_i) \in \mathbb{R}^p $$

모형은 embedding space에서 정의한다.

$$ X = (x_1, \dots, x_n)^\top \in \mathbb{R}^{n \times p} $$

실제 구현에서는 필요하면 PCA 또는 whitening을 쓸 수 있다.

$$ x_i = U_q^\top (x_i - \bar{x}) $$

또는

$$ x_i = \Lambda_q^{-1/2} U_q^\top (x_i - \bar{x}) $$

단, PCA/whitening은 방법론 contribution이 아니라 numerical preprocessing으로 둔다.

---

## 4. 기본 Gaussian mixture model

전처리 후의 embedding을 다시 $x_i$라고 쓰면,

$$ x_i \stackrel{\text{iid}}{\sim} \sum_{j=1}^k \pi_j \mathcal{N}_p(\mu_j, \Sigma_j), \quad i=1, \dots, n $$

여기서 $\pi_j > 0$, $\sum_{j=1}^k \pi_j = 1$, $\mu_j \in \mathbb{R}^p$, $\Sigma_j \in \mathbb{S}_{++}^p$.

모수는 $\theta = (\pi, \mu, \Sigma)$.

Incomplete log-likelihood는 다음과 같다.

$$ \ell_n(\theta) = \sum_{i=1}^n \log \left[ \sum_{j=1}^k \pi_j \varphi(x_i; \mu_j, \Sigma_j) \right] $$

> **참고:** Miller and Alexander의 short-text clustering 논문도 LLM embedding을 만든 뒤 GMM으로 embedding space에서 cluster를 찾고, human reviewer 및 generative LLM으로 interpretability를 평가했다. 따라서 단순 “LLM embedding + GMM + LLM 해석”은 이미 가까운 선행연구가 있다.

---

## 5. Summary prototype

각 component $j$에 대해 summary input set을 만든다.

$$ S_j^{(t)} = \text{documents selected from } \{(d_i, \gamma_{ij}^{(t)})\}_{i=1}^n $$

Summarization operator를 $f: \mathcal{P}_{\text{fin}}(\mathcal{T}) \to \mathcal{T}$ 라고 두면,

$$ s_j^{(t)} = f(S_j^{(t)}) $$

이고, summary embedding은

$$ c_j^{(t+1)} = \phi(s_j^{(t)}) $$

PCA/whitening을 사용한 경우에는 같은 transform을 summary embedding에도 적용한다.

$$ c_j^{(t+1)} = P_q W\{\phi(s_j^{(t)}) - \bar{x}\} $$

이때 $c_j^{(t+1)}$는 component $j$의 semantic prototype이다.

---

## 6. 핵심 목적함수

제안하는 summary-regularized log-likelihood는

$$ Q_\rho(\theta; c) = \ell_n(\theta) - \frac{\rho}{2} \sum_{j=1}^k (\mu_j - c_j)^\top \Sigma_j^{-1} (\mu_j - c_j) $$

여기서 $\rho \ge 0$는 summary regularization strength이다.

**각 항의 의미:**

- $\ell_n(\theta)$: embedding data의 GMM likelihood
    
- $\frac{\rho}{2} (\mu_j - c_j)^\top \Sigma_j^{-1} (\mu_j - c_j)$: component mean과 summary prototype의 Mahalanobis discrepancy.
    

따라서,

$$ \hat{\theta}^{(c)} = \arg\max_\theta Q_\rho(\theta; c) $$

Mahalanobis 형태를 쓰는 이유는 단순 미관이 아니라 계산적으로 중요하다. Euclidean ridge를 쓰면 $\mu_j$ update에서 $N_j^{(t)} \Sigma_j^{-1} + \rho I_p$의 $p \times p$ inverse가 필요하지만, Mahalanobis penalty를 쓰면 $\Sigma_j^{-1}$가 gradient에서 소거되어 scalar-weighted convex combination이 된다. 현재 methodology draft에서도 이 점을 핵심 장점으로 정리하고 있다.

---

## 7. Empirical Bayes 해석

Penalty 항은 $-\frac{\rho}{2} (\mu_j - c_j)^\top \Sigma_j^{-1} (\mu_j - c_j)$ 이고, 이는 $\mu_j \mid \Sigma_j \sim \mathcal{N}(c_j, \Sigma_j / \rho)$의 quadratic kernel에 해당한다.

다만 proper normal prior density에는 $-\frac{1}{2} \log |\Sigma_j|$ 항이 추가로 포함된다. 현재 objective에는 이 log-determinant prior term이 없으므로, **full posterior equivalence가 아니라, conjugate normal location prior의 quadratic kernel로 해석한다.**

따라서 $\rho$는 전체 모수에 대한 uniform prior sample size가 아니라, **location-anchor precision**으로 해석하는 것이 안전하다.

---

## 8. SR-EM 추정 알고리즘

### 8.1 E-step

현재 모수 $\theta^{(t)} = (\pi^{(t)}, \mu^{(t)}, \Sigma^{(t)})$에서 posterior responsibility를 계산한다.

$$ \gamma_{ij}^{(t)} = \frac{\pi_j^{(t)} \varphi(x_i; \mu_j^{(t)}, \Sigma_j^{(t)})}{\sum_{\ell=1}^k \pi_\ell^{(t)} \varphi(x_i; \mu_\ell^{(t)}, \Sigma_\ell^{(t)})} $$

- **Effective cluster size:** $N_j^{(t)} = \sum_{i=1}^n \gamma_{ij}^{(t)}$
    
- **Weighted mean:** $\bar{x}_j^{(t)} = \frac{1}{N_j^{(t)}} \sum_{i=1}^n \gamma_{ij}^{(t)} x_i$
    

### 8.2 Prototype refresh step

Summary period를 $L$이라 두고, $L = \{t : t \equiv 0 \pmod L\}$.

만약 $t+1 \in L$ 이면 prototype을 갱신한다.

$$ s_j^{(t)} = f(S_j^{(t)}), \quad c_j^{(t+1)} = \phi(s_j^{(t)}) $$

그렇지 않으면,

$$ c_j^{(t+1)} = c_j^{(t)} $$

### 8.3 M-step: mean update

Penalized expected complete log-likelihood에서 $\mu_j$에 대해 미분한다.

$$ \frac{\partial}{\partial \mu_j} \left[ \sum_{i=1}^n \gamma_{ij}^{(t)} (x_i - \mu_j)^\top \Sigma_j^{-1} (x_i - \mu_j) + \rho(\mu_j - c_j^{(t+1)})^\top \Sigma_j^{-1} (\mu_j - c_j^{(t+1)}) \right] = 0 $$

따라서,

$$ \Sigma_j^{-1} \left[ \sum_{i=1}^n \gamma_{ij}^{(t)} (x_i - \mu_j) - \rho(\mu_j - c_j^{(t+1)}) \right] = 0 $$

정리하면 $N_j^{(t)} \bar{x}_j^{(t)} - N_j^{(t)} \mu_j - \rho \mu_j + \rho c_j^{(t+1)} = 0$.

따라서 closed-form update는

$$ \mu_j^{(t+1)} = \frac{N_j^{(t)} \bar{x}_j^{(t)} + \rho c_j^{(t+1)}}{N_j^{(t)} + \rho} $$

또는

$$ \mu_j^{(t+1)} = \left(\frac{N_j^{(t)}}{N_j^{(t)} + \rho}\right) \bar{x}_j^{(t)} + \left(\frac{\rho}{N_j^{(t)} + \rho}\right) c_j^{(t+1)} $$

**해석:**

$\mu_j^{(t+1)} =$ data-driven posterior mean + semantic summary prototype shrinkage.

**극한:**

- $\rho = 0 \implies \mu_j^{(t+1)} = \bar{x}_j^{(t)}$
    
- $\rho \to \infty \implies \mu_j^{(t+1)} \to c_j^{(t+1)}$
    

### 8.4 M-step: covariance update

$$ \Sigma_j^{(t+1)} = \frac{1}{N_j^{(t)}} \left[ \sum_{i=1}^n \gamma_{ij}^{(t)} (x_i - \mu_j^{(t+1)})(x_i - \mu_j^{(t+1)})^\top + \rho(\mu_j^{(t+1)} - c_j^{(t+1)})(\mu_j^{(t+1)} - c_j^{(t+1)})^\top \right] $$

- **첫 번째 항:** $S_j^{(t+1)} = \sum_{i=1}^n \gamma_{ij}^{(t)} (x_i - \mu_j^{(t+1)})(x_i - \mu_j^{(t+1)})^\top$
    
- **두 번째 항:** $A_j^{(t+1)} = (\mu_j^{(t+1)} - c_j^{(t+1)})(\mu_j^{(t+1)} - c_j^{(t+1)})^\top$
    

따라서,

$$ \Sigma_j^{(t+1)} = \frac{S_j^{(t+1)} + \rho A_j^{(t+1)}}{N_j^{(t)}} $$

**중요한 해석:**

$\rho$는 $\mu_j$ update에서는 semantic precision처럼 작동한다. 하지만 $\Sigma_j$ update에서는 denominator가 $N_j + \rho$가 아니라 $N_j$이다.

즉, $\rho$는 covariance에 대해 prior sample size처럼 작동하는 것이 아니라, **summary prototype과 data-driven mean 사이의 disagreement scatter를 추가한다.**

> 수치 검증에서는 $\mu$와 $\Sigma$의 closed-form update가 각각 수치 최적화 결과와 매우 작은 오차로 일치했다. 검증 보고서에 따르면 $\mu$ update의 최대 오차는 $2.31 \times 10^{-7}$, $\Sigma$ update의 최대 오차는 $1.38 \times 10^{-8}$이다.

### 8.5 M-step: mixing proportion update

Penalty는 $\pi_j$에 의존하지 않으므로 표준 GMM과 동일하다.

$$ \pi_j^{(t+1)} = \frac{N_j^{(t)}}{n} $$

---

## 9. Restricted covariance structures

고차원 embedding에서는 unrestricted covariance가 불안정할 수 있으므로 다음 covariance family를 고려한다.

- $(C_S)$: $\Sigma_j = \sigma_j^2 I_p$
    
- $(C_D)$: $\Sigma_j = \text{diag}(\sigma_{j1}^2, \dots, \sigma_{jp}^2)$
    
- $(C_E)$: $\Sigma_j = \Sigma$
    
- $(C_U)$: $\Sigma_j$ unrestricted
    

자유도는 예를 들어 다음과 같다.

- $r(k, C_S) = (k-1) + kp + k$
    
- $r(k, C_D) = (k-1) + kp + kp$
    
- $r(k, C_E) = (k-1) + kp + \frac{p(p+1)}{2}$
    
- $r(k, C_U) = (k-1) + kp + k \frac{p(p+1)}{2}$
    

실제 embedding 차원이 크면 $C_U$는 PCA-reduced space에서만 사용하는 것이 안전하다.

---

## 10. Summary input rules

### R1. Hard-thresholded input

$$ S_j^{(t), HT} = \{d_i : \gamma_{ij}^{(t)} > \tau\}, \quad \tau \in (0,1) $$

- **해석:** $\tau$ 큼 $\implies$ summary input은 작지만 purity 높음. $\tau$ 작음 $\implies$ summary input은 크지만 heterogeneous할 수 있음.
    

### R2. Top-m posterior input

$$ r_j^{(t)} = \text{argsort}_i \{-\gamma_{ij}^{(t)}\} $$

$$ S_j^{(t), TopM} = \{d_{r_j^{(t)}(1)}, \dots, d_{r_j^{(t)}(m)}\} $$

이 방식은 LLM context window가 제한된 경우 가장 실용적이다.

### R3. Posterior-weighted diversified sampling

Top-M candidate pool: $P_j^{(t)} = \{r_j^{(t)}(1), \dots, r_j^{(t)}(M)\}$.

이미 선택된 set을 $Z_{j, r-1}^{(t)}$ 라고 하면,

$$ D_i^2 = \min_{h \in Z_{j, r-1}^{(t)}} \|x_i - x_h\|^2 $$

Stochastic selection:

$$ \text{Pr}(i \text{ selected at step } r) \propto \gamma_{ij}^{(t)} D_i^2, \quad i \in P_j^{(t)} \setminus Z_{j, r-1}^{(t)} $$

이 rule은 posterior responsibility와 diversity를 동시에 반영한다.

---

## 11. Monotonicity와 prototype refresh

**Proposition 1. Fixed-prototype 구간의 monotonicity**

만약 $t+1 \notin L$ 이면 $c^{(t+1)} = c^{(t)}$.

이때 SR-EM은 $c^{(t)}$를 고정한 penalized objective에 대한 EM-type ascent이다.

$$ Q_\rho(\theta^{(t+1)}; c^{(t)}) \ge Q_\rho(\theta^{(t)}; c^{(t)}) $$

Prototype refresh가 있을 때는 $c$가 바뀌므로 $Q_\rho(\theta^{(t+1)}; c^{(t+1)}) \ge Q_\rho(\theta^{(t)}; c^{(t)})$를 일반적으로 주장하면 안 된다.

> 검증 보고서에서는 refresh 사이 31개 non-refresh iteration에서 monotonicity violation이 0개로 보고되었다.

**Proposition 2. Prototype refresh perturbation bound**

Prototype refresh에서 $c_j^+ = c_j + \delta_j$ 라고 하자. 고정된 $\theta$에 대해, $Q_\rho(\theta; c^+) - Q_\rho(\theta; c)$의 변화는 penalty 부분에서만 발생한다. Bound는

$$ |Q_\rho(\theta; c^+) - Q_\rho(\theta; c)| \le \frac{\rho}{2} \sum_{j=1}^k \|\Sigma_j^{-1}\|_{\text{op}} \|\delta_j\| [2\|\mu_j - c_j\| + \|\delta_j\|] $$

따라서 refresh-time dip은 $\rho$, $\|\delta_j\|$, $\|\Sigma_j^{-1}\|_{\text{op}}$, $|\mu_j - c_j|$에 의해 제어된다.

> 검증 보고서에서는 summarizer noise와 refresh-time dip 사이 상관이 $\text{corr}(\text{noise}, \text{dip}) = 0.97$로 보고되었다.

---

## 12. 모형 선택

### 12.1 $k$와 covariance family 선택

고정된 $\rho$에서 $(k, C)$는 BIC 또는 ICL로 선택한다.

$$ \text{BIC}(k, C \mid \rho) = -2\ell_n(\hat{\theta}_{k, C, \rho}) + r(k, C) \log n $$

ICL은 posterior uncertainty를 추가로 penalize한다.

$$ \text{Ent} = -\sum_{i=1}^n \sum_{j=1}^k \gamma_{ij} \log \gamma_{ij} $$

$$ \text{ICL} = \text{BIC} + 2\text{Ent} $$

### 12.2 $\rho$ 선택

**$\rho$는 BIC로 직접 선택하지 않는 것이 좋다.**

- **이유:** 고정된 $(k, C)$에서 $\rho = 0$은 unpenalized likelihood MLE이다. 따라서 unpenalized likelihood 기반 BIC로 $\rho$까지 선택하면 원칙적으로 $\rho = 0$이 우세하다.
    

따라서 권장 방식은 다음이다.

- **방법 1. Validation log-likelihood:** $\hat{\rho} = \arg\max_{\rho \in \mathbb{R}} \ell_{\text{val}}(\hat{\theta}_\rho)$. 단, well-specified Gaussian synthetic data에서는 validation likelihood도 $\rho=0$ 또는 작은 $\rho$를 선호할 수 있다.
    
- **방법 2. Sensitivity analysis:** $\rho = \kappa \bar{N}$, $\bar{N} = \frac{n}{k}$, $\kappa \in \{0, 0.1, 0.3, 1, 3, \infty\}$. 현재 단계에서는 이 방식이 가장 정직하다.
    

---

## 13. 수치 검증 결과 요약

현재 검증 결과는 크게 두 부류다.

### 13.1 수학적 정합성

- $\mu$ closed-form update: **PASS**
    
- $\Sigma$ closed-form update: **PASS**
    
- $\rho = 0 \implies$ standard GMM과 일치
    
- $\rho \to \infty \implies \mu \to c$
    
- non-refresh 구간 monotonicity: **PASS**
    
- refresh perturbation bound 방향성: **PASS**
    

검증 보고서상 $\mu$ closed-form은 수치 최적화와 $2.31 \times 10^{-7}$, $\Sigma$ closed-form은 $1.38 \times 10^{-8}$ 수준의 차이로 일치했고, $\rho = 0$은 sklearn GMM과 **0.6%** 이내로 일치했다.

### 13.2 합성 실험 결과

- **중간 난이도 setting:** $n=600, p=20, k=4, \text{sep}=3.0$
    
    - sklearn GMM: mean ARI = 0.425
        
    - SR-EM $\kappa=1.0$: mean ARI = 0.466
        
    - gain = +0.040, $p=0.094$
        
- **고차원 setting:** $p=50$
    
    - sklearn = 0.179, SR-EM $\kappa=3.0$ = 0.227
        
    - gain = +0.048, $p=0.099$
        
- **약한 분리 setting:** $\text{sep}=2.0$
    
    - sklearn = 0.233, SR-EM $\kappa=3.0$ = 0.234 (거의 개선 없음)
        
- Summarizer noise가 큰 경우에는 SR-EM $\kappa=0.3$가 +0.056 ($p=0.043$)의 gain을 보여, 현재까지 가장 의미 있는 개선 사례다.
    

---

## 14. 현재 결과의 해석

**강하게 주장 가능한 점:**

- SR-GMM은 standard GMM과 summary-replacement clustering을 잇는 one-parameter family다.
    
- SR-EM은 closed-form M-step을 가지며, fixed prototype 구간에서 monotone ascent를 가진다.
    
- 중간 $\rho$는 data-driven mean과 summary prototype의 절충을 제공한다.
    

**조심해야 할 점:**

- 현재 합성 실험에서 ARI gain은 작고 상황 의존적이다.
    
- 따라서 지금 단계에서 “항상 clustering accuracy를 향상시킨다”고 주장하면 안 된다.
    

**더 적절한 주장:**

> SR-GMM은 well-specified Gaussian data에서 보편적 성능 향상을 목표로 하기보다, misspecified text-embedding clustering에서 semantic prototype을 확률적 혼합모형 안에 안정적으로 통합하는 방법이다.

검증 보고서에서도 합성 실험만으로는 부족하며, Bank77, CLINC, GoEmo, MASSIVE 같은 실제 text benchmark에서 BERT/e5 embedding과 실제 LLM summarizer를 사용한 실험이 필수라고 정리되어 있다.

---

## 15. 연구미팅용 최종 알고리즘 요약

**Algorithm: SR-EM**

**Input:** $\{d_i\}_{i=1}^n$, $\{x_i = \phi(d_i)\}_{i=1}^n$, $k, \rho, L, f$

**Initialize:** $\theta^{(0)} = (\pi^{(0)}, \mu^{(0)}, \Sigma^{(0)})$, $c^{(0)}$

For $t = 0, 1, \dots, T-1$:

**Step 1. E-step**

$$ \gamma_{ij}^{(t)} = \frac{\pi_j^{(t)} \varphi(x_i; \mu_j^{(t)}, \Sigma_j^{(t)})}{\sum_{\ell=1}^k \pi_\ell^{(t)} \varphi(x_i; \mu_\ell^{(t)}, \Sigma_\ell^{(t)})} $$

$$ N_j^{(t)} = \sum_{i} \gamma_{ij}^{(t)}, \quad \bar{x}_j^{(t)} = \frac{1}{N_j^{(t)}} \sum_{i} \gamma_{ij}^{(t)} x_i $$

**Step 2. Prototype refresh**

If $t+1 \in L$, then

$$ s_j^{(t)} = f(S_j^{(t)}), \quad c_j^{(t+1)} = \phi(s_j^{(t)}) $$

Otherwise,

$$ c_j^{(t+1)} = c_j^{(t)} $$

**Step 3. Mean update**

$$ \mu_j^{(t+1)} = \frac{N_j^{(t)} \bar{x}_j^{(t)} + \rho c_j^{(t+1)}}{N_j^{(t)} + \rho} $$

**Step 4. Covariance update**

$$ \Sigma_j^{(t+1)} = \frac{1}{N_j^{(t)}} \left[ \sum_{i} \gamma_{ij}^{(t)} (x_i - \mu_j^{(t+1)})(x_i - \mu_j^{(t+1)})^\top + \rho(\mu_j^{(t+1)} - c_j^{(t+1)})(\mu_j^{(t+1)} - c_j^{(t+1)})^\top \right] $$

**Step 5. Mixing weight update**

$$ \pi_j^{(t+1)} = \frac{N_j^{(t)}}{n} $$

_Convergence는 prototype refresh 직후가 아니라 non-refresh iteration에서 평가한다._

---

## 16. 발표용 한 문장

> 기존 summary-as-centroid 방법은 k-means의 centroid를 summary embedding으로 대체하지만, 본 연구는 summary embedding을 GMM component mean에 대한 Mahalanobis regularizer로 사용한다. 그 결과 $\rho=0$에서는 standard GMM, $\rho \to \infty$에서는 summary-replacement GMM analogue, 중간 $\rho$에서는 data-driven mean과 semantic prototype의 절충 추정량을 얻는다. 핵심 수식은 $\mu_j^{\text{new}} = (N_j \bar{x}_j + \rho c_j) / (N_j + \rho)$이며, 이 closed-form update가 본 방법의 가장 중요한 통계적 장점이다.

---

## 17. 다음 단계

현재 연구미팅에서 결론은 다음처럼 제시하면 좋습니다.

1. 방법론 수식은 정리되었고, closed-form update는 수치적으로 검증되었다.
    
2. 합성 Gaussian data에서는 gain이 작고 상황 의존적이다.
    
3. 따라서 논문의 승부는 실제 text embedding benchmark에서 intermediate $\rho$가 얼마나 효과적인지에 달려 있다.
    

**실험 계획 요약:**

- **필수 real-data benchmark:** Bank77, CLINC, GoEmo, MASSIVE.
    
- **비교군:** k-means, GMM, k-NLPmeans, k-LLMmeans, BERTopic, SR-GMM.
    
- **평가지표:** ACC, NMI, ARI, LLM call cost, interpretability of summaries.
    

> **참고:** 기존 Summaries as Centroids도 Bank77, CLINC, GoEmo, MASSIVE 등을 사용하고, k-means, GMM, BERTopic, ClusterLLM, LLMEdgeRefine 등과 비교했기 때문에, 이 benchmark 구성을 그대로 따라가는 것이 논문 설득력에 가장 좋다.
