# The stress Hessian — interactive note (NEI)

모든 쌍이 spring 인 network 에서 node 하나를 움직일 때 curvature 가 어떻게 바뀌는지 보는 interactive page. `index.html` 을 열면 바로 동작한다 (KaTeX·font 동봉, 외부 접속 없음).

$$F(X)=\sum_{i<j}(d_{ij}-\delta_{ij})^2,\qquad \mathbf H_{ii}=\underbrace{2\sum_{j\ne i}\mathbf u_{ij}\mathbf u_{ij}^{\mathsf T}}_{K\succeq0}+\underbrace{\sum_{j\ne i}\omega_{ij}\bigl(\mathbf I-\mathbf u_{ij}\mathbf u_{ij}^{\mathsf T}\bigr)}_{S},\qquad \omega_{ij}=\frac{2(d_{ij}-\delta_{ij})}{d_{ij}}$$

- $\delta_{ij}$ = unweighted hop distance, node position = metric MDS terminal (classical MDS → SMACOF → L-BFGS, deterministic).
- $\operatorname{tr}K=2(N-1)$ 은 위치와 무관하므로 curvature type (BOWL / SADDLE / DOME) 을 정하는 것은 prestress $S$ 다.
- network 5개: Zachary karate club, Lusseau dolphins, Les Misérables, college football, ca-netscience.
- 각 terminal 에서 full Hessian 이 zero mode 3개 (translation 2 + rotation 1) 외에 모두 양수임을 확인했다.

## NEI ensemble (red)
NEI 는 terminal 분포의 statistic 이다. 각 network 를 census protocol (Uniform[0,1] 시작, seed 1000+m → SMACOF → L-BFGS) 로 $M=500$ 번 relax 해서
$\mathrm{NEI}=\langle\operatorname{Var}_m(d_{ij})/\langle d_{ij}\rangle_m^2\rangle_{i<j}$ 를 계산하고, terminal 들을 기준 MDS terminal 에 Procrustes 로 맞춘 뒤
ensemble-mean slice $\langle F_i\rangle_M$ 와 그 Hessian $\langle\mathbf H_{ii}\rangle_M=\langle K\rangle_M+\langle S\rangle_M$ 을 붉은 계열로 그린다.
gauge 와 무관한 비교로 각 terminal 자기 위치의 $\mathbf H_{ii}^{(m)}$ eigenvalue 분포도 함께 그린다.

| network | N | NEI (M=500) |
|---|---|---|
| karate | 34 | 0.0528 |
| dolphins | 62 | 0.0443 |
| Les Misérables | 77 | 0.0750 |
| college football | 115 | 0.0531 |
| ca-netscience | 352 | 0.0329 |

## Page
- plan view 를 다섯 layer 로 분리: contour ($\log F_i$) · spring stress · network · eigenvector · NEI ensemble. overlay / split 전환.
- 3D landscape: top / tilt / side view, azimuth·elevation·zoom·height scale, drag·wheel·pinch.
- live readout: $\mathbf H_{ii}=K+S$ (KaTeX), eigenvalue, curvature type.
- design: KRDS 디자인 스타일 확장형 + 경상국립대 색·서체 (로고 없음). token 은 `assets/krds/` (Gahyoun/gnu-network-science-skills `templates/krds-design-system`).

## Build
```sh
python build/make_nets.py        # build/data/*.edgelist → build/nets/*.json (terminal, preset path, checks)
python build/ensemble.py         # M=500 census terminals → build/nets/*_ens.npz (NEI, aligned terminals; ~10 min, 9 procs)
python build/build_page.py       # → index.html
node build/check_katex.cjs index.html   # math compile check (raw '<' in markup math 도 실패로 처리)
```
Python: numpy, scipy, networkx.

## Scope
- 보여 주는 것은 node 하나의 2D slice 와 그 block $\mathbf H_{ii}$ 다. barrier 와 second basin 은 다른 node 를 고정한 conditional 값이다.
- preset node 는 현상이 잘 보이는 node 를 고른 것이다 (선택 규칙은 page 의 Scope 절).

## License
KaTeX (MIT), Noto Sans KR · SUITE (SIL OFL 1.1). 데이터 출처는 page 의 References.
