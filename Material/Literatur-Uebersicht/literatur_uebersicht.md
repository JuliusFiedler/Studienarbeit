
<!-- Markdown Code Kann z.B. mit https://github.com/joeyespo/grip gerendert werden -->



### 1997 Christopher G. Atkeson and Stefan Schaal: "Robot Learning From Demonstration"

1997_Atkeson_Schaal__ICML__Robot Learning From Demonstration.pdf

- College of Computing, Atlanta
- ATR Human Information Processing, Kyoto


### 2004 Weiwei Li, Emanuel Todorov: "Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems"


2004_Li_Todorov__conf_ICINCO__Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems.pdf

- Univ. San Diego
- Erstes Paper zu iLQR



### 2005 Emanuel Todorov and Weiwei Li: "A generalized iterative LQG method for locally-optimal feedback control of constrained nonlinear stochastic systems"


2005_Todorov_Li__CDC43__A generalized iterative LQG method for locally-optimal feedback control of constrained nonlinear stochastic systems.pdf


- Univ. San Diego
- Erweiterung des iLQR auf stochastische Systeme


### 2008 Djordje Mitrovic, Stefan Klanke, Sethu Vijayakumar: "optimal control with adaptive internal dynamics models"

2008_Djordje_Mitrovic_Klanke_Vijayakumar_conf_INCINCO__optimal control with adaptive internal dynamics models.pdf

- Univ. Edinburgh


### 2011: Marc Deisenroth: "PILCO"

- Paderborn
- braucht wenig Interaktionszeit, aber zwischen den Experimenten viel Rechenzeit
- Ursprünglich: Systemmodell: GP
- Erweiterung von McAllister: Bayssche neuronales Netze (kein "Baysches Netz")



### 2014 Joschka Boedecker*, Jost Tobias Springenberg*, Jan Wülfing*, Martin Riedmiller: Approximate Real-Time Optimal Control Based on Sparse Gaussian Process Models

2014_Boedecker_Springenberg_Wuelfing_Riedmiller__ieee_ADPRL__Approximate Real-Time Optimal Control Based on Sparse Gaussian Process Models


- Machine Learning Lab, University of Freiburg


---
@misc{1606.01540,
  Author = {Greg Brockman and Vicki Cheung and Ludwig Pettersson and Jonas Schneider and John Schulman and Jie Tang and Wojciech Zaremba},
  Title = {OpenAI Gym},
  Year = {2016},
  Eprint = {arXiv:1606.01540},
}

---
### Akihiko Yamaguchi and Christopher G. Atkeson: Neural Networks and Differential Dynamic Programming for
Reinforcement Learning Problems

- ähnlicher Ansatz, wie unserer (Erweiterung auf stochastische DDP) und andere Anwendung (Robotik/Manipulation)
### 2017 Gilwoo Lee, Siddhartha S. Srinivasa, Matthew T. Mason:  "GP-ILQG: Data-driven Robust Optimal Control for Uncertain Nonlinear Dynamical Systems"


2017_Lee_Srinivasa_Mason__arxiv__GP-ILQG__Data-driven Robust Optimal Control for Uncertain Nonlinear Dynamical Systems.pdf

- Carnegie Mellon
-



### 2017_Anusha Nagabandi, Gregory Kahn, Ronald S. Fearing, Sergey Levine: "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"

Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning

2017_Nagabandi_Kahn_Fearing_Levine__arxiv__Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning

- University of California, Berkeley
- Animationsscreenshot: Vierbeinspinnen auf Schachbrett (Mujoco)
- Levine: RL-Guru, hat Connections zu openAI
- Ansatzpunkt: bei hochdimensionalen Systemen (Walkern, ...) gehen GP-Modelle bzw. PILCO nicht mehr
- Umgebung: Mujoco

---

- Hier: Bisher nur MPC durch Naives Sampeln (besten Kandidaten auswählen)
- Empfehlen Kombination mit iLQR (Referenz {39})
- → **Basis für unseren Ansatz**



### 2014 Yuval Tassa, Nicolas Mansard, Emo Todorov: "Control-Limited Differential Dynamic Programming"


2014_Tassa_Mansard_Todorov__ieee_icra__Control-Limited Differential Dynamic Programming

- Univ. Washington
- Univ. Toulouse
- Hier werden Eingangsbeschränkungen mit berücksichtigt

---
- kein RL, sondern klassische Trajektorienplanung
-



### 2018 Kurtland Chua,  Roberto Calandra,  Rowan McAllister,  Sergey Levine: "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models"


2018_Chua_Calandra_McAllister_Levine__NeurIPS__Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models


- University of California, Berkeley
- Our comparison to state-of-the-art model-based {efficient} and model-free {good asymptotic performance} deep RL algorithms shows that our approach matches the asymptotic performance of model-free algorithms on several challenging benchmark tasks, while requiring significantly fewer samples


### 2019 Sarah Bechtle, Akshara Rai, Yixin Lin, Ludovic Righetti, Franziska Meier: "Curious iLQR: Resolving Uncertainty in Model-based RL"


2019_Bechtle_Rai_Lin_Righetti_Meier__arxiv__Curious iLQR: Resolving Uncertainty in Model-based RL

- MPI Intelligent Systems Tübingen
- Facebook AI,
- Univ. New York
---
- we propose a model-based reinforcement learning (MBRL) framework that combines Bayesian modeling of the system dynamics with curious iLQR, a risk-seeking iterative LQR approach
-  curious iLQR attempts to minimize both the task-dependent cost and the uncertainty in the dynamics model. We scale this approach to perform reaching tasks
on 7-DoF manipulators, to perform both simulation and real robot reaching experiments.
- Our experiments consistently show that MBRL with curious iLQR more easily overcomes bad initial dynamics models and reaches desired joint configurations more reliably

### 2019 Kadierdan Kaheman, Eurika Kaiser, Benjamin Strom, J. Nathan Kutz and Steven L. Brunton: "Learning Discrepancy Models From Experimental Data"


2019_Kaheman_Kaiser_Strom_Kutz_Brunton__arxiv__Learning Discrepancy Models From Experimental Data.pdf


- Univ. Washington (Mechanical Eng. + Math)
- In this work, we use the sparse identification of nonlinear dynamics (SINDy) algorithm
- we assume that the model mismatch can be sparsely represented in a library of candidate model terms.
- We further design and implement a feed-forward controller {for the double pendulum} in simulations, showing improvement with a discrepancy model

---

- ähnlich zu "equation learning"
- Kann gut Extrapolieren
- (-) Man braucht gute Ansatzfunktionen

---
- keine KNN /GP
- Nur Simulationsergebnisse fürs Aufschwingen
- wie wird Trajektorie geplant?





### 2018 Paavo Parmas, Carl Edward Rasmussen, Jan Peters, Kenji Doya: "PIPPS: Flexible Model-Based Policy Search Robust to the Curse of Chaos"

2018_Parmas_Rasmussen_Peters_Doya__conf_ICML35__PIPPS- Flexible Model-Based Policy Search Robust to the Curse of Chaos.pdf

- Univ. Okinawa
- Univ. Cambridge
- TU Darmstadt,
- Max Planck Institute for Intelligent Systems, Tübingen

---

- We show that reparameterization gradients suffer
from the problem, while likelihood ratio gradi-
ents are robust. Using our insights, we develop
a model-based policy search framework, Proba-
bilistic Inference for Particle-Based Policy Search
(PIPPS), which is easily extensible, and allows
for almost arbitrary models and policies, while
simultaneously matching the performance of pre-
vious data-efficient learning algorithms. Finally,
we invent the total propagation algorithm, which
efficiently computes a union over all pathwise
derivative depths during a single backwards pass,
automatically giving greater weight to estimators
with lower variance, sometimes improving over
reparameterization gradients by 106 times.



