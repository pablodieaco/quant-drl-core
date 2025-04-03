# 📈 quant-drl-core

**quant-drl-core** is a research-focused framework for applying Deep Reinforcement Learning (DRL) to portfolio management and financial decision-making.  
It is designed to be **modular**, **extensible**, and easy to integrate into experimentation pipelines or production systems.

> 📘 This project was developed as part of my **Bachelor's Thesis (TFG)** for the **Double Degree in Mathematics and Computer Engineering** at the **University of Seville**.


---

## 🧠 Key Features

- ✅ Modular training pipeline for multiple DRL algorithms: `PPO`, `SAC`, `DDPG`, `TD3`
- ✅ Custom `Gymnasium`-compatible environments for realistic financial simulations
- ✅ Support for `CNN`, `LSTM`, `Transformer` and hybrid feature extractors
- ✅ Integrated technical indicators and raw market data handling
- ✅ Easy portfolio configuration (by sector and number of companies)
- ✅ Clean logging with `loguru`
- ✅ Compatible with interactive dashboards for visualization *(e.g., Streamlit webapp)*

---

## 🛠️ Getting Started

### 🔧 Installation

```bash
git clone https://github.com/your-username/quant-drl-core.git
cd quant-drl-core
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## 🧪 Training models

Basic command (default config)

```bash
python train_experiments.py
```

Custom training with options:

```bash
python train_experiments.py \
  --algorithms PPO SAC \
  --features LSTM CNNLSTM \
  --learning_rates 0.0003 0.0001 \
  --n_companies 10 18 \
  --length_train_data 8 \
  --length_eval_data 2 \
  --end_date_year 2021 \
  --update_hierarchy
```

Models and logs will be saved in:

- 📁 `models/` – trained model files
- 📁 `logs/` – experiment logs
- 📄 `models/metadata/hierarchy.json` – metadata for evaluation

---

## 📊 Evaluating models
Use the `evaluate_models.py` script to evaluate trained models.

- Option 1: Evaluate final models
```bash
python evaluate_models.py \
  --use_final_model
```

- Option 2: Evaluate by specific checkpoint steps
```bash
python evaluate_models.py \
  --steps_override PPO=500000 SAC=300000
```

- Option 3: Full customized evaluation
```bash
python evaluate_models.py \
  --hierarchy_path models/metadata/hierarchy.json \
  --results_path results/eval_sac_vs_ppo.csv \
  --models_dir models \
  --out_dir models/metadata/out \
  --steps_override PPO=500000 SAC=300000 \
  --no_move
```

---

## 📚 Notebooks

Inside `notebooks/` you’ll find:

- 📈  Data exploration
- 📊 Testing models
- 📝 Exporting results to LaTeX

---

## 🔮 Planned Features

- [ ] Risk-adjusted rewards (Sharpe, Sortino, etc.)
- [ ] Hyperparameter optimization with Optuna
- [ ] Integration with MLFlow
- [ ] Multi-agent extensions
- [X] Integration with a live dashboard (Streamlit) ([quant-drl-dashboard](https://github.com/your-username/quant-drl-dashboard))

---
## 📝 License

MIT License

---

## 🙋‍♂️ Author

Developed by **Pablo Diego Acosta**  
Connect via [LinkedIn](https://www.linkedin.com/in/pablodiegoacosta) or open an issue on GitHub 🚀

---

## 📚 References

This project is based on a wide range of research articles, theses, technical reports, and open-source implementations in the field of reinforcement learning and portfolio management.

Some of the most relevant references include:

- Filos, A. (2019). *Reinforcement Learning for Portfolio Management*. [arXiv:1909.09571](https://arxiv.org/abs/1909.09571)
- Jiang, Z. et al. (2017). *A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem*. [arXiv:1706.10059](https://arxiv.org/abs/1706.10059)
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Haarnoja, T. et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning*. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable RL Implementations*. [JMLR](http://jmlr.org/papers/v22/20-1364.html)
- Markowitz, H. (1959). *Portfolio Selection: Efficient Diversification of Investments*. Yale University Press.
- Open-source: [Zenlii – Deep RL Portfolio Management](https://github.com/Zenlii/Deep-Reinforcement-Learning-for-Portfolio-Management)

A complete list of all references (BibTeX format) used in this project is available [here](references.bib).