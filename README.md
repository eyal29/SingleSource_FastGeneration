# Single-Source Fast Generation
### Distilling a Single-Image Diffusion Model into a One-Step GAN

**Cours :** GENAI — Generative Artificial Intelligence  
**Autrices :** LACHHEB Eya, FELGINES Sara, HADIOUCHE Lyliane, ERRADI Lina

---

## Concept

Ce projet combine deux articles pour créer un pipeline de génération **rapide** à partir d'une **image source unique** :

| Article | Rôle |
|---|---|
| [SinFusion](https://arxiv.org/abs/2211.11743) — Nikankin et al., ICML 2023 | **Teacher** : DDPM entraîné sur une seule image, génération via DDIM (50 étapes) |
| [Diffusion2GAN](https://arxiv.org/abs/2405.05967) — Kang et al., ECCV 2024 | **Distillation** : remplacer les 50 étapes par un GAN one-step |

**Idée centrale :** le teacher génère des paires `(bruit z_T → image x_0)` via DDIM déterministe. Un GAN student apprend à reproduire ce mapping en **une seule passe forward**, avec un speedup de ×49.

---

## Résultats obtenus

| Image | Speedup | L1 (student vs teacher) | LPIPS (student vs teacher) |
|---|---|---|---|
| `fruit.png` | ×49.3 | 0.3207 | 0.5009 |
| `colusseum.png` | ×48.9 | 0.2649 | 0.5099 |

> Teacher : ~2100 ms/image (50 étapes DDIM) — Student : ~43 ms/image (1 étape GAN)

---

## Pipeline

```
[Image source]
      │
      ▼
┌─────────────────────────────────────────────┐
│ Notebook 01 — Entraîner SinFusion (teacher) │
│   → 50 000 steps, checkpoint .ckpt          │
│   → copié dans teacher_checkpoint/{image}/  │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│ Notebook 02 — Construire le dataset ODE     │
│   → 2000+ paires (z_T, x_0) via DDIM       │
│   → stocké dans outputs/{image}/ode_dataset/│
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│ Notebook 03 — Distillation GAN (student)    │
│   → PatchGAN conditionnel + hinge + LPIPS   │
│   → checkpoint .pt sauvegardé              │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│ Notebook 04 — Inférence & Comparaison       │
│   → benchmark vitesse, grille visuelle      │
│   → métriques L1 / LPIPS / speedup         │
└─────────────────────────────────────────────┘
```

---

## Structure du projet

```
SingleSource_FastGeneration/
├── notebooks/                        # Templates (sans outputs — à lancer sur OVH)
│   ├── 01_train_sinfusion_ovhcloud.ipynb
│   ├── 02_build_ode_dataset.ipynb
│   ├── 03_train_distillation.ipynb
│   └── 04_inference_comparison.ipynb
│
├── notebooks_results/                # Archives avec outputs par image testée
│   ├── fruit.png/
│   └── colusseum.png/
│
├── teacher_checkpoint/               # Checkpoints teacher (Git LFS)
│   ├── fruit.png/last.ckpt
│   └── colusseum.png/last.ckpt
│
├── outputs/                          # Résultats générés
│   ├── fruit.png/fruit_teacher/
│   │   ├── distillation/checkpoints/ # Checkpoints student (.pt, Git LFS)
│   │   ├── distillation/samples/     # Générations pendant l'entraînement
│   │   └── inference/                # Comparaison teacher vs student
│   └── colusseum.png/colusseum_teacher/
│       └── ...
│
├── sinfusion/                        # Git submodule (code SinFusion original)
├── setup.sh                          # Script d'installation OVH
└── .gitattributes                    # Git LFS : *.ckpt, *.pt
```

---

## Utilisation

### Prérequis

- Accès à un GPU (OVH AI Notebooks recommandé — V100 32 GB)
- Git LFS installé : `git lfs install`

### 1. Cloner le repo

```bash
git clone --recurse-submodules https://github.com/eyal29/SingleSource_FastGeneration.git
cd SingleSource_FastGeneration
git lfs pull   # télécharger les checkpoints
```

### 2. Lancer sur OVH AI Notebooks

```bash
bash setup.sh   # installe PyTorch cu118, dépendances, init submodule
```

Ouvrir et exécuter les notebooks dans l'ordre :

1. `01_train_sinfusion_ovhcloud.ipynb` — changer `IMAGE_NAME` en haut de cellule 2
2. `02_build_ode_dataset.ipynb` — même `IMAGE_NAME`
3. `03_train_distillation.ipynb` — même `IMAGE_NAME`, `RESUME = False` pour repartir
4. `04_inference_comparison.ipynb` — même `IMAGE_NAME`

> **Un seul paramètre à changer** : `IMAGE_NAME = 'votre_image.png'` dans chaque notebook.  
> Tous les chemins (outputs, checkpoints) se dérivent automatiquement.

### 3. Pousser les résultats

```bash
# Après notebook 01 : copier le checkpoint teacher
git add teacher_checkpoint/
git commit -m "feat: teacher checkpoint {image}"
git push

# Après notebooks 03/04 : pousser outputs et notebooks_results
git add outputs/{image}/ notebooks_results/{image}/
git commit -m "feat: résultats {image}"
git push
```

---

## Choix techniques

| Composant | Choix | Justification |
|---|---|---|
| Architecture student | NextNet (même que teacher) | Réutilise le backbone entraîné sur l'image |
| Conditionnement | `t=0` constant | Le student n'a pas besoin du timestep |
| Discriminateur | PatchGAN conditionnel sur `z_T` | Évite le mode collapse, juge les textures locales |
| Normalisation | Spectral norm | Stabilise l'entraînement GAN |
| Losses | Hinge + LPIPS (VGG) + L1 | Réalisme + fidélité perceptuelle + ancrage pixel |
| Optimisation | TTUR : `lr_D = 4 × lr_G`, Adam β=(0, 0.99) | Standard GAN, maintient la pression adversariale |
| Précision | AMP (float16/float32) | ×1.5 speedup entraînement |
| Dataset ODE | 2000 paires DDIM déterministe | step_size=1 (50 étapes complètes) pour qualité maximale |
