# DiffCoRe-Mix : Context-Guided Responsible Data Augmentation with Diffusion Models [ICLR-2025]
<p align="center">
    <img src="assets/placeholder.png" alt="DiffCoRe-Mix Overview">
</p>

<p align="center">
  <a href="https://github.com/your_repo_link"><img src="https://img.shields.io/badge/Paper-ICLR2025-blue" alt="Paper"></a>&nbsp;&nbsp;
  <a href="https://github.com/your_repo_link"><img src="https://img.shields.io/badge/Code-GitHub-brightgreen" alt="Code"></a>&nbsp;&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellowgreen" alt="License"></a>
</p>

#### [Khawar Islam](mailto:khawar.islam@student.unimelb.edu.au), [Naveed Akhtar](mailto:naveed.akhtar1@unimelb.edu.au)
#### **School of Computing and Information Systems, The University of Melbourne**


## 📢 Latest Updates
- **Mar-15-25**: Public release of the code and models.
- **Mar-10-25**: Paper accepted at **ICLRw-2025**.
- **Mar-01-25**: Preprint is available.

---


### Key Features

- **Contextual & Negative Prompting:** Guides the diffusion process to generate domain-specific images while suppressing undesired content.
- **Hard Cosine Similarity Filtration:** Uses CLIP embeddings to filter out generated samples that do not meet semantic alignment criteria.
- **Composite Image Mixing:** Combines real and generative images using both pixel-wise and patch-wise strategies.


---

## Contents
- [1. Install](#1-install)
  - [1.1 Clone the Repository](#11-clone-the-repository)
  - [1.2 Create and Activate a Conda Environment](#12-create-and-activate-a-conda-environment)
  - [1.3 Install Dependencies](#13-install-dependencies)
- [2. Usage](#2-usage)
- [3. Train](#3-train)
- [4. Evaluation](#4-evaluation)
- [5. Repository Structure](#5-repository-structure)
- [6. Citation](#6-citation)
- [7. Acknowledgements](#7-acknowledgements)

---

## 1. Install

### 1.1 Clone the Repository
```bash
git clone https://github.com/your_username/DiffCoRe-Mix.git
cd DiffCoRe-Mix
