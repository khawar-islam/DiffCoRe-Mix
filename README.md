# DiffCoRe-Mix <img src="images/logo_diffcoremix.png" height="40">: Context-Guided Responsible Data Augmentation with Diffusion Models [ICLR-2025]
<p align="center">
    <img src="https://i.imgur.com/your_placeholder.png" alt="DiffCoRe-Mix Overview">
</p>

#### [Khawar Islam](mailto:khawar.islam@student.unimelb.edu.au)\*, [Naveed Akhtar](mailto:naveed.akhtar1@unimelb.edu.au)


#### **School of Computing and Information Systems, The University of Melbourne**

[![Paper](https://img.shields.io/badge/Paper-ICLR2025-blue)](https://github.com/your_repo_link)  
[![Code](https://img.shields.io/badge/Code-GitHub-brightgreen)](https://github.com/your_repo_link)  
[![License](https://img.shields.io/badge/License-MIT-yellowgreen)](LICENSE)

---

## 📢 Latest Updates
- **Mar-15-25**: Public release of the code and models.
- **Mar-10-25**: Paper accepted at **ICLR-2025**.
- **Mar-01-25**: Preprint is available.

---

## <img src="images/logo_diffcoremix.png" height="40"> Overview

**DiffCoRe-Mix** is a novel data augmentation approach that leverages text-to-image (T2I) diffusion models to generate semantically aligned augmentation samples. By combining contextual and negative prompts with a hard cosine similarity filtration in the CLIP feature space, DiffCoRe-Mix ensures that only high-quality generative images are mixed with real data. This robust augmentation strategy has been shown to improve model generalization on both general and fine-grained classification tasks.

**Key Features:**
- **Contextual & Negative Prompting:** Guides the diffusion process to generate domain-specific images while suppressing undesired content.
- **Hard Cosine Similarity Filtration:** Uses CLIP embeddings to filter out generated samples that do not meet semantic alignment criteria.
- **Composite Image Mixing:** Combines real and generative images using both pixel-wise and patch-wise strategies.

For visual insights, refer to:
- **Figure 2:** Overview of DiffCoRe-Mix data augmentation method.
- **Figure 8:** Computational cost analysis on Flowers102 and Stanford Cars datasets.
- **Figure 15:** Training batch visualization of DiffCoRe-Mix.

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
