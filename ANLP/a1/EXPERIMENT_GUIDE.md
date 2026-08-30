# ANLP Assignment 1: Experiment & Architecture Guide

**Course:** Advanced Natural Language Processing (Spring 2026)  
**Topic:** Custom Transformers, XOR Cipher Decryption, Architectural Ablations (C1–C5), and Byte Latent Transformers (BLT)

---

## 1. Executive Summary & Cipher Analysis

### The Underlying Cipher
Analysis of `data/brown_cipher.txt` and `data/brown_plain.txt` reveals that the dataset is a **deterministic XOR stream cipher** using the repeating 8-byte ASCII key:

$$\text{Key} = \text{"ANLP2026"} \quad (\text{ASCII bytes: } [65, 78, 76, 80, 50, 48, 50, 54])$$

- **Encryption Formula**: $C[i] = P[i] \oplus K[i \pmod 8]$
- **Theoretical Optimum**: Because the mapping is 100% deterministic and information-preserving, a properly trained Seq2Seq model can achieve **~100% bit accuracy** and **~100% sequence accuracy**.

---

## 2. Failure Analysis & The "English Prior" Trap

In initial experiments, all models plateaued with **0% sequence accuracy** and characteristic bit accuracies:

| Configuration | Initial Bit Acc | Baseline Match | Behavior |
| :--- | :---: | :---: | :--- |
| **C1_Base** | `0.6564` | `0.6614` | Collapsed to predicting space / common English ASCII bits (`"the the..."`) |
| **C2_RoPE** | `0.6489` | `0.6614` | Collapsed to predicting space / common English ASCII bits |
| **C3_GQA** | `0.6591` | `0.6614` | Collapsed to predicting space / common English ASCII bits |
| **C4_RMSNorm** | `0.6522` | `0.6614` | Collapsed to predicting space / common English ASCII bits |
| **C5_BLT** | `0.5454` | `0.5470` | Collapsed to predicting empty / padding tokens |

