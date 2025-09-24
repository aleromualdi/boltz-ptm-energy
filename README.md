# Boltz with Implicit Binding Energies

This fork extends the original [Boltz repository](https://github.com/jwohlwend/boltz) with **Energy computation functionality** based on the Joint Energy-based Model (JEM) framework. The implementation provides two key energy functions derived from PAE (Predicted Alignment Error) logits:

- **pAEnergy**: Computes energy from PAE logits using LogSumExp over error bins
- **pTMEnergy**: Weighted variant incorporating TM-score scaling kernel for inter-chain interactions

Energy computations are automatically performed during structure prediction. Energy values are saved as `.npz` files alongside other prediction outputs.

The energy functions are implemented in `src/energy/pae_energy.py` and integrated into the model prediction pipeline. The implementation follows the mathematical framework described in the paper *Protein Structure Predictors Implicitly Define Binding Energy Functions* Jingzhou Wang, Yang Zhang, & Jian Peng. (2023).


---

## Theoretical Background

This work is grounded in the **Joint Energy-based Model (JEM)** framework from  *"Your classifier is secretly an energy-based model and you should treat it like one"* (Grathwohl et al., 2019/2020).

### Classifiers as Energy Models

A standard classifier with logits $f_\theta(x)[y]$ implicitly defines an **energy function**:

$E_\theta(x,y) = -f_\theta(x)[y].$

By marginalizing over labels $y$:

$E_\theta(x) = -\log \sum_y \exp(f_\theta(x)[y])$,

we see that classifiers assign an **energy landscape** over inputs $x$, not just class probabilities.  

---

### Application to Protein Structure Predictors

Protein structure predictors like Boltz output **pAE (Predicted Alignment Error) logits**, which are distributions over error bins for each residue–residue pair. **pAEnergy** is defined as:

$E_{\text{pAE}}(x) = -\frac{1}{M} \sum_{i<j} \log \sum_b \exp(f_{\text{pAE}}(x)[i,j,b]).$

This aggregates the log-sum-exp of logits across all residue pairs. It reflects overall **confidence of alignment**: concentrated logits → lower energy; spread logits → higher energy.

**pTMEnergy** refines this by applying a **TM-score kernel** $g(d)$ and restricting to inter-chain pairs:

$E_{\text{pTM}}(x) = -\frac{1}{M}\sum_{i<j \in \text{interface}}
\log \sum_b \exp\ \big(f_{\text{pAE}}(x)[i,j,b] + \log (g(d_b))\big),$

where

$g(d) = \frac{1}{(1 + d/d_0(N))^2}, \quad d_0(N) = 1.24 \sqrt[3]{N-15} - 1.8.$

This penalizes large error bins and focuses the energy on **interface stability**, making it much better correlated with binding free energie $\Delta G$.

---

### Steering in Boltz

Boltz uses steering forces during diffusion sampling to bias structures toward binding-competent conformations. I believe that using ipTM, which is bounded in $[0,1]$, limits steering to relatively small adjustments. In contrast, pTMEnergy, being unbounded and based on a log-sum-exp formulation, sharpens logits at interface residues, leading to stronger shifts and improved discrimination between binders and non-binders.


## Extending Boltz: PAE Energy Computation

### New Files

1. **`src/energy/pae_energy.py`**
   - **`compute_pae_energy()`**
     - Input: PAE logits tensor (batch × residues × residues × bins)
     - Computes energy via LogSumExp over error bins:

       $E_{pTM}[i,j] = \text{LogSumExp}_y (\log (g(y)) + f(x)[i,j,y])$

     - Excludes diagonal elements (self-pairs)
     - Returns scalar energy per batch element
   - **`compute_ptm_energy()`**
     - Incorporates TM-score scaling kernel  
       $g(d) = \frac{1}{(1 + d/d_0)^2}, \quad d_0 = 1.24 \cdot (N-15)^{1/3} - 1.8$

     - Formula $E_{pTM}[i,j] = \text{LogSumExp}_y (\log (g(y)) + f(x)[i,j,y])$

     - Focuses only on inter-chain residue pairs


### Model Integration

- `src/boltz/model/models/boltz1.py`: Added energy computation inside `predict_step()`

- `src/boltz/model/models/boltz2.py`: Same integration as `boltz1.py`

- `src/boltz/data/write/writer.py`: Extended confidence summary to include `pae_energy` and `ptm_energy`



