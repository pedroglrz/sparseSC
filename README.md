# sparseSC
Sparse Autoencoder for Single Cell data

## Project Outline

  

I am interested in exploring how [sparse autoencoders](https://arxiv.org/abs/2309.08600) can be leveraged to extract interpretable features from learned representations of [single cell data.](https://www.nature.com/articles/s41596-020-00409-w)

## Background and Key Terms

- Transcriptomic methods measure the differential [RNA](https://en.wikipedia.org/wiki/RNA) expression transcribed from [DNA](https://en.wikipedia.org/wiki/DNA).  Within groups of cells with identical DNA, RNA expression levels massively differ across genes driving differentiable cell states.
    
- Single cell RNA-seq data is a particular transcriptomic measurement where individual cells have their RNA sequenced.  The data is represented as a cell x gene counts matrix where each cell is a vector of gene counts.  Because we are dealing with count data the raw data is non-negative integers.
    
- Cell Types are one of the most straightforward categorical labels that are differentiated by transcriptomic signal.  An example is for a given sample of white blood cells there might be [CD4 cells](https://en.wikipedia.org/wiki/CD4) that are characterized for their role in regulating a system’s immune response and [CD8 cells](https://en.wikipedia.org/wiki/Cytotoxic_T_cell) which directly attack invasive pathogens.  For the purposes of this exploration cell types are a useful categorical variable that characterize true and verifiable biological signal.
    

## Problem:

- Biologically Uninterpretable Representations: Many current single-cell RNA-seq models rely heavily on non-linear transformations, resulting in representations that are effective for downstream tasks (e.g., clustering, prediction tasks, dataset integration) but lack biological interpretability.
    
- Polysemantic Representations: By reducing the representation space, these models risk producing dimensions where multiple biological features are entangled, making it difficult to find independent axes of variation that describe biological "directions" (features).
    

## Proposition:

- Combining the signal-extracting ability of deep learning models with sparse autoencoders, we aim to create rich, interpretable, monosemantic axes of variation for these representations. These axes will allow us to better decode high-dimensional transcriptomic signals that underlie distinct biological states.
    

  

## Methods:

### Dataset and Preparation:

- Two related single cell datasets, from Lee et al. and Arunachalam et al., were queried from [this](https://cellxgene.cziscience.com/collections/b9fc3d70-5a72-4479-a046-c2cc1ab19efc) data repository.
    
- Both datasets are from peripheral blood samples from patients with and without covid.
    
- Since the studies were conducted separately, the technical signal between experiments overwhelms the interesting biological signal.  These are called batch effects.
    
- Basic preprocessing standard to the domain were conducted.
    

  

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfWglmrKoBaVGsQUWSRM-TZBFH4dCkkECw8NOHwdNRSPblQhzKOB-fJC9CfscxrcYXyQMqnol9QAyP-JFh8jlfnPoO1jUdRkpWSNHy1n8W_q0rAAHodeID31bF5c_OX7dEyH8g6_63a55IdWl_4Aii6ug4?key=B_PLCf9l-ZN0jQYO7bdzDw)

  
  
  
  
  

#### Dataset Integration and Representation Learning:

- To align these datasets with their shared biological signals we employ a standard implementation  [scVI](https://docs.scvi-tools.org/en/1.1.5/user_guide/models/scvi.html).
    
- SCVI is a deep learning architecture that assumes a latent cell state representation ‘z’ that is independent of technical variables such as batch, or experiment.
    
- Crucially we use scVI to learned a reduced representation of cell across both datasets that corrects for technical batch effects and maintain biological signal.  This take the place of the final MLP layer used in monosemanticity method.
    
-   
    

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXcsCRHQN5GOS8csDEY7PIt8pwy8Skpt2HLk5hzVm2cL7-plX1ieWNYDSStK4KfQm03cUk5TLlcQOlbmKS81dyCCkq_GSc6HpQkBuveq_QZW4SQsfUeOoeSEhrtuDZxOWhX3HIASTPYNH6RUgRDMytapxp0s?key=B_PLCf9l-ZN0jQYO7bdzDw)

  
  
  

#### Learning a Sparse representation with an Autoencoder:

- A sparse autoencoder built to the general configuration specified in the literature is trained on the SCVI representation. Full code here.
    

  

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeukhd44m_CLDuMshKBf9AB8iDbgAmkTtdR_E1bkbDlKJEfGvE2AoIFFkJ9Yms9FJupRQAdFUvya2SlfJKO9qoR74X8S06budIQWWc6EBsZ26Vj3ONLHs7LGnyvWI1vr0NqJvE5RskAzLLxDP1g4gDZLXA?key=B_PLCf9l-ZN0jQYO7bdzDw)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXewORi6fOPbqeX5JBBDgvKnycXpPPfeiKcklipwNW9UxRG4A2URbjD4culLpntumuOiAL0LZPFfS11UadJf_w22HdBEl9JA6RsSd-I-j6HzLUgYJqxL2hAsyqcqOGnwLghAS6zUIUjvHSpups0s81eDv4Ox?key=B_PLCf9l-ZN0jQYO7bdzDw)

  

#### Results

**