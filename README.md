# Contextual Autocomplete 

This is the codebase for the analyses used in the MLHC 2020 paper, **Fast, Structured Clinical Documentation via Contextual Autocomplete**.

It does not contain the full system implementation of the BIDMC-JClinic system. While the notebooks cannot be run by the general public as they require access to PHI-derived data and models, utility files to load and call autocomplete for conditions, symptoms, medications, and labs are given, along with iPython notebooks to replicate the Results section of the paper.

## Citation

```
@inproceedings{gopinath2020contextual,
 abstract = { We present a system that uses a learned autocompletion mechanism to facilitate rapid creation of semi-structured clinical documentation. We dynamically suggest relevant clinical concepts as a doctor drafts a note by leveraging features from both unstructured and structured medical data. By constraining our architecture to shallow neural networks, we are able to make these suggestions in real time. Furthermore, as our algorithm is used to write a note, we can automatically annotate the documentation with clean labels of clinical concepts drawn from well-structured medical hierarchies to make notes more structured and readable for physicians, patients, and future algorithms. To our knowledge, this system is the only machine learning-based documentation utility for clinical notes deployed in a live hospital setting, and it reduces keystroke burden of clinical concepts by 67% in real environments.},
 author = {Divya Gopinath and Monica Agrawal and Luke Murray and Steven Horng and David Karger and David Sontag},
 booktitle = {Proceedings of the Machine Learning for Healthcare Conference},
 title = {Fast, Structured Clinical Documentation via Contextual Autocomplete},
 url_paper = {https://arxiv.org/abs/2007.15153},
 year = {2020}
}
```
