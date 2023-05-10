# Enabling Fully-Qualified Name Resolution of API Elements via Generative Text Infilling

## Abstract
Software development relies heavily on the effective use of API elements, which provide pre-built functionalities for specific programming tasks. Online forums, e.g., StackOverflow and GitHub Gists contain vast collections of code snippets that utilize these APIs. However, due to the incomplete nature of these code snippets, determining the correct fully-qualified name (FQN) for API elements is challenging. Several approaches have been proposed to automatically resolve the FQNs, but face one or more of the following issues: (1) are not comprehensive; (2) can not handle out-of-vocabulary API elements; (3) do not consider dependence of the API element on other program elements. In this work, we present DeepFQN, a deep learning-based approach that addresses these issues by formulating FQN resolution as a fill-in-the-blank task. For a given code snippet, our tool systematically identifies all API locations, inserts a blank ahead of the simple name corresponding to the API, leverages a causal language model (CLM) to fill the missing blank with the corresponding FQN. Our empirical evaluation shows relatively improves over the state-of-the-art FQN resolution approaches by 15% on complete, and 20% on real-world code snippets. Upon further analysis, we were able to tie the performance improvement to our tool’s capabilities to capture dependencies between the API and other program elements in the code snippet.


## Dataset Links:
* Full datasets: https://drive.google.com/drive/folders/1HL95T0HAX2j45HRHS5QrgKmGNZte62IJ?copy
* Model weights: https://drive.google.com/drive/folders/1HL95T0HAX2j45HRHS5QrgKmGNZte62IJ?copy
