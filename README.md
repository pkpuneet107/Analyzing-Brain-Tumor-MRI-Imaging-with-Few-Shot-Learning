# Analyzing Brain Tumor MRI Imaging with Few-Shot Learning
Accurate classification of brain tumors from MRI scans is often hindered by the limited availability of labeled medical data. In this work, we investigate metric- based few-shot learning approaches, Prototypical Networks and Siamese Networks, as solutions for low-data tumor classification scenarios. Using a curated dataset of T1-weighted grayscale brain MRI images across four diagnostic categories, we compare these methods against a standard supervised convolutional neural network (CNN) baseline. The few-shot models are trained and evaluated using an episodic learning framework designed to test generalization to unseen tumor classes. Our findings show that Prototypical Networks offer strong performance under limited supervision, while Siamese Networks are more sensitive to the structure of training pairs. Though the baseline CNN outperforms both few-shot models when ample data is available, our results demonstrate the promise of few- shot learning for reliable medical image classification in data-scarce environments. This work contributes a replicable framework for exploring low-shot diagnostic tools in clinical imaging applications.


- **Prototypical Networks**
- **Siamese Networks**
- **Supervised CNN Baseline**

- ##  Dataset
- [Brain Tumor MRI Dataset](https://doi.org/10.34740/KAGGLE/DSV/2645886)
- 7023 grayscale T1-weighted MRI images
- Classes: `glioma`, `meningioma`, `pituitary tumor`, `no tumor
