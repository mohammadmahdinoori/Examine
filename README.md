# Examine
A contrastive approach for measuring the semantic similarity of persian academic questions based on Transformers in deep learning.

# What is Examine
Examine is a platform for measuring the similarity of persian academic questions based on deep learning techniques, mainly Transformers and especially Google's BERT model. In this work we conducted a research on various nlp techniques in the field of text similarity and various techniques in the field of OCR epecially when images include LATEX besides natural language. In addition, we tried to come up with a novel deep learning model for efficient text similarity called Linear Attention Summarizer Model or `LASM`.

![](https://github.com/mohammadmahdinoori/Examine/blob/main/Images/Examine%20Main%20Figure.jpg?raw=true)

# Text Similarity
Text Similarity is the problem of measuring the similarity of given pieces of texts in terms of real-valued scores. Text Similarity is mainly done by learning a meaningful latent represetnation of texts which can be then used for similarity measurement. In other words, we first learn a model which is used to encode the meaning of each piece of text into an n-dimensional vector, and then use common distance functions like cosine-similarity or euclidean distance as the similarity factor for the obtained vectors. In this scenario, similar vectors in the embedding space or latent space represent similar texts and unsimilar vectors represent unsimilar texts in terms of meaning. Overall, we aim to learn a model which can satisfy explained factors.

## Common Models For Text Similarity
With the advent of Transformers and attention-based models, especially google's BERT model, Text Similarity is mostly done by BERT and other similar variations like RoBERTa, DeBERTa and etc. 

## Common Losses For Text Similarity

There are different ways to model the task of Text Similarity, However the most common losses for this task are `Contrastive Losses` which are designed in such way that they make encodings of the similar texts close in the embedding space and encodings of unsimilar texts far in the embedding space. The loss that we are using in our model is as follows:

$Loss = Y \times D^2 + (1 - Y) \times max(Margin - D, 0)^2$

`Y` is the label for the given pair of texts and indicates whether the texts are similar or not. It can be either 1 (similar) or 0 (unsimilar).
<br/>
`D` is the distance between the encodings of the given pair of texts. 
<br/>
`Margin` is the minimum distance between two unsimilar texts. 

# Problems With The Current Methods And Introducing LASM
Since self-attention mecahnism comes with the major disadvantage of Quadratic Complexity, it is not useful for long sequences and since we want our method to be efficient we need to make the complexity linear. Which can be done by various methods like local attention, kernels for decomposing softmax, using global memories, and etc. But for this work we chose to use a simple kernel which is elu(x) + 1 to decompose softmax function and achieve linear complexity. This method was previously introduced in the following [paper](https://arxiv.org/abs/2006.16236). So this is the first improvement in LASM.

For the second improvment we tried to use multiple `[CLS]` (pooler) tokens to form a versetile global memory while maintaining the linear complexity of the new attention mechanism. As a result, instead of relying on the pooled representation obtained by only one token we use n tokens to obtain a summary of the given text which tends to be more accurate.

![](https://github.com/mohammadmahdinoori/Examine/blob/main/Images/LASM%20Main%20Figure.jpg?raw=true)

# How LASM Works

[Watch our video](https://drive.google.com/file/d/1PWQMLPVo9dh3sam2lc8YEeaIAH6SRqYF/view?usp=sharing)

# OCR
For the OCR section of our work, we are mainly relying on the open source tesseract library. However, we are aiming to build our own ocr models based on Vision Transformers and try to train it jointly with our LASM model so they can match pretty well together and make the process of turining images into texts easier.

However as mentioned before, There is a problem in our context which is that images might contain LATEX as well and it is super hard for a deep learning model to accurately convert images that contain both natural language and LATEX to text. Threfore, we found a model called FI-FO which is designed to find Figures and Formulas in an image and by using this model we will be able to convert images of natural language and LATEX seprately to their corresponding text and learn different models that are specialized in one of the areas.

# Visualizing Embedding Space

This is the embedding space before training:

![](https://github.com/mohammadmahdinoori/Examine/blob/main/Images/Embeddings%20Before%20Training.png?raw=true)

This is the embedding space after training:

![](https://github.com/mohammadmahdinoori/Examine/blob/main/Images/Embeddings%20After%20Training.png?raw=true)
