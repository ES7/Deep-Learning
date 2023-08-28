# Generative Adversarial Neural Network
Generative modeling is an unsupervised learning task in machine learning that involves automatically discovering and learning the regularities or patterns in input data in such a way that the model can be used to generate or output new examples that plausibly could have been drawn from the original dataset.
We solve this problem as a supervised learning problem using 2 models :- the **generator** model that we train to generate new examples, and the **discriminator** model that tries to classify examples as either real (from the domain) or fake (generated). It treats the unsupervised problem as supervised.

**GANs :-** https://towardsdatascience.com/must-read-papers-on-gans-b665bbae3317 

## Discriminative vs. Generative Modeling
* In supervised learning, we develop a model to predict a class label given an example of input variables. This predictive modeling task is called **classification**. Classification is also traditionally referred to as discriminative modeling. This is because a model must discriminate examples of input variables across classes; it must choose or make a decision as to what class a given example belongs.
* Alternately, unsupervised models that summarize the distribution of input variables may be able to be used to create or generate new examples in the input distribution.
* Navie Bayes is an example of a generative model that is more often used as a discriminative model.

## The Generator Model
* The generator model takes a fixed-length random vector as input and generates a sample in the domain. The vector is drawn from randomly from a Gaussian distribution, and the vector is used to seed the generative process. After training, points in this multidimensional vector space will correspond to points in the problem domain, forming a compressed representation of the data distribution.
* This vector space is referred to as a latent space. Latent variables, or hidden variables, are those variables that are important for a domain but are not directly observable. In the case of GANs, the generator model applies meaning to points in a chosen latent space, such that new points drawn from the latent space can be provided to the generator model as input and used to generate new and different output examples.

## The Discriminator Model
* The discriminator model takes an example from the domain as input (real or generated) and predicts a binary class label of real or fake (generated). The real example comes from the training dataset. The generated examples are output by the generator model. The discriminator is a normal (and well understood) classification model.
* After the training process, the discriminator model is discarded as we are interested in the generator. Sometimes, the generator can be repurposed as it has learned to effectively extract features from examples in the problem domain. Some or all of the feature extraction layers can be used in transfer learning applications using the same or similar input data.

![image](https://github.com/ES7/Deep-Learning/assets/95970293/5c4a9057-3e00-4bc5-a289-be781ad76c3f)

* GANs typically work with image data and use Convolutional Neural Networks, or CNNs, as the generator and discriminator models. The reason for this may be both because the first description of the technique was in the field of computer vision and used CNNs and image data, and because of the remarkable progress that has been seen in recent years using CNNs more generally to achieve state-of-the-art results on a suite of computer vision tasks such as object detection and face recognition.
* Modeling image data means that the latent space, the input to the generator, provides a compressed representation of the set of images or photographs used to train the model. It also means that the generator generates new images or photographs, providing an output that can be easily viewed and assessed by developers or users of the model.

* **Cycle GAN :-** https://machinelearningmastery.com/what-is-cyclegan/ 
* **Pix2Pix GAN :-** https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
