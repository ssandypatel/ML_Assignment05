# Machine Learning Assignment 05 <br>

##   Question 01.  Are the results as expected? Why or why not?
### Answer: In comparison to the VGG model with only one block, the results demonstrate that the VGG model with three blocks (both without and with data augmentation) has a higher training accuracy and a marginally better testing accuracy. This is expected since more complicated features in the data can be captured by deeper models. However, because deeper models need to perform better computation, the training time for VGG models with more blocks is also much longer. High training accuracy and testing accuracy are achieved by the transfer learning model, which is to be expected given that transfer learning uses pre-trained models to take advantage on their learnt characteristics.<br>


##  Question 02. Does data augmentation help? Why or why not?
### Answer:  Comparable training and testing accuracy are shown by the VGG model with 3 blocks plus data augmentation compared to the VGG model with 3 blocks only. This shows that, in this instance, data augmentation may not significantly affect the performance of VGG models, probably as a result of the limited dataset.<br>



## Question 03.  Does it matter how many epochs you fine tune the model? Why or why not?
### Answer: The performance of the model is affected by the number of epochs used for fine-tuning. It is significant to note that the performance of the model is adversely affected by either too few or too many epochs. Overfitting is caused by using too many epochs, and underfitting is caused by using too few. The dataset and model complexity determine the ideal number of epochs for fine-tuning.<br>



##  Question 04.  Are there any particular images that the model is confused about? Why or why not?
### There are numerous images that the model is unsure of based on the test set's images and their predictions. This might be the result of a number of factors, including the fact that both classes of images have similar characteristics or patterns, poor image quality, or a lack of diversity in the training data. The model's performance on these kinds of images might be enhanced by fine-tuning it with more diverse data or by applying techniques like regularisation.<br>


## Question 05 : What can you conclude from the MLP model performance?
### Answer: In comparison to VGG models, the user-defined MLP model of a comparable number of parameters performs better in terms of training accuracy, but it performs poorer in terms of testing accuracy. As evidenced by the testing accuracy, this shows that while a larger MLP model can fit the training data in a better way, it may not necessarily generalise well to unseen data. This emphasises how model architecture, rather than just the quantity of parameters, is crucial for determining model performance.

