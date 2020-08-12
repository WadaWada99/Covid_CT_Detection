# Covid_CT_Detection
### Utilizing Tensor Flow and Keras in training a neural network to distinguish Normal CT scans from Covid-19 CT scans.

Data Sourced From:https://ieee-dataport.org/documents/ccap

### Sample Images:

##### Normal Lung CT Scan:

![IMG-0001-00016](https://user-images.githubusercontent.com/60201899/89960570-66774700-dc0d-11ea-8847-9bd8d0e5b40d.jpg)

##### COVID-19 Infected Lung CT Scan:

![IMG-0162-00042](https://user-images.githubusercontent.com/60201899/89960581-6c6d2800-dc0d-11ea-8f51-f59f5a9380fb.jpg)

### Utlized 4 hidden layer of Rectified Linear Units and a Sigmoid Layer for output (example of layer 1):

![7](https://user-images.githubusercontent.com/60201899/89960864-1482f100-dc0e-11ea-81cd-0533c78406de.PNG)

![3](https://user-images.githubusercontent.com/60201899/89960605-798a1700-dc0d-11ea-9fd7-af685f793932.PNG)

### Adding a dropout of 50% per hidder layer and a early stopping helped the relationship between loss and validation loss

##### Before adding dropout and early stopping:

![2](https://user-images.githubusercontent.com/60201899/89960606-7b53da80-dc0d-11ea-8dc6-7b798984d4b6.png)

##### After adding dropout and early stopping, you notice that with these features added, as the loss decrease so does the validation loss:

![4](https://user-images.githubusercontent.com/60201899/89960610-7db63480-dc0d-11ea-91cb-b428f46a2f1b.png)

### Epoch 1:

![5](https://user-images.githubusercontent.com/60201899/89960649-99213f80-dc0d-11ea-8f8d-1e5bbde4e010.PNG)

### Last Epoch Run:

![6](https://user-images.githubusercontent.com/60201899/89960650-99b9d600-dc0d-11ea-8f73-5ad0c4a6cb4a.PNG)

The models accuracy increases as the Epochs increase which is a given. I accouted for over fitting with drop outs and early stoppings. Because this is a binary model (0 = COVID, 1 = Normal) it returns 0 and 1

# Conclusion the model it self performs well and can distinguish between Normal Lung CT scan vs Covid Lung CT Scan with great accuracy. 

Future adjustments:
  - Train Model on Mycoplasma Pneumoniae, and Viral
