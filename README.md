# Bayesnet-Backpropagation
Basic R code which use backpropagation to learn parameters in a Bayesian Network. This is more a demonstration/tutorial rather than a usable library. It assumes that you know how backpropagation works in neural networks/some other variety of model.

This repo is a basic demonstration of using backpropagation to learn CPT values in a Bayesian Network. I have used PDF images of networks from Netica to visualize the before and after results, however it is not nescessary to have Netica to execute the code.

We start with the Bayesian Network in the file ......

![image](https://user-images.githubusercontent.com/35029869/109891571-6d66c380-7cdd-11eb-8464-ac15d13d7fff.png)

However, we later go to validate the network against observed data and find that the probabilities associated with the nodes for Wet_Grass and Take_Off_From_Work are incorrect. We find that the probability for Wet_Grass is 60%, and the probability for Take_Off_From_Work is 10%. As such, we update the ground truth values in the R code.

![image](https://user-images.githubusercontent.com/35029869/109891824-df3f0d00-7cdd-11eb-9f25-5b6c3e6999b9.png)

And then execute:

![image](https://user-images.githubusercontent.com/35029869/109893396-b455b880-7cdf-11eb-9486-e3b546deea6a.png)

Finally, I copied the new values from the CPTs into the Netica network, which now results in the probabilities below. Note that there are a few differences as Netica requires significant rounding when compared to the CPT's produced by the R code.

![image](https://user-images.githubusercontent.com/35029869/109893611-11ea0500-7ce0-11eb-8c4b-c423480f6a9d.png)

Thanks for reading :)
