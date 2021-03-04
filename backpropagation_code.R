
scalar_vector_multiply <- function(X,b){
  return(X*b)
}

#Setup CPT values
Windy <- matrix(data = c(0.001, 0.999), ncol = 2, nrow = 1)
Cloudy <- matrix(data = c(0.002, 0.001, 0.1, 0.05, 0.998,0.999,0.9,0.95),
                 ncol = 2, nrow = 4, byrow = FALSE)
El_Nino <- matrix(data = c(0.40, 0.60), ncol = 2, nrow = 1)
Seeded <- matrix(data = c(0.001, 0.999), ncol = 2, nrow = 1)
Rain <-matrix(data = c(0.95,0.95,0.29,0.001, 0.05,0.05,0.71,0.999),
              ncol = 2, nrow = 4, byrow = FALSE)
Wet_Grass <- matrix(data = c(0.95, 0.05, 0.05, 0.95), ncol = 2, nrow = 2,
                    byrow = FALSE)
Take_Off_From_Work <- matrix(data = c(0.7, 0.01, 0.3, 0.99), ncol = 2, nrow = 2,
                             byrow = FALSE)

q = 150
z = 0

#TODO:establish learning rate decay and early stopping/checkpointing as future
#features
while(z < q){

  #Set probabilities of root nodes
  Windy_Probs <-  Windy
  El_Nino_Probs <- El_Nino
  Seeded_Probs <- Seeded

  #Calculate probability of it being Cloudy
  a <- c()
  for(prob in El_Nino_Probs){
    a <- c(a, prob * Seeded_Probs)
  }
  Cloudy_true <- sum(a * Cloudy[,1])
  Cloudy_false <- sum(a * Cloudy[,2])
  Cloudy_Probs <- matrix(c(Cloudy_true, Cloudy_false), ncol = 2, nrow = 1)

  #Calculate probability of it Raining
  a <- c()
  for(prob in Windy_Probs){
    a <- c(a, prob * Cloudy_Probs)
  }
  Rain_true <- sum(a * Rain[,1])
  Rain_false <- sum(a * Rain[,2])
  Rain_Probs <- matrix(c(Rain_true, Rain_false), ncol = 2, nrow = 1)

  #Calculate probability of outputs
  Wet_Grass_Probs <- c(sum(Rain_Probs * Wet_Grass[,1]),
                       sum(Rain_Probs * Wet_Grass[,2]))

  Wet_Grass_Probs <- matrix(data = Wet_Grass_Probs, ncol = 2, nrow = 1)

  Take_Off_From_Work_Probs <- c(sum(Rain_Probs * Take_Off_From_Work[,1]),
                                sum(Rain_Probs * Take_Off_From_Work[,2]))

  Take_Off_From_Work_Probs <- matrix(data = Take_Off_From_Work_Probs,
                                     ncol = 2, nrow = 1)

  print("Take off from work probabilities are:")
  print(Take_Off_From_Work_Probs)
  print("Wet Grass probabilities are:")
  print(Wet_Grass_Probs)

  #Now lets calculate the loss total. First specify the ground truths
  gt_Wet_Grass_Probs <- matrix(data = c(.6, 0.4), ncol = 2, nrow = 1)
  gt_Take_Off_From_Work_Probs <- matrix(data = c(0.1, 0.9), ncol = 2, nrow = 1)

  Wet_Grass_Loss_True <- .5*(gt_Wet_Grass_Probs[1] - Wet_Grass_Probs[1])**2
  Wet_Grass_Loss_False <- .5*(gt_Wet_Grass_Probs[2] - Wet_Grass_Probs[2])**2
  Take_Off_From_Work_True <- .5*(gt_Take_Off_From_Work_Probs[1] -
                                Take_Off_From_Work_Probs[1])**2
  Take_Off_From_Work_False <- .5*(gt_Take_Off_From_Work_Probs[2] -
                                 Take_Off_From_Work_Probs[2])**2

  total_loss <- (Wet_Grass_Loss_False + Wet_Grass_Loss_True +
    Take_Off_From_Work_True + Take_Off_From_Work_False)

  print("Total loss is:")
  print(total_loss)

  #Step by step back-propgation for the individual CPT values

  #Step 1: Derivative of the loss functions
  delta_Wet_Grass_Loss_True <- -1 * (gt_Wet_Grass_Probs[1] - Wet_Grass_Probs[1])
  delta_Wet_Grass_Loss_False <- -1 * (gt_Wet_Grass_Probs[2] - Wet_Grass_Probs[2])
  delta_Take_Off_From_Work_True <-
    -1 * (gt_Take_Off_From_Work_Probs[1] - Take_Off_From_Work_Probs[1])
  delta_Take_Off_From_Work_False <-
  -1 * (gt_Take_Off_From_Work_Probs[2] - Take_Off_From_Work_Probs[2])

  delta_Wet_Grass_loss <- matrix(
    c(delta_Wet_Grass_Loss_True, delta_Wet_Grass_Loss_False),
    ncol = 2, nrow = 1)
  delta_Take_Off_From_Work_loss <- matrix(
    c(delta_Take_Off_From_Work_True, delta_Take_Off_From_Work_False),
    ncol = 2, nrow = 1)

  #Step 2: Derivatives of Output Nodes themselves
  delta_wet_grass <- c(Rain_Probs[1] * delta_Wet_Grass_loss[1],
                       Rain_Probs[2] * delta_Wet_Grass_loss[1],
                       Rain_Probs[1] * delta_Wet_Grass_loss[2],
                       Rain_Probs[2] * delta_Wet_Grass_loss[2])

  delta_wet_grass <- matrix(delta_wet_grass, nrow = 2, ncol = 2, byrow = FALSE)

  delta_Take_Off_From_Work <- c(Rain_Probs[1] * delta_Take_Off_From_Work_loss[1],
                                Rain_Probs[2] * delta_Take_Off_From_Work_loss[1],
                                Rain_Probs[1] * delta_Take_Off_From_Work_loss[2],
                                Rain_Probs[2] * delta_Take_Off_From_Work_loss[2])

  delta_Take_Off_From_Work <- matrix(delta_Take_Off_From_Work,
                                     nrow = 2, ncol = 2, byrow = FALSE)

  #Step 3: Derivatives of it Raining:
  delta_prev_rain_true <- sum(delta_wet_grass[,1]) +
    sum(delta_Take_Off_From_Work[,1])

  delta_prev_rain_false <- sum(delta_wet_grass[,2]) +
    sum(delta_Take_Off_From_Work[,2])

  delta_raining <- c(Windy_Probs[1] * Cloudy_Probs[1] * delta_prev_rain_true,
                     Windy_Probs[2] * Cloudy_Probs[1] * delta_prev_rain_true,
                     Windy_Probs[1] * Cloudy_Probs[2] * delta_prev_rain_true,
                     Windy_Probs[2] * Cloudy_Probs[2] * delta_prev_rain_true,
                     Windy_Probs[1] * Cloudy_Probs[1] * delta_prev_rain_false,
                     Windy_Probs[2] * Cloudy_Probs[1] * delta_prev_rain_false,
                     Windy_Probs[1] * Cloudy_Probs[2] * delta_prev_rain_false,
                     Windy_Probs[2] * Cloudy_Probs[2] * delta_prev_rain_false)

  delta_raining <- matrix(delta_raining,
                          nrow = 4, ncol = 2, byrow = FALSE)

  #Step 4: Derivatives of it being Cloudy:
  delta_prev_cloudy_true <- sum(delta_raining[1:2,1:2])
  delta_prev_cloudy_false <- sum(delta_raining[3:4,1:2])

  delta_cloudy <- c(El_Nino_Probs[1] * Seeded_Probs[1] * delta_prev_cloudy_true,
                    El_Nino_Probs[1] * Seeded_Probs[2] * delta_prev_cloudy_true,
                    El_Nino_Probs[2] * Seeded_Probs[1] * delta_prev_cloudy_true,
                    El_Nino_Probs[2] * Seeded_Probs[2] * delta_prev_cloudy_true,
                    El_Nino_Probs[1] * Seeded_Probs[1] * delta_prev_cloudy_false,
                    El_Nino_Probs[1] * Seeded_Probs[2] * delta_prev_cloudy_false,
                    El_Nino_Probs[2] * Seeded_Probs[1] * delta_prev_cloudy_false,
                    El_Nino_Probs[2] * Seeded_Probs[2] * delta_prev_cloudy_false)

  delta_cloudy <- matrix(delta_cloudy,
                         nrow = 4, ncol = 2, byrow = FALSE)

  #Step 5: Derivatives of Input Nodes:

  delta_prev_windy_true <- sum(delta_raining[1]) + sum(delta_raining[3])
  delta_prev_windy_false <- sum(delta_raining[2]) + sum(delta_raining[4])
  delta_windy <- c(delta_prev_windy_true, delta_prev_windy_false)

  delta_prev_El_Nino_true <- sum(delta_cloudy[1:2,1:2])
  delta_prev_El_Nino_false <- sum(delta_cloudy[3:4,1:2])
  delta_El_Nino <- c(delta_prev_El_Nino_true, delta_prev_El_Nino_false)

  delta_prev_seeded_true <- sum(delta_cloudy[1]) + sum(delta_cloudy[3])
  delta_prev_seeded_false <- sum(delta_cloudy[2]) + sum(delta_cloudy[4])
  delta_seeded <- c(delta_prev_seeded_true, delta_prev_seeded_false)

  #Set learning rate
  learn_rate = .1

  #Ok, time to update all the CPT's!!
  Windy <- Windy - (delta_windy * learn_rate)
  Cloudy <- Cloudy - (delta_cloudy * learn_rate)
  El_Nino <- El_Nino - (delta_El_Nino * learn_rate)
  Seeded <- Seeded - (delta_seeded * learn_rate)
  Rain <- Rain - (delta_raining * learn_rate)
  Wet_Grass <- Wet_Grass - (delta_wet_grass * learn_rate)
  Take_Off_From_Work <- Take_Off_From_Work - (delta_Take_Off_From_Work
                                              * learn_rate)

  z <- z + 1
  #Renormalize back to correct probability distribution
  Windy<- Windy/(sum(Windy))
  Seeded <- Seeded/(sum(Seeded))
  El_Nino <- El_Nino/(sum(El_Nino))
  for(i in 1:4){
    Cloudy[i,] <- abs(Cloudy[i,])/(sum(abs(Cloudy[i,])))
  }
  for(i in 1:4){
    Rain[i,] <- abs(Rain[i,])/(sum(abs(Rain[i,])))
  }
  for(i in 1:2){
    Wet_Grass[i,] <- abs(Wet_Grass[i,])/(sum(abs(Wet_Grass[i,])))
  }
  for(i in 1:2){
    Take_Off_From_Work[i,] <- abs(Take_Off_From_Work[i,])/
      (sum(abs(Take_Off_From_Work[i,])))
  }
}



