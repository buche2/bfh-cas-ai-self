from flask import Flask
app = Flask(__name__)

import rpy2.robjects as robjects

@app.route("/")
def hello_world():
  #low < -c("Serious", "Slow", "Useless", "Tiring", "Old", "Hard", "Long")
  #high < -c("Fun", "Fast", "Useful", "Light", "New", "Easy", "Short")
  #scale < -c("SA", "A", "N", "A", "SA")
  #grp1means < -c(4.2, 4.6, 4.3, 4.1, 4.5, 4.5, 4.0)
  #grp2means < -c(3.8, 3.9, 3.7, 4.5, 4.4, 4.3, 4.4)
  #grp3means < -c(4.5, 4.7, 4.4, 4.2, 4.6, 4.4, 3.9)
  #data < -matrix(
  #  cbind(low, high, grp1means, grp2means, grp3means),
  #  nrow=7, ncol=5, byrow=FALSE,
  #  dimnames=list(c("I1", "I2", "I3", "I4", "I5", "I6", "I7"),
  #                c("Low", "High", "Grp1", "Grp2", "Grp3"))
  #)
  #sdRplot(5, scale, 7, data)
  robjects.r.source("sdRplot.R", encoding="utf-8")


  return "Hello, World!"