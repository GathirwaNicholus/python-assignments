#loading the ds
#first install pandas before running (can create a new venv)
import pandas as pd
data = pd.read_csv (r"tested.csv")
data.info()

# count of people who survived in each passenger class
#we aggregate the Survived column.
grouped = data.groupby("Pclass")["Survived"].sum() #returns the count where the value is 1 or true
print("Total number of survivors by class:")
print(grouped)

grouped = data.groupby("Pclass")["Survived"].agg("count") #counts the number of columns ie returns all passengers in that class
print("Total number of passengers that boarded by class:")
print(grouped)

grouped = data.groupby("Pclass")["Survived"].agg("mean") #Calculates the mean (average) survival value for each passenger class.
#abit different from conventional mean eg if we have 100 passengers and 60 survivors, the mean would be 0.6 etc.
print("Survival rate by class:")
print(grouped)
