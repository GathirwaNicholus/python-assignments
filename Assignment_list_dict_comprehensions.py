#assignment
import itertools


people = [
    "Alice", "Bob", "Charlie", "Diana", "Ethan",
    "Fiona", "George", "Hannah", "Isaac", "Julia",
    "Kevin", "Laura", "Michael", "Nina", "Oscar"
]
heights = [
    165, 178, 172, 160, 185,
    170, 182, 158, 174, 169,
    180, 162, 176, 168, 181
]
print (f"Are the lists people and heights equal: ", len(people) == len(heights))

#combining the 2 lists, using indexing with range. (can also use enumerate - see my article)
people_heights_dict = {people[i]:heights[i] for i in range (len(heights))}
print(people_heights_dict)

#function to create sample pairs then calculate sample means
def sample_mean_calc (sample_size, dict):
    #1. Creating the sample pairs using iter tools.
    n = sample_size
    combs = list(itertools.combinations(dict.keys(), n))
    
    #2. calculating the sample means (from the combination, combs above):
    sample_means = []
    for comb in combs:
        #combs only has keys so ->
        sample_weight = [dict[i] for i in comb]
        sample_mean = sum(sample_weight)/n
        sample_means.append(sample_mean)
    
    #storing the combinations and their respective sample means
    combs_sample_means = {key: value for key, value in zip (combs, sample_means)}
    return combs_sample_means

#Calling the function ie passing the sample size and created dictionary
a = sample_mean_calc(2,people_heights_dict)
print(" \n")
print("Combinations\t \tSample means")
for key, value in a.items():
    print(key, "\t", value)

mean_of_sample_means = sum(a.values())/len(a.values())
print("The mean of sample means = ", mean_of_sample_means)