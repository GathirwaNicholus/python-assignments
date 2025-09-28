#list assignment - remove banana from the list below.
fruits = ["apples", "banana", "cherry"]
fruits.append("oranges")
print(fruits)

#remove banana
fruits.remove("banana")
print(fruits)

#lamda assignment
#Q. week 6 day 1
#write a python function that checks whether a number is even or odd
#write the function above as a lamda function as well.
def evenOdd(x):
    if (x%2 == 0):
        print("The number", x,"is even")
    else:
        print("The number", x,"is odd")

#testing the function above
evenOdd(3) #prints "This number is odd"
evenOdd(4) #prints "This number is even"

#Writing the function above as a lamda function
evenOdd_lambda = lambda x: x % 2 == 0 

#testing the function above
x = 3
print("Is this", x, "an even number =>", evenOdd_lambda(x))