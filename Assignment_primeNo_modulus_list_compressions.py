# List comprehension to get prime numbers from 0 to 10
prime_numbers = [i for i in range(2, 11) if all(i % j != 0 for j in range(2, i))]
print("Prime numbers:", prime_numbers)

# List comprehension to get numbers divisible by 3 from 0 to 10
modulus_by_3 = [i for i in range(0, 11) if i % 3 == 0]
print("Numbers divisible by 3:", modulus_by_3)