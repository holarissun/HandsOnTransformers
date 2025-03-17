
'''
python knowledge about *() tuple unpacking
'''
print('tuple unpacking')
def foo(a, b, c):
    print(a, b, c)
    
foo(*[1, 2, 3])
foo(*(1, 2, 3))

# unpacking with *: any iterable can be unpacked
print([1, *[2, 3], 4])        # get [1, 2, 3, 4]
print([*"hello"])              # get ['h', 'e', 'l', 'l', 'o']

# multi-level unpacking
nested = [(1, 2), (3, 4)]
print([*nested])               # get [(1, 2), (3, 4)]
print([*nested[0], *nested[1]]) # get [1, 2, 3, 4]

# unpacking with different types
print([0, *(1, 2), 3, *{'a': 4, 'b': 5}])  # get [0, 1, 2, 3, 'a', 'b']

# merge lists
list1 = [1, 2]
list2 = [3, 4]
merged = [*list1, *list2]  # [1, 2, 3, 4]

#  dynamical parameters
def func(a, b, c):
    return a + b + c
args = (2, 3)
func(1, *args)  # = func(1, 2, 3)

# unpacking with **: only dict can be unpacked

dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}
merged_dict = {**dict1, **dict2}
print(merged_dict)  # get {'a': 1, 'b': 2, 'c': 3, 'd': 4}

# merge dicts
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
merged = {**dict1, **dict2}  # {'a': 1, 'b': 3, 'c': 4}
# dict2 will overwrite dict1 if they have the same key

# dynamical parameters
def func(a, b, c):
    print(a, b, c)

params = {"a": 1, "b": 2, "c": 3}
func(**params)  # get 1 2 3
