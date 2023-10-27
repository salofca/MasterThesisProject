def find_most_common_binary_string(binary_strings):
    # Create a dictionary to store the count of each binary string
    count_dict = {}

    # Iterate through the list of binary strings
    for binary_string in binary_strings:
        # If the binary string is already in the dictionary, increment its count
        if binary_string in count_dict:
            count_dict[binary_string] += 1
        else:
            # If the binary string is not in the dictionary, add it with a count of 1
            count_dict[binary_string] = 1

    # Find the binary string with the maximum count
    most_common_string = max(count_dict, key=count_dict.get)

    return most_common_string


# Input list of binary strings
binary_strings = ["001", "110", "001", "101", "110", "001", "001"]

# Find the most common binary string
result = find_most_common_binary_string(binary_strings)

# Output the result
print("Most common binary string:", result)
