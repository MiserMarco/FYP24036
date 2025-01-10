# Specify the path to the text file
file_path = "/Users/songdelin/Desktop/fyp/nasdaqlisted.txt"  # Change this to your actual file path

# Initialize a list to hold the symbols
symbols = []

# Open the file and read it line by line
with open(file_path, "r") as file:
    for line in file:
        # Split the line by the '|' character
        parts = line.strip().split('|')
        # Check if there are enough parts to get the symbol
        if len(parts) > 0:
            symbols.append(parts[0])  # Append the symbol (first part)
print("len of nsdq", len(symbols))

# Print the extracted symbols
for symbol in symbols:
    print(symbol)
