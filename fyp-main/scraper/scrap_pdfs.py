import requests
import shutil
import os

def nasdaq_list():
    # Specify the path to the text file
    file_path = "./nasdaqlisted.txt"  # Change this to your actual file path

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
        
    return symbols


symbols = nasdaq_list()
symbols = symbols[2539:]

print(symbols)


for ticker in symbols:
# URL of the file to download
    number = 0
    url = f"https://www.annualreports.com/HostedData/AnnualReports/PDF/NASDAQ_{ticker}_2023.pdf"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        number += 1
        # Specify the download directory
        download_directory = "../test"  # Change this to your desired path
        os.makedirs(download_directory, exist_ok=True)  # Create the directory if it doesn't exist
    
        # Save the downloaded file
        file_path = os.path.join(download_directory, f"{ticker}.pdf")
        with open(file_path, "wb") as file:
            file.write(response.content)
    
        print(f"Download completed successfully! (file {number})")

    else:
        print(f"Failed to download file. Status code: {response.status_code}")


