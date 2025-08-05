import re

def extract_steps_and_loss(input_file_path, output_file_path):
    try:
        # Open input file for reading
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            # Read all lines
            lines = input_file.readlines()
        
        # Prepare data for output
        output_data = []
        for line in lines:
            # Extract step and loss using regex
            match = re.match(r'Sample (\d+) \| Loss: (\d+\.\d+)', line)
            if match:
                step = match.group(1)
                loss = match.group(2)
                output_data.append(f"{step}, {loss}")
        
        # Write to output file
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            # Write header
            output_file.write("step_name, loss\n")
            # Write data
            for data in output_data:
                output_file.write(f"{data}\n")
        
        print(f"Successfully created output file: {output_file_path}")
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file_path}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
input_file_path = r"E:\newtacotron\tacotron\our_code\my_logs\loss_16k.log"  # Replace with your input file path
output_file_path = r"E:\newtacotron\tacotron\our_code\step_loss(16k).txt"  # Replace with your desired output file path
extract_steps_and_loss(input_file_path, output_file_path)