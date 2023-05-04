# Reversible Data Hiding

This project implements Reversible Data Hiding using Adaptive IPVO and Two-Segment Pairwise PEE. The aim is to embed user-provided data into an image in a reversible way, allowing the data to be extracted and the image to be restored with minimal distortion.

## Prerequisites

- Python 3.6 or later
- Compatible with Windows, macOS, and Linux operating systems

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/fa1ya2/reversible_data_hiding.git
   ```

2. Change to the `reversible_data_hiding` directory:

   ```
   cd reversible_data_hiding
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare an image to be used for data embedding. The image should be in a format supported by the Python Imaging Library (PIL), such as JPEG, PNG, or BMP. It is recommended to use lossless formats like PNG for better results.

2. Choose the data to be embedded. You can provide the data as a string using the `--data` command-line argument.

3. Run the project using the Command Line Interface (CLI):

   ```
   python cli.py path/to/image.png --data "Your string data to be embedded"
   ```

   Replace `path/to/image.png` with the path to your input image and `"Your string data to be embedded"` with the data you want to embed into the image. You can use single quotes if you prefer.

4. The CLI will output the embedded image, the extracted data, and the restored image in the `output/` directory. The following files will be generated:

   - `embedded_image.png`: The image with the data embedded using Adaptive IPVO and Two-Segment Pairwise PEE algorithms.
   - `extracted_data.txt`: The data extracted from the embedded image.
   - `restored_image.png`: The restored image after data extraction, which should be almost identical to the original image.

5. Check the PSNR (Peak Signal-to-Noise Ratio) value displayed in the terminal. A higher PSNR value indicates better image quality after the embedding and extraction process.

## Testing and Development Guidelines

To run the tests for this project, navigate to the project root directory and execute the following command:

```
pytest
```

This will run all test cases located in the `tests/` directory. Feel free to add more test cases or modify the existing ones as you work on the project.

## Contributing Guidelines

1. Fork the repository on GitHub.
2. Clone your fork and create a new branch for your feature or bugfix.
3. Commit your changes to your branch, following the project's coding standards and adding appropriate test cases.
4. Push your changes to your fork on GitHub.
5. Submit a pull request to the main repository for review.

Please ensure that your code is well-documented, follows best practices, and passes all tests before submitting a pull request.

## License

This project is licensed under the [MIT License](LICENSE).